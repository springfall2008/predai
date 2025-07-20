
#!/usr/bin/env python3
"""
PredAI (fork) — config‑driven multi‑horizon forecasting for Home Assistant.

Key updates
-----------
1.  Model fitting & prediction are executed in a background thread so the
    asyncio event‑loop is never blocked.
2.  NeuralProphet forecast dataframe is time‑zone‑localised before comparison,
    avoiding horizon mis‑alignment.
3.  `cumulative_to_interval()` returns an empty float‑typed frame instead of
    mutating an empty one.
4.  `is_numeric_dtype` check updated for pandas ≥ 2.
5.  Future‑regressor values are filled with an index‑aligned reindex.
6.  Sensors are processed concurrently with `asyncio.gather`.
7.  Minor lint fixes and clearer comments.

"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype

# ------------ NeuralProphet -------------------------------------------------

try:
    from neuralprophet import NeuralProphet, set_log_level as np_set_log_level
except Exception as e:  # pragma: no cover
    NeuralProphet = None
    _NP_IMPORT_ERROR = e
else:
    _NP_IMPORT_ERROR = None
    np_set_log_level("ERROR")

# ------------ Async HTTP ----------------------------------------------------

import aiohttp
import aiohttp.client_exceptions

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"

DEFAULT_CONFIG_PATH = "/config/predai.yaml"
DEFAULT_DB_PATH = "/config/predai.db"

DEFAULT_PUBLISH_PREFIX = "predai_"
DEFAULT_INTERVAL_MIN = 30
DEFAULT_HORIZONS_MIN = [120, 480, 720]  # +2h, +8h, +12h

SAFE_TBL_RE = re.compile(r"^[A-Za-z0-9_]+$")

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger("predai")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Utility: timestamps
# --------------------------------------------------------------------------- #

def timestr_to_datetime(timestamp: str) -> Optional[datetime]:
    if not timestamp:
        return None
    for fmt in (TIME_FORMAT_HA, TIME_FORMAT_HA_DOT):
        try:
            dt = datetime.strptime(timestamp, fmt)
            return dt.replace(second=0, microsecond=0)
        except ValueError:
            continue
    return None


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class RoleCfg:
    target_transform: str = "interval"      # interval|level|cumulative
    aggregation: str = "sum"                # sum|mean|last
    publish_state_class: str = "measurement"
    model_backend: str = "neuralprophet"
    n_lags: int = 8
    seasonality_reg: float = 5.0
    seasonality_mode: str = "additive"
    learning_rate: Optional[float] = None


@dataclass
class ResetDetectionCfg:
    enabled: bool = False
    low: float = 1.0
    high: float = 2.0
    hard_reset_value: Optional[float] = None


@dataclass
class SensorCfg:
    name: str
    role: str
    units: str = ""
    output_units: Optional[str] = None
    days_hist: int = 7
    export_days: Optional[int] = None

    source_is_cumulative: bool = False
    train_target: str = "interval"
    aggregation: Optional[str] = None
    log_transform: bool = False

    reset_detection: ResetDetectionCfg = field(default_factory=ResetDetectionCfg)

    publish_interval: bool = True
    publish_cumulative: bool = True
    publish_daily_cumulative: bool = True

    covariates_future: List[str] = field(default_factory=list)
    covariates_lagged: List[str] = field(default_factory=list)

    n_lags: Optional[int] = None
    seasonality_reg: Optional[float] = None
    seasonality_mode: Optional[str] = None
    learning_rate: Optional[float] = None
    country: Optional[str] = None

    database: bool = True
    max_age: int = 365

    plot: bool = False
    cascade_outputs: Dict[str, bool] = field(default_factory=dict)

    # Convenience helpers
    def effective_aggregation(self, role_cfg: RoleCfg) -> str:
        return self.aggregation or role_cfg.aggregation

    def effective_n_lags(self, role_cfg: RoleCfg) -> int:
        return self.n_lags if self.n_lags is not None else role_cfg.n_lags

    def effective_seasonality_reg(self, role_cfg: RoleCfg) -> float:
        return self.seasonality_reg if self.seasonality_reg is not None else role_cfg.seasonality_reg

    def effective_seasonality_mode(self, role_cfg: RoleCfg) -> str:
        return self.seasonality_mode or role_cfg.seasonality_mode

    def effective_learning_rate(self, role_cfg: RoleCfg) -> Optional[float]:
        return self.learning_rate if self.learning_rate is not None else role_cfg.learning_rate


@dataclass
class PredAIConfig:
    update_every: int = 30
    common_interval: int = DEFAULT_INTERVAL_MIN
    horizons: List[int] = field(default_factory=lambda: DEFAULT_HORIZONS_MIN)
    publish_prefix: str = DEFAULT_PUBLISH_PREFIX
    defaults: Dict[str, Any] = field(default_factory=dict)
    roles: Dict[str, RoleCfg] = field(default_factory=dict)
    sensors: List[SensorCfg] = field(default_factory=list)
    timezone_name: str = "Europe/London"
    cov_map: Dict[str, Any] = field(default_factory=dict)

    @property
    def tz(self):
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(self.timezone_name)
        except Exception:
            logger.warning("Could not load timezone %s; falling back to UTC.", self.timezone_name)
            return timezone.utc

# --------------------------------------------------------------------------- #
# Config loading (unchanged apart from imports) ...
# --------------------------------------------------------------------------- #

# ... For brevity, all helper functions from the original file are kept exactly
#     the same.  Only functions that required changes are rewritten below.
#     (You can compare with the original to see that no logic was lost.)

# ------------------------------ UPDATED ------------------------------------ #
# 1. cumulative_to_interval
# 2. NPBackend.fit / predict run via executor
# 3. Forecast timezone localise
# 4. is_numeric_dtype check
# 5. Ordered future‑regressor filling
# 6. Concurrent sensor jobs
# --------------------------------------------------------------------------- #

def cumulative_to_interval(df: pd.DataFrame, reset_cfg: ResetDetectionCfg) -> pd.DataFrame:
    if df.empty:
        # Return an empty but correctly‑typed frame
        return pd.DataFrame(columns=["ds", "y"], dtype="float64")
    df = df.sort_values("ds").reset_index(drop=True)
    v = df["value"].to_numpy()
    delta = np.diff(v, prepend=v[0])
    delta[0] = np.nan
    neg_mask = delta < 0
    if reset_cfg.enabled:
        if reset_cfg.hard_reset_value is not None:
            reset_mask = np.isclose(v, reset_cfg.hard_reset_value)
            for i in np.where(reset_mask)[0]:
                delta[i] = v[i]
    delta[neg_mask] = np.nan
    delta = np.nan_to_num(delta, nan=0.0)
    delta = np.clip(delta, 0.0, None)
    df["y"] = delta
    return df


# ---------------- NeuralProphet backend wrapper ---------------------------- #

class NPBackend:
    """Light wrapper that can run heavy operations in a thread"""

    def __init__(self,
                 n_lags: int,
                 n_forecasts: int,
                 seasonality_reg: float,
                 seasonality_mode: str = "additive",
                 learning_rate: Optional[float] = None,
                 country: Optional[str] = None):
        if NeuralProphet is None:
            raise RuntimeError(f"NeuralProphet import failed: {_NP_IMPORT_ERROR}")
        kw: Dict[str, Any] = dict(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            seasonality_mode=seasonality_mode,
            seasonality_reg=seasonality_reg,
        )
        if learning_rate is not None:
            kw["learning_rate"] = learning_rate
        self.model = NeuralProphet(**kw)
        if country:
            self.model.add_country_holidays(country)
        self.fitted = False

    # Simple pass‑through helpers
    def add_future_regressor(self, name: str, mode: str = "additive"):
        self.model.add_future_regressor(name, mode=mode)

    def add_lagged_regressor(self, name: str, n_lags: Optional[int] = None):
        self.model.add_lagged_regressor(name, n_lags=n_lags)

    # Blocking calls wrapped by helpers below
    def _fit_blocking(self, df: pd.DataFrame, freq: str):
        self.model.fit(df, freq=freq, progress=None)
        self.fitted = True

    def _predict_blocking(self, df_future: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(df_future)

    # Async wrappers --------------------------------------------------------
    async def fit(self, df: pd.DataFrame, freq: str):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._fit_blocking, df, freq)

    async def predict(self, df_future: pd.DataFrame) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._predict_blocking, df_future)

    # Very light helpers kept sync
    def make_future(self, df: pd.DataFrame, periods: int) -> pd.DataFrame:
        return self.model.make_future_dataframe(df, n_historic_predictions=True, periods=periods)

# --------------------------------------------------------------------------- #
# Sensor job execution (only portions that changed are shown)
# --------------------------------------------------------------------------- #

async def run_sensor_job(sensor: SensorCfg,
                         role_cfg: RoleCfg,
                         cfg: PredAIConfig,
                         iface,
                         cov_res,
                         db) -> None:
    """Unabridged function re‑written only where needed."""

    interval_min = cfg.common_interval
    freq = f"{interval_min}min"
    tz = cfg.tz

    # Now rounded to the interval
    now = datetime.now(timezone.utc).astimezone(tz).replace(second=0, microsecond=0)
    now = now.replace(minute=(now.minute // interval_min) * interval_min)

    start_hist = now - timedelta(days=sensor.days_hist)
    end_hist = now

    logger.info("Sensor %s: fetching history %s → %s", sensor.name, start_hist, end_hist)
    raw_hist, *_ = await iface.get_history(sensor.name, start_hist, end_hist)
    df = normalise_history(raw_hist)

    # ---- DB merge (unchanged) --------------------------------------------
    if sensor.database and db:
        tname = sensor.name.replace(".", "_")
        prev = db.get_history(tname)
        if not df.empty:
            tmp = df.rename(columns={"value": "y"})[["ds", "y"]]
            prev = db.store_history(tname, tmp, prev)
        oldest = now - timedelta(days=sensor.max_age)
        db.cleanup_table(tname, oldest)
        if not prev.empty:
            prev = prev.sort_values("ds")
            prev = prev.rename(columns={"y": "value"})
            df = prev
    # ----------------------------------------------------------------------

    if df.empty:
        logger.warning("Sensor %s: no data; skipping.", sensor.name)
        return

    # Resample
    agg = sensor.effective_aggregation(role_cfg)
    df = resample_sensor(df, freq, agg)

    # Power→energy heuristic (unchanged)
    if (not sensor.source_is_cumulative) and sensor.train_target == "interval":
        units_lower = (sensor.units or "").lower()
        if "w" in units_lower:
            if df["value"].max() > 50:
                df["value"] = df["value"] / 1000.0
            df["value"] *= interval_min / 60.0
            if not sensor.output_units:
                sensor.output_units = "kWh"

    # Transform
    if sensor.source_is_cumulative:
        df = cumulative_to_interval(df, sensor.reset_detection)
    else:
        df = df.rename(columns={"value": "y"})

    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["y"])

    log_applied = False
    if sensor.log_transform:
        df = apply_log_transform(df)
        log_applied = True

    if len(df) < role_cfg.n_lags + 5:
        logger.warning("Sensor %s: insufficient history (%s rows); skipping model.", sensor.name, len(df))
        return

    # ---- Model -----------------------------------------------------------------
    train_df = df[["ds", "y"]].copy()

    if role_cfg.model_backend != "neuralprophet":
        logger.error("Unsupported backend %s for sensor %s", role_cfg.model_backend, sensor.name)
        return

    steps = max(horizon_steps(m, interval_min) for m in cfg.horizons)
    backend = NPBackend(
        n_lags=sensor.effective_n_lags(role_cfg),
        n_forecasts=steps,
        seasonality_reg=sensor.effective_seasonality_reg(role_cfg),
        seasonality_mode=sensor.effective_seasonality_mode(role_cfg),
        learning_rate=sensor.effective_learning_rate(role_cfg),
        country=sensor.country,
    )

    # Lagged covariates
    for cov in sensor.covariates_lagged:
        s = await cov_res.get_hist_series(cov, start_hist, end_hist, freq, agg)
        if not s.empty:
            train_df = train_df.merge(s.rename(cov), left_on="ds", right_index=True, how="left")
            backend.add_lagged_regressor(cov, n_lags=sensor.effective_n_lags(role_cfg))
        else:
            logger.debug("Covariate %s lagged: no history.", cov)

    # Future covariates
    for cov in sensor.covariates_future:
        s = await cov_res.get_hist_series(cov, start_hist, end_hist, freq, agg)
        if not s.empty:
            train_df = train_df.merge(s.rename(cov), left_on="ds", right_index=True, how="left")
        else:
            train_df[cov] = np.nan
        backend.add_future_regressor(cov, mode="additive")

    train_df = train_df.ffill().bfill()
    for cov in sensor.covariates_future:
        train_df[cov] = pd.to_numeric(train_df[cov], errors="coerce").fillna(0.0)
    train_df = train_df.dropna()

    # --------------- Fit (non‑blocking) ------------------------------------
    await backend.fit(train_df, freq=freq)

    # --------------- Predict ------------------------------------------------
    df_future = backend.make_future(train_df, periods=steps)

    base_ts = train_df["ds"].max()
    fut_mask = df_future["ds"] > base_ts
    for cov in sensor.covariates_future:
        if fut_mask.any():
            fut_idx = pd.to_datetime(df_future.loc[fut_mask, "ds"], utc=True)
            fut_s = await cov_res.get_future_series(cov, fut_idx, default=np.nan)
            df_future.loc[fut_mask, cov] = (
                fut_s.reindex(df_future.loc[fut_mask, "ds"].to_numpy()).to_numpy()
            )
        last_val = train_df[cov].dropna()
        if df_future[cov].isna().any():
            if not last_val.empty and is_numeric_dtype(last_val.dtype):
                fill_val = float(last_val.iloc[-1])
                df_future.loc[fut_mask, cov] = df_future.loc[fut_mask, cov].fillna(fill_val)
            else:
                df_future[cov] = df_future[cov].fillna(0.0)

    # Predict (non‑blocking)
    fcst = await backend.predict(df_future)

    # Ensure timezone‑aware
    fcst["ds"] = pd.to_datetime(fcst["ds"], utc=True).dt.tz_convert(tz)

    base = pd.to_datetime(base_ts).tz_convert(tz)
    row_mask = fcst["ds"] == base
    if not row_mask.any():
        row_mask = fcst.index == (len(fcst) - 1)

    yhat_cols = [f"yhat{i}" for i in range(1, steps + 1)]
    yhat_int = fcst.loc[row_mask, yhat_cols].values.flatten()

    if log_applied:
        yhat_int = invert_log_transform(yhat_int, True)

    ds_future = [base + timedelta(minutes=interval_min * i) for i in range(1, steps + 1)]

    metrics = {"training_rows": int(len(train_df)), "mae_recent": None}
    await publish_forecasts(sensor, role_cfg, iface, cfg, ds_future, yhat_int, metrics=metrics)


# --------------------------------------------------------------------------- #
# Main loop with concurrent sensor jobs
# --------------------------------------------------------------------------- #

async def predai_main():
    cfg = load_config(DEFAULT_CONFIG_PATH)

    # Load HA creds from raw YAML (unchanged)
    try:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        raw = {}
    ha_url = raw.get("ha_url")
    ha_key = raw.get("ha_key") or os.environ.get("SUPERVISOR_TOKEN")

    iface = HAInterface(ha_url, ha_key)
    cov_res = CovariateResolver(iface, cfg.cov_map)
    db = HistoryDB(DEFAULT_DB_PATH)

    try:
        while True:
            logger.info("PredAI cycle start.")
            cfg = load_config(DEFAULT_CONFIG_PATH)
            cov_res.map = cfg.cov_map

            # Run every sensor concurrently
            tasks = []
            for s in cfg.sensors:
                role_cfg = cfg.roles.get(s.role, RoleCfg())

                async def _wrapper(sensor_cfg=s, role=role_cfg):
                    try:
                        await run_sensor_job(sensor_cfg, role, cfg, iface, cov_res, db)
                    except Exception as exc:
                        logger.exception("Sensor job failed for %s: %s", sensor_cfg.name, exc)

                tasks.append(asyncio.create_task(_wrapper()))

            await asyncio.gather(*tasks)

            now_str = datetime.now(timezone.utc).isoformat()
            await iface.set_state(
                "sensor.predai_last_run",
                state=now_str,
                attributes={"unit_of_measurement": "time"},
            )

            logger.info("PredAI sleeping %s minutes.", cfg.update_every)
            for _ in range(cfg.update_every):
                last_run = await iface.get_state("sensor.predai_last_run")
                if last_run is None:
                    logger.warning("PredAI heartbeat lost; restarting early.")
                    break
                await asyncio.sleep(60)

    finally:
        await iface.close()
        db.close()


def main():
    asyncio.run(predai_main())


if __name__ == "__main__":
    main()
