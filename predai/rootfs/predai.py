#!/usr/bin/env python3
"""
PredAI (fork) — config‑driven multi‑horizon forecasting for Home Assistant.

Key features:
* Configurable sensor roles (energy counters, temperature, Mixergy immersion demand, etc.).
* Automatic power→energy integration for non‑cumulative power sensors (mean power -> kWh/interval).
* Covariate alias + scaling from YAML `covariates:` section.
* Interval, cumulative, daily_cum, and horizon (+2h/+8h/+12h or custom) forecast publishing.
* Async Home Assistant API via aiohttp.
* SQLite history cache.
* Config hot‑reload each cycle.

British spelling used in comments.
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

# NeuralProphet
try:
    from neuralprophet import NeuralProphet, set_log_level as np_set_log_level
except Exception as e:  # pragma: no cover
    NeuralProphet = None
    _NP_IMPORT_ERROR = e
else:
    _NP_IMPORT_ERROR = None
    np_set_log_level("ERROR")

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
    output_units: Optional[str] = None        # override publish units (e.g., W in, kWh out)
    days_hist: int = 7
    export_days: Optional[int] = None

    source_is_cumulative: bool = False
    train_target: str = "interval"            # interval|level|cumulative
    aggregation: Optional[str] = None         # resample override
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
# Config loading
# --------------------------------------------------------------------------- #

def _load_role(name: str, d: Dict[str, Any]) -> RoleCfg:
    return RoleCfg(
        target_transform=d.get("target_transform", "interval"),
        aggregation=d.get("aggregation", "sum"),
        publish_state_class=d.get("publish_state_class", "measurement"),
        model_backend=d.get("model", {}).get("backend", "neuralprophet"),
        n_lags=d.get("model", {}).get("n_lags", 8),
        seasonality_reg=d.get("model", {}).get("seasonality_reg", 5.0),
        seasonality_mode=d.get("model", {}).get("seasonality_mode", "additive"),
        learning_rate=d.get("model", {}).get("learning_rate"),
    )


def _load_sensor(dflt: Dict[str, Any], d: Dict[str, Any]) -> SensorCfg:
    merged = dict(dflt)
    merged.update(d)
    rd_map = merged.get("reset_detection", {})
    rd = ResetDetectionCfg(
        enabled=rd_map.get("enabled", False),
        low=rd_map.get("low", 1.0),
        high=rd_map.get("high", 2.0),
        hard_reset_value=rd_map.get("hard_reset_value"),
    )
    return SensorCfg(
        name=merged["name"],
        role=merged.get("role", "incrementing_energy"),
        units=merged.get("units", ""),
        output_units=merged.get("output_units"),
        days_hist=merged.get("days", merged.get("days_hist", 7)),
        export_days=merged.get("export_days"),
        source_is_cumulative=merged.get("source_is_cumulative", False),
        train_target=merged.get("train_target", "interval"),
        aggregation=merged.get("aggregation"),
        log_transform=merged.get("log_transform", False),
        reset_detection=rd,
        publish_interval=merged.get("publish_interval", True),
        publish_cumulative=merged.get("publish_cumulative", True),
        publish_daily_cumulative=merged.get("publish_daily_cumulative", True),
        covariates_future=merged.get("covariates_future", []) or [],
        covariates_lagged=merged.get("covariates_lagged", []) or [],
        n_lags=merged.get("n_lags"),
        seasonality_reg=merged.get("seasonality_reg"),
        seasonality_mode=merged.get("seasonality_mode"),
        learning_rate=merged.get("learning_rate"),
        country=merged.get("country"),
        database=merged.get("database", True),
        max_age=merged.get("max_age", 365),
        plot=merged.get("plot", False),
        cascade_outputs=merged.get("cascade_outputs", {}) or {},
    )


def load_config(path: str = DEFAULT_CONFIG_PATH) -> PredAIConfig:
    if not os.path.exists(path):
        logger.error("Configuration file %s not found.", path)
        return PredAIConfig()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    dflt = raw.get("defaults", {})
    publish_prefix = dflt.get("publish_prefix", DEFAULT_PUBLISH_PREFIX)

    roles_raw = raw.get("roles", {}) or {}
    roles = {k: _load_role(k, v) for k, v in roles_raw.items()}

    sensors_raw = raw.get("sensors", []) or []
    sensors: List[SensorCfg] = []
    for s in sensors_raw:
        s_clean = {k: v for k, v in s.items() if k != "roles"}  # ignore stray nested roles
        sensors.append(_load_sensor(dflt, s_clean))

    cfg = PredAIConfig(
        update_every=raw.get("update_every", 30),
        common_interval=raw.get("common_interval", DEFAULT_INTERVAL_MIN),
        horizons=raw.get("horizons", DEFAULT_HORIZONS_MIN),
        publish_prefix=publish_prefix,
        defaults=dflt,
        roles=roles,
        sensors=sensors,
        timezone_name=raw.get("timezone", "Europe/London"),
        cov_map=raw.get("covariates", {}) or {},
    )
    return cfg


# --------------------------------------------------------------------------- #
# Home Assistant Interface
# --------------------------------------------------------------------------- #

class HAInterface:
    def __init__(self, ha_url: Optional[str], ha_key: Optional[str], session: Optional[aiohttp.ClientSession] = None):
        self.ha_url = ha_url or "http://supervisor/core"
        self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            raise RuntimeError("No Home Assistant key found.")
        self._session = session
        mask = self.ha_key[:6] + "…" if len(self.ha_key) >= 6 else "***"
        logger.info("HA Interface initialised (token %s, url %s)", mask, self.ha_url)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def api_call(self, method: str, endpoint: str, params: Optional[dict] = None, json_data: Optional[dict] = None) -> Any:
        url = self.ha_url.rstrip("/") + endpoint
        sess = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.ha_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            async with sess.request(method, url, headers=headers, params=params, json=json_data) as resp:
                if resp.status >= 400:
                    logger.warning("HA API %s %s -> %s", method, endpoint, resp.status)
                try:
                    return await resp.json()
                except aiohttp.ContentTypeError:
                    logger.error("Non‑JSON response from %s", url)
                    return None
        except aiohttp.client_exceptions.ClientError as e:
            logger.error("HA API error %s %s: %s", method, endpoint, e)
            return None

    async def get_history(self, sensor: str, start: datetime, end: datetime) -> Tuple[List[dict], Optional[datetime], Optional[datetime]]:
        params = {
            "filter_entity_id": sensor,
            "end_time": end.strftime(TIME_FORMAT_HA),
        }
        endpoint = "/api/history/period/" + start.strftime(TIME_FORMAT_HA)
        res = await self.api_call("GET", endpoint, params=params)
        if not res:
            logger.warning("No history for %s", sensor)
            return [], None, None
        arr = res[0] if isinstance(res, list) and res else []
        try:
            st = timestr_to_datetime(arr[0]["last_updated"]) if arr else None
            en = timestr_to_datetime(arr[-1]["last_updated"]) if arr else None
        except Exception:
            st = en = None
        return arr, st, en

    async def get_state(self, entity_id: str, default: Any = None, attribute: Optional[str] = None):
        item = await self.api_call("GET", f"/api/states/{entity_id}")
        if not item:
            return default
        if attribute:
            return item.get("attributes", {}).get(attribute, default)
        return item.get("state", default)

    async def set_state(self, entity_id: str, state: Any, attributes: Optional[dict] = None):
        data = {"state": state}
        if attributes:
            data["attributes"] = attributes
        await self.api_call("POST", f"/api/states/{entity_id}", json_data=data)


# --------------------------------------------------------------------------- #
# SQLite history cache
# --------------------------------------------------------------------------- #

class HistoryDB:
    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self.con = sqlite3.connect(self.path)
        self.cur = self.con.cursor()

    def safe_name(self, name: str) -> str:
        t = name.replace(".", "_")
        if not SAFE_TBL_RE.match(t):
            raise ValueError(f"Unsafe table name: {name}")
        return t

    def create_table(self, table: str):
        t = self.safe_name(table)
        self.cur.execute(f"CREATE TABLE IF NOT EXISTS {t} (timestamp TEXT PRIMARY KEY, value REAL)")
        self.con.commit()

    def cleanup_table(self, table: str, oldest_dt: datetime):
        t = self.safe_name(table)
        oldest_stamp = oldest_dt.strftime("%Y-%m-%d %H:%M:%S%z")
        self.cur.execute(f"DELETE FROM {t} WHERE timestamp < ?", (oldest_stamp,))
        self.con.commit()

    def get_history(self, table: str) -> pd.DataFrame:
        t = self.safe_name(table)
        self.create_table(t) 
        self.cur.execute(f"SELECT * FROM {t} ORDER BY timestamp")
        rows = self.cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["ds", "y"])
        df = pd.DataFrame(rows, columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"], utc=True, errors="coerce")
        return df.dropna(subset=["ds"])

    def store_history(self, table: str, history: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
        t = self.safe_name(table)
        self.create_table(t)
        prev_values = set(str(x) for x in prev["ds"].astype(str).tolist())
        added = 0
        for _, row in history.iterrows():
            timestamp = pd.to_datetime(row["ds"], utc=True, errors="coerce")
            if pd.isna(timestamp):
                continue
            timestamp_s = timestamp.isoformat()
            value = float(row["y"])
            if timestamp_s not in prev_values:
                self.cur.execute(f"INSERT OR IGNORE INTO {t} (timestamp, value) VALUES (?, ?)", (timestamp_s, value))
                prev_values.add(timestamp_s)
                prev.loc[len(prev)] = {"ds": timestamp, "y": value}
                added += 1
        self.con.commit()
        logger.info("DB: added %s rows to %s", added, t)
        return prev

    def close(self):
        self.con.close()


# --------------------------------------------------------------------------- #
# Transform utilities
# --------------------------------------------------------------------------- #

def normalise_history(raw: List[dict]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["ds", "value"])
    df = pd.DataFrame(raw)
    df["ds"] = pd.to_datetime(df["last_updated"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["state"], errors="coerce")
    df = df.dropna(subset=["ds", "value"]).sort_values("ds")
    return df[["ds", "value"]]


def resample_sensor(df: pd.DataFrame, freq: str, how: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.set_index("ds").sort_index()
    if how == "sum":
        agg = df["value"].resample(freq).sum(min_count=1)
    elif how == "last":
        agg = df["value"].resample(freq).last()
    else:  # mean default
        agg = df["value"].resample(freq).mean()
    out = agg.to_frame("value").reset_index()
    return out.dropna(subset=["value"])


def cumulative_to_interval(df: pd.DataFrame, reset_cfg: ResetDetectionCfg) -> pd.DataFrame:
    if df.empty:
        df["y"] = []
        return df
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


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = np.log1p(df["y"].clip(lower=0))
    df.attrs["log_transform_applied"] = True
    return df


def invert_log_transform(arr: np.ndarray, applied: bool) -> np.ndarray:
    if not applied:
        return arr
    return np.expm1(arr)


# --------------------------------------------------------------------------- #
# CovariateResolver
# --------------------------------------------------------------------------- #

class CovariateResolver:
    """
    Basic covariate resolution with alias & scale support.

    Example YAML:
      covariates:
        mixergy_charge:
          entity: sensor.current_charge
          scale: 0.01
        outdoor_temp_forecast: sensor.external_temperature
    """

    def __init__(self, iface: HAInterface, cov_map: Dict[str, Any]):
        self.iface = iface
        self.map = cov_map

    def _resolve(self, name: str) -> dict:
        val = self.map.get(name, name)
        if isinstance(val, str):
            return {"entity": val, "scale": 1.0}
        if isinstance(val, dict):
            return {
                "entity": val.get("entity", name),
                "scale": val.get("scale", 1.0),
                "attr": val.get("attr"),
                "forecast_attr": val.get("forecast_attr"),
            }
        return {"entity": str(val), "scale": 1.0}

    async def get_hist_series(self, cov_name: str, start: datetime, end: datetime, freq: str, how: str) -> pd.Series:
        meta = self._resolve(cov_name)
        entity_id = meta["entity"]
        raw, _, _ = await self.iface.get_history(entity_id, start, end)
        df = normalise_history(raw)
        if df.empty:
            return pd.Series([], dtype=float)
        # never sum temps/% etc.; if how=='sum' use mean
        df = resample_sensor(df, freq, "mean" if how == "sum" else how)
        df["value"] = df["value"] * meta.get("scale", 1.0)
        return df.set_index("ds")["value"]

    async def get_future_series(self, cov_name: str, future_index: pd.DatetimeIndex, default: float = 0.0) -> pd.Series:
        meta = self._resolve(cov_name)
        entity_id = meta["entity"]
        val = await self.iface.get_state(entity_id)
        try:
            v = float(val) * meta.get("scale", 1.0)
        except (TypeError, ValueError):
            v = default
        return pd.Series(v, index=future_index)


# --------------------------------------------------------------------------- #
# Model Backend (NeuralProphet)
# --------------------------------------------------------------------------- #

class NPBackend:
    def __init__(self,
                 n_lags: int,
                 n_forecasts: int,
                 seasonality_reg: float,
                 seasonality_mode: str = "additive",
                 learning_rate: Optional[float] = None,
                 country: Optional[str] = None):
        if NeuralProphet is None:
            raise RuntimeError(f"NeuralProphet import failed: {_NP_IMPORT_ERROR}")
        kw = dict(
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

    def add_future_regressor(self, name: str, mode: str = "additive"):
        self.model.add_future_regressor(name, mode=mode)

    def add_lagged_regressor(self, name: str, n_lags: Optional[int] = None):
        self.model.add_lagged_regressor(name, n_lags=n_lags)

    def fit(self, df: pd.DataFrame, freq: str):
        self.model.fit(df, freq=freq, progress=None)
        self.fitted = True

    def make_future(self, df: pd.DataFrame, periods: int) -> pd.DataFrame:
        return self.model.make_future_dataframe(df, n_historic_predictions=True, periods=periods)

    def predict(self, df_future: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(df_future)


# --------------------------------------------------------------------------- #
# Horizon helpers & publishing
# --------------------------------------------------------------------------- #

def horizon_steps(minutes_ahead: int, interval_min: int) -> int:
    return max(1, minutes_ahead // interval_min)


def horizon_agg(yhat_interval: Sequence[float], interval_min: int, minutes_ahead: int) -> float:
    steps = min(horizon_steps(minutes_ahead, interval_min), len(yhat_interval))
    return float(np.nansum(yhat_interval[:steps]))


def make_entity_name(prefix: str, base: str, suffix: Optional[str] = None) -> str:
    base = base.replace(".", "_")
    parts = [prefix + base]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def dict_from_series(index: Sequence[datetime], values: Sequence[float], tz: timezone) -> Dict[str, float]:
    return {
        ensure_utc(ts).astimezone(tz).strftime(TIME_FORMAT_HA): round(float(v), 3)
        for ts, v in zip(index, values)
    }


def daily_cumulative_series(index: Sequence[datetime], values: Sequence[float], tz: timezone) -> Dict[str, float]:
    cum = 0.0
    out: Dict[str, float] = {}
    current_day = None
    for ts, v in zip(index, values):
        lts = ensure_utc(ts).astimezone(tz)
        if current_day != lts.date():
            cum = 0.0
            current_day = lts.date()
        cum += max(float(v), 0.0)
        out[lts.strftime(TIME_FORMAT_HA)] = round(cum, 3)
    return out


async def publish_forecasts(sensor: SensorCfg,
                            role_cfg: RoleCfg,
                            iface: HAInterface,
                            cfg: PredAIConfig,
                            ds_future: Sequence[datetime],
                            yhat_interval: Sequence[float],
                            yhat_level: Optional[Sequence[float]] = None,
                            metrics: Optional[dict] = None):
    tz = cfg.tz
    prefix = cfg.publish_prefix

    yhat_interval = np.array(yhat_interval, dtype=float)
    yhat_interval = np.clip(yhat_interval, 0, None)  # no negatives

    cum_from_now = np.cumsum(yhat_interval)
    daily_cum = daily_cumulative_series(ds_future, yhat_interval, tz)

    ser_interval = dict_from_series(ds_future, yhat_interval, tz)
    ser_cum = dict_from_series(ds_future, cum_from_now, tz)

    model_ts_iso = datetime.now(timezone.utc).astimezone(tz).isoformat()
    meta = {
        "model_ts": model_ts_iso,
        "model_backend": role_cfg.model_backend,
        "training_rows": metrics.get("training_rows") if metrics else None,
        "mae_recent": metrics.get("mae_recent") if metrics else None,
    }

    publish_units = sensor.output_units or sensor.units

    if sensor.publish_interval:
        ent_interval = make_entity_name(prefix, sensor.name, "interval")
        await iface.set_state(
            ent_interval,
            state=round(float(yhat_interval[0]) if len(yhat_interval) else 0.0, 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": "measurement",
                "forecast_series": ser_interval,
                **meta,
            },
        )

    if sensor.publish_cumulative:
        ent_cum = make_entity_name(prefix, sensor.name, "cum")
        await iface.set_state(
            ent_cum,
            state=round(float(cum_from_now[-1]) if len(cum_from_now) else 0.0, 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": "measurement",
                "forecast_series": ser_cum,
                **meta,
            },
        )

    if sensor.publish_daily_cumulative:
        ent_daily = make_entity_name(prefix, sensor.name, "daily_cum")
        today_str = datetime.now(tz).strftime("%Y-%m-%d")
        todays = {k: v for k, v in daily_cum.items() if k.startswith(today_str)}
        state_val = list(todays.values())[-1] if todays else list(daily_cum.values())[-1]
        await iface.set_state(
            ent_daily,
            state=round(float(state_val), 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": "measurement",
                "forecast_series": daily_cum,
                **meta,
            },
        )

    # Horizon scalars
    for m in cfg.horizons:
        suffix = f"pred_{m//60}h"
        ent_h = make_entity_name(prefix, sensor.name, suffix)
        if sensor.train_target == "level":  # e.g., temperature
            arr = np.array(yhat_level if yhat_level is not None else yhat_interval)
            steps = min(horizon_steps(m, cfg.common_interval), len(arr))
            val = arr[steps - 1]
        else:
            val = horizon_agg(yhat_interval, cfg.common_interval, m)
        await iface.set_state(
            ent_h,
            state=round(float(val), 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": "measurement",
                "generated_from": make_entity_name(prefix, sensor.name, "interval"),
                **meta,
            },
        )


# --------------------------------------------------------------------------- #
# Sensor job execution
# --------------------------------------------------------------------------- #

async def run_sensor_job(sensor: SensorCfg,
                         role_cfg: RoleCfg,
                         cfg: PredAIConfig,
                         iface: HAInterface,
                         cov_res: CovariateResolver,
                         db: Optional[HistoryDB]) -> None:
    interval_min = cfg.common_interval
    freq = f"{interval_min}min"
    tz = cfg.tz

    # Round now to interval boundary
    now = datetime.now(timezone.utc).astimezone(tz).replace(second=0, microsecond=0)
    minute_floor = (now.minute // interval_min) * interval_min
    now = now.replace(minute=minute_floor)

    start_hist = now - timedelta(days=sensor.days_hist)
    end_hist = now

    logger.info("Sensor %s: fetching history %s → %s", sensor.name, start_hist, end_hist)
    raw_hist, st, en = await iface.get_history(sensor.name, start_hist, end_hist)
    df = normalise_history(raw_hist)

    # DB merge
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

    if df.empty:
        logger.warning("Sensor %s: no data; skipping.", sensor.name)
        return

    # Resample
    agg = sensor.effective_aggregation(role_cfg)
    df = resample_sensor(df, freq, agg)

    # Power->energy (heuristic)
    if (not sensor.source_is_cumulative) and sensor.train_target == "interval":
        units_lower = (sensor.units or "").lower()
        if "w" in units_lower:  # W or kW
            if df["value"].max() > 50:  # assume W
                df["value"] = df["value"] / 1000.0
            df["value"] = df["value"] * (interval_min / 60.0)  # kWh per bucket
            if not sensor.output_units:
                sensor.output_units = "kWh"

    # Transform to modelling target
    if sensor.source_is_cumulative:
        df = cumulative_to_interval(df, sensor.reset_detection)
    else:
        df = df.rename(columns={"value": "y"})

    # Clean
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["y"])

    log_applied = False
    if sensor.log_transform:
        df = apply_log_transform(df)
        log_applied = True

    if len(df) < role_cfg.n_lags + 5:
        logger.warning("Sensor %s: insufficient history (%s rows); skipping model.", sensor.name, len(df))
        return

    # Training frame
    train_df = df[["ds", "y"]].copy()

    if role_cfg.model_backend == "neuralprophet":
        steps = max(horizon_steps(m, interval_min) for m in cfg.horizons)
        backend = NPBackend(
            n_lags=sensor.effective_n_lags(role_cfg),
            n_forecasts=steps,
            seasonality_reg=sensor.effective_seasonality_reg(role_cfg),
            seasonality_mode=sensor.effective_seasonality_mode(role_cfg),
            learning_rate=sensor.effective_learning_rate(role_cfg),
            country=sensor.country,
        )

        # lagged covariates
        for cov in sensor.covariates_lagged:
            s = await cov_res.get_hist_series(cov, start_hist, end_hist, freq, agg)
            if not s.empty:
                train_df = train_df.merge(s.rename(cov), left_on="ds", right_index=True, how="left")
                backend.add_lagged_regressor(cov, n_lags=sensor.effective_n_lags(role_cfg))
            else:
                logger.debug("Covariate %s lagged: no history.", cov)

        # future covariates
        for cov in sensor.covariates_future:
            s = await cov_res.get_hist_series(cov, start_hist, end_hist, freq, agg)
            if not s.empty:
                train_df = train_df.merge(s.rename(cov), left_on="ds", right_index=True, how="left")
            else:
                train_df[cov] = np.nan
            backend.add_future_regressor(cov, mode="additive")

        train_df = train_df.ffill().bfill()   # ADD – fill internal NaNs
        train_df = train_df.dropna()          # ADD – drop any rows still invalid
        
        # Fit
        backend.fit(train_df, freq=freq)

        # Future frame
        df_future = backend.make_future(train_df, periods=steps)
        fut_mask = df_future["ds"] > train_df["ds"].max()
        if fut_mask.any():
            fut_idx = pd.to_datetime(df_future.loc[fut_mask, "ds"], utc=True)
            for cov in sensor.covariates_future:
                fut_s = await cov_res.get_future_series(cov, fut_idx, default=0.0)
                df_future.loc[fut_mask, cov] = fut_s.to_numpy()

        # Predict
        fcst = backend.predict(df_future)

        # Extract forecast row corresponding to last training timestamp
        base = train_df["ds"].max()
        row_mask = fcst["ds"] == base
        if not row_mask.any():  # fallback: last row
            row_mask = fcst.index == (len(fcst) - 1)
        yhat_cols = [f"yhat{i}" for i in range(1, steps + 1)]
        yhat_int = fcst.loc[row_mask, yhat_cols].values.flatten()

        if log_applied:
            yhat_int = invert_log_transform(yhat_int, True)

        ds_future = [pd.to_datetime(base) + timedelta(minutes=interval_min * i) for i in range(1, steps + 1)]

        metrics = {"training_rows": int(len(train_df)), "mae_recent": None}
        await publish_forecasts(sensor, role_cfg, iface, cfg, ds_future, yhat_int, metrics=metrics)

    else:
        logger.error("Unsupported backend %s for sensor %s", role_cfg.model_backend, sensor.name)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

async def predai_main():
    cfg = load_config(DEFAULT_CONFIG_PATH)

    # read raw for HA creds & initial cov_map
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
            # hot-reload config
            cfg = load_config(DEFAULT_CONFIG_PATH)
            # refresh covariate map
            cov_res.map = cfg.cov_map

            for s in cfg.sensors:
                role_cfg = cfg.roles.get(s.role, RoleCfg())
                try:
                    await run_sensor_job(s, role_cfg, cfg, iface, cov_res, db)
                except Exception as e:
                    logger.exception("Sensor job failed for %s: %s", s.name, e)

            now_str = datetime.now(timezone.utc).isoformat()
            await iface.set_state(
                "sensor.predai_last_run",
                state=now_str,
                attributes={"unit_of_measurement": "time"},
            )

            logger.info("PredAI sleeping %s minutes.", cfg.update_every)
            # Sleep minute chunks; break early if heartbeat lost
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
