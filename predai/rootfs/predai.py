#!/usr/bin/env python3
"""
predai.py – Hardened PredAI with covariate support & NaN‑robust training
========================================================================
Key additions:
* Covariates (future & lagged) – optional
* Clean fallback to target‑only mode (mirrors original behaviour)
* Explicit dataset cleaning (drops NaN/Inf targets, de‑dupes timestamps)
* Naïve forecast fallback for constant / tiny series
* Auto n_lags relaxation
* Safe JSON output (never sends NaN / Inf to HA)
* Deprecation‑proof forward fill (.ffill())
* Detailed one‑off NaN diagnostics (disable after stabilisation)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os
import math
import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import sqlite3
import requests
from neuralprophet import NeuralProphet, set_log_level
import yaml

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def timestr_to_datetime(ts: str | None) -> datetime | None:
    """Convert HA timestamp strings to minute‑aligned tz‑aware datetime."""
    if ts is None:
        return None
    for fmt in (TIME_FORMAT_HA, TIME_FORMAT_HA_DOT):
        try:
            d = datetime.strptime(ts, fmt)
            return d.replace(second=0, microsecond=0)
        except ValueError:
            continue
    return None

def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)

_nan_report_issued = False
def report_nans(label: str, df: pd.DataFrame):
    """Log first appearance of NaN / Inf in a dataframe (one time per cycle)."""
    global _nan_report_issued
    if _nan_report_issued or df.empty:
        return
    chk = df.replace([float("inf"), -float("inf")], math.nan).isna().any()
    if chk.any():
        cols = chk[chk].index.tolist()
        counts = df[cols].isna().sum().to_dict()
        print(f"[NaN‑debug] {label}: non‑finite values in {counts}")
        _nan_report_issued = True

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned copy: replace Inf→NaN, drop NaN targets, forward fill regressors, de‑dup."""
    if df.empty:
        return df
    c = df.copy()
    c.replace([float("inf"), -float("inf")], math.nan, inplace=True)
    if "y" in c.columns:
        c.dropna(subset=["y"], inplace=True)
    # Forward fill regressors (everything except ds,y)
    reg_cols = [col for col in c.columns if col not in ("ds", "y")]
    if reg_cols:
        c[reg_cols] = c[reg_cols].ffill()
    c.drop_duplicates(subset=["ds"], keep="last", inplace=True)
    c.sort_values("ds", inplace=True)
    c.reset_index(drop=True, inplace=True)
    return c

# ---------------------------------------------------------------------
# Home Assistant interface
# ---------------------------------------------------------------------
class HAInterface:
    def __init__(self, url: str | None, token: str | None):
        self.ha_url = url or "http://supervisor/core"
        self.ha_key = token or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            raise SystemExit("Missing Home Assistant token")
        print(f"HA interface → {self.ha_url}")

    async def api_call(self, endpoint: str, *, post=False, json_body=None, params=None):
        url = self.ha_url + endpoint
        headers = {
            "Authorization": f"Bearer {self.ha_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            if post:
                resp = await asyncio.to_thread(
                    requests.post, url, headers=headers, json=json_body, timeout=TIMEOUT
                )
            else:
                resp = await asyncio.to_thread(
                    requests.get, url, headers=headers, params=params, timeout=TIMEOUT
                )
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            print(f"HA api_call error {url}: {exc}")
            return None

    async def get_history(self, entity: str, now: datetime, *, days: int) -> Tuple[list, datetime | None, datetime | None]:
        start = now - timedelta(days=days)
        data = await self.api_call(
            f"/api/history/period/{start.strftime(TIME_FORMAT_HA)}",
            params={"filter_entity_id": entity, "end_time": now.strftime(TIME_FORMAT_HA)},
        )
        if not data:
            return [], None, None
        arr = data[0]
        return arr, timestr_to_datetime(arr[0]["last_updated"]), timestr_to_datetime(arr[-1]["last_updated"])

    async def get_state(self, entity: str, *, attr: str | None = None, default: Any = None):
        s = await self.api_call(f"/api/states/{entity}")
        if not s:
            return default
        return s["attributes"].get(attr, default) if attr else s.get("state", default)

    async def set_state(self, entity: str, state: Any, *, attributes: dict | None = None):
        payload = {"state": state}
        if attributes:
            payload["attributes"] = attributes
        # Drop any non‑serialisable floats before send
        if attributes:
            for d in (attributes.get("results"), attributes.get("source")):
                if isinstance(d, dict):
                    for k, v in list(d.items()):
                        if not _finite(v):
                            d.pop(k, None)
        await self.api_call(f"/api/states/{entity}", post=True, json_body=payload)

# ---------------------------------------------------------------------
# Prophet wrapper
# ---------------------------------------------------------------------
class Prophet:
    def __init__(self, period: int = 30):
        set_log_level("ERROR")
        self.period = period
        self.model: NeuralProphet | None = None
        self.forecast: pd.DataFrame | None = None

    async def _dataset_from_ha(
        self,
        raw: list[dict[str, Any]],
        start: datetime,
        end: datetime,
        *,
        incrementing: bool,
        max_increment: float,
        reset_low: float,
        reset_high: float,
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=["ds", "y"])
        cur = start.replace(second=0, microsecond=0, minute=0)
        idx = 0
        last_val = None
        total = 0.0
        while cur <= end and idx < len(raw):
            try:
                val = float(raw[idx]["state"])
            except (TypeError, ValueError):
                idx += 1
                continue
            if last_val is None:
                last_val = val
            if incrementing:
                if val < last_val and val < reset_low and last_val > reset_high:
                    total += val  # counter reset
                else:
                    if max_increment and abs(val - last_val) > max_increment:
                        val = last_val
                    total = max(total + val - last_val, 0)
            last_val = val
            pt = timestr_to_datetime(raw[idx]["last_updated"])
            if not pt or pt < cur:
                idx += 1
                continue
            df.loc[len(df)] = {"ds": cur, "y": total if incrementing else val}
            cur += timedelta(minutes=self.period)
        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        return df

    async def historical_dataframe(
        self,
        iface: HAInterface,
        sensor: str,
        now: datetime,
        *,
        days: int,
        incrementing: bool,
        max_increment: float,
        reset_low: float,
        reset_high: float,
    ) -> pd.DataFrame:
        raw, start, end = await iface.get_history(sensor, now, days=days)
        if not raw or start is None or end is None:
            return pd.DataFrame(columns=["ds", "y"])
        return await self._dataset_from_ha(
            raw,
            start,
            end,
            incrementing=incrementing,
            max_increment=max_increment,
            reset_low=reset_low,
            reset_high=reset_high,
        )

    async def train(
        self,
        dataset: pd.DataFrame,
        future_periods: int,
        *,
        regressor_meta: List[dict[str, Any]],
        future_frames: Dict[str, pd.DataFrame],
        target_n_lags: int,
        country: str | None,
        disable_yearly: bool,
        learning_rate: float | None,
    ):
        # Clean dataset
        dataset = clean_dataset(dataset)
        report_nans("dataset-cleaned", dataset)

        if dataset.empty:
            print("WARN: dataset empty after cleaning – skipping training.")
            self.forecast = None
            return

        # Constant series fallback
        if dataset["y"].nunique() < 2:
            print("INFO: constant / flat target – using naïve last-value forecast.")
            last_val = float(dataset.iloc[-1]["y"])
            last_ds = dataset.iloc[-1]["ds"]
            future_idx = pd.date_range(
                start=last_ds + timedelta(minutes=self.period),
                periods=future_periods,
                freq=f"{self.period}min",
            )
            full_ds = list(dataset["ds"]) + list(future_idx)
            self.forecast = pd.DataFrame(
                {
                    "ds": full_ds,
                    "y": list(dataset["y"]) + [math.nan] * future_periods,
                    "yhat1": [last_val] * (len(full_ds)),
                }
            )
            report_nans("forecast", self.forecast)
            return

        # Auto relax n_lags if dataset too short
        if target_n_lags and len(dataset) <= target_n_lags + 10:
            print(f"Auto-reducing n_lags from {target_n_lags} to 0 (rows={len(dataset)}).")
            target_n_lags = 0

        self.model = NeuralProphet(
            n_lags=target_n_lags,
            yearly_seasonality=not disable_yearly,
            weekly_seasonality=True,
            daily_seasonality=True,
            learning_rate=learning_rate,  # if None, NP will search/choose
        )
        if country:
            self.model.add_country_holidays(country)

        for meta in regressor_meta:
            if meta["type"] == "future":
                self.model.add_future_regressor(meta["name"])
            else:
                self.model.add_lagged_regressor(meta["name"], n_lags=meta.get("n_lags", 24))

        self.model.fit(dataset, freq=f"{self.period}min", progress=None)

        df_future = self.model.make_future_dataframe(
            dataset, n_historic_predictions=True, periods=future_periods
        )

        # Inject known future regressors
        for name, fr in future_frames.items():
            if fr.empty:
                continue
            fr = fr.copy()
            fr["ds"] = pd.to_datetime(fr["ds"], utc=True)
            df_future = df_future.merge(fr, on="ds", how="left", suffixes=("", "_dup"))
            dup = f"{name}_dup"
            if dup in df_future.columns:
                df_future.drop(columns=[dup], inplace=True)

        df_future.ffill(inplace=True)
        self.forecast = self.model.predict(df_future)
        report_nans("forecast", self.forecast)

    async def save_prediction(
        self,
        entity: str,
        now: datetime,
        iface: HAInterface,
        *,
        incrementing: bool,
        reset_daily: bool,
        units: str,
        history_days: int,
    ):
        if self.forecast is None or self.forecast.empty:
            return

        total = total_org = 0.0
        res: Dict[str, float] = {}
        src: Dict[str, float] = {}
        last_finite_pred = None

        for _, r in self.forecast.iterrows():
            ds = r["ds"]
            if ds.tzinfo is None:
                ds = ds.tz_localize(timezone.utc)
            ts = ds
            if (now - ts).days > history_days:
                continue

            v_pred = r.get("yhat1")
            v_org = r.get("y")

            if not _finite(v_pred):
                continue  # skip non‑finite prediction
            last_finite_pred = v_pred

            if incrementing:
                if ts <= now and reset_daily and ts.hour == 0 and ts.minute == 0:
                    total = total_org = 0.0
                total += v_pred
                if _finite(v_org):
                    total_org += v_org
                if _finite(total):
                    res[ts.strftime(TIME_FORMAT_HA)] = round(total, 2)
                if _finite(total_org):
                    src[ts.strftime(TIME_FORMAT_HA)] = round(total_org, 2)
            else:
                res[ts.strftime(TIME_FORMAT_HA)] = round(v_pred, 2)
                if _finite(v_org):
                    src[ts.strftime(TIME_FORMAT_HA)] = round(v_org, 2)

        if incrementing and _finite(total):
            final_state = round(total, 2)
        elif _finite(last_finite_pred):
            final_state = round(last_finite_pred, 2)
        else:
            final_state = 0

        await iface.set_state(
            entity,
            final_state,
            attributes={
                "last_updated": str(now),
                "unit_of_measurement": units,
                "state_class": "measurement",
                "results": res,
                "source": src,
            },
        )

# ---------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------
class Database:
    def __init__(self):
        self.con = sqlite3.connect("/config/predai.db")
        self.cur = self.con.cursor()

    async def create_table(self, name: str):
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS {name} (timestamp TEXT PRIMARY KEY, value REAL)"
        )
        self.con.commit()

    async def get_history(self, table: str) -> pd.DataFrame:
        self.cur.execute(f"SELECT * FROM {table} ORDER BY timestamp")
        rows = self.cur.fetchall()
        df = (
            pd.DataFrame(rows, columns=["ds", "y"])
            if rows
            else pd.DataFrame(columns=["ds", "y"])
        )
        if not df.empty:
            df["ds"] = pd.to_datetime(
                df["ds"], format="ISO8601", utc=True, errors="coerce"
            )
            df.dropna(subset=["ds"], inplace=True)
        return df

    async def store_history(
        self, table: str, history: pd.DataFrame, prev: pd.DataFrame
    ) -> pd.DataFrame:
        prev_set = set(prev["ds"].astype(str).tolist())
        added = 0
        for _, row in history.iterrows():
            ts_str = str(row["ds"])
            val = row["y"]
            if ts_str not in prev_set:
                prev.loc[len(prev)] = {"ds": row["ds"], "y": val}
                self.cur.execute(
                    "INSERT INTO {} (timestamp, value) VALUES (?, ?)".format(table),
                    (ts_str, float(val)),
                )
                added += 1
        self.con.commit()
        if added:
            print(f"SQLite: added {added} rows to {table}")
        return prev

    async def cleanup_table(self, table: str, max_age: int):
        oldest = (
            datetime.now(timezone.utc) - timedelta(days=max_age)
        ).strftime("%Y-%m-%d %H:%M:%S%z")
        self.cur.execute(f"DELETE FROM {table} WHERE timestamp < '{oldest}'")
        self.con.commit()

# ---------------------------------------------------------------------
# Covariate builder
# ---------------------------------------------------------------------
async def build_covariate_frames(
    cov_configs: List[dict[str, Any]],
    iface: HAInterface,
    prophet: Prophet,
    now: datetime,
    *,
    period: int,
    days: int,
) -> Tuple[List[dict[str, Any]], pd.DataFrame, Dict[str, pd.DataFrame]]:
    meta: List[dict[str, Any]] = []
    merged: pd.DataFrame | None = None
    future_frames: Dict[str, pd.DataFrame] = {}

    for cov in cov_configs:
        ent = cov.get("entity_id")
        if not ent:
            continue
        col = ent.replace(".", "_")
        future_known = bool(cov.get("future", False))
        n_lags = int(cov.get("n_lags", 24))
        incrementing = bool(cov.get("incrementing", False))

        hist = await prophet.historical_dataframe(
            iface,
            ent,
            now,
            days=days,
            incrementing=incrementing,
            max_increment=0,
            reset_low=0.0,
            reset_high=0.0,
        )
        hist.rename(columns={"y": col}, inplace=True)

        merged = hist if merged is None else merged.merge(hist, on="ds", how="outer")

        meta.append(
            {
                "name": col,
                "type": "future" if future_known else "lagged",
                "n_lags": n_lags,
            }
        )

        if future_known:
            attr = await iface.get_state(ent, attr="results")
            fut: Dict[datetime, float] = {}
            if isinstance(attr, dict):
                for ts_str, val in attr.items():
                    dt = timestr_to_datetime(ts_str)
                    if dt and dt >= now:
                        try:
                            fut[dt] = float(val)
                        except (TypeError, ValueError):
                            continue
            if fut:
                future_frames[col] = pd.DataFrame(
                    {"ds": list(fut.keys()), col: list(fut.values())}
                )

    if merged is None or merged.empty:
        return meta, pd.DataFrame(), future_frames

    merged.sort_values("ds", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged.ffill(inplace=True)
    merged["ds"] = pd.to_datetime(merged["ds"], utc=True, errors="coerce")
    merged.dropna(subset=["ds"], inplace=True)
    return meta, merged, future_frames

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
async def main() -> None:
    config_path = "/config/predai.yaml"
    cfg = yaml.safe_load(open(config_path))
    iface = HAInterface(cfg.get("ha_url"), cfg.get("ha_key"))

    while True:
        cfg = yaml.safe_load(open(config_path)) or {}
        interval_run = cfg.get("update_every", 30)
        sensors_cfg = cfg.get("sensors", [])
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0, minute=0)

        for s_cfg in sensors_cfg:
            target = s_cfg.get("name")
            if not target:
                continue
            print(f"\n=== {target} @ {now.isoformat()} ===")

            # Parameters
            days = s_cfg.get("days", 7)
            export_days = s_cfg.get("export_days", days)
            incrementing = s_cfg.get("incrementing", False)
            reset_daily = s_cfg.get("reset_daily", False)
            period = s_cfg.get("interval", 30)
            units = s_cfg.get("units", "")
            future_periods = s_cfg.get("future_periods", 96)
            use_db = s_cfg.get("database", True)
            reset_low = s_cfg.get("reset_low", 1.0)
            reset_high = s_cfg.get("reset_high", 2.0)
            max_increment = s_cfg.get("max_increment", 0.0)
            n_lags_target = s_cfg.get("n_lags", 0)
            country = s_cfg.get("country")
            max_age = s_cfg.get("max_age", 365)
            learning_rate = s_cfg.get("learning_rate")  # optional override
            disable_yearly_cfg = bool(s_cfg.get("disable_yearly", False))

            prophet = Prophet(period)

            main_df = await prophet.historical_dataframe(
                iface,
                target,
                now,
                days=days,
                incrementing=incrementing,
                max_increment=max_increment,
                reset_low=reset_low,
                reset_high=reset_high,
            )
            main_df["ds"] = pd.to_datetime(main_df["ds"], utc=True, errors="coerce")
            main_df.dropna(subset=["ds"], inplace=True)
            report_nans("main_df", main_df)

            # Subtract sensors (optional)
            subtract_names = s_cfg.get("subtract")
            if subtract_names:
                if isinstance(subtract_names, str):
                    subtract_names = [subtract_names]
                for sub in subtract_names:
                    sub_df = await prophet.historical_dataframe(
                        iface,
                        sub,
                        now,
                        days=days,
                        incrementing=incrementing,
                        max_increment=max_increment,
                        reset_low=reset_low,
                        reset_high=reset_high,
                    )
                    main_df = main_df.merge(
                        sub_df, on="ds", how="left", suffixes=("", "_sub")
                    )
                    main_df["y"] = (
                        main_df["y"].ffill() - main_df["y_sub"].fillna(0)
                    )
                    main_df.drop(columns=["y_sub"], inplace=True)

            # Database cache
            if use_db:
                tbl = target.replace(".", "_")
                db = Database()
                await db.create_table(tbl)
                prev_hist = await db.get_history(tbl)
                main_df = await db.store_history(tbl, main_df, prev_hist)
                await db.cleanup_table(tbl, max_age)

            # Covariates
            cov_meta, cov_hist, future_frames = await build_covariate_frames(
                s_cfg.get("covariates", []),
                iface,
                prophet,
                now,
                period=period,
                days=days,
            )

            if cov_hist.empty:
                dataset = main_df.copy()
            else:
                main_df["ds"] = pd.to_datetime(main_df["ds"], utc=True, errors="coerce")
                cov_hist["ds"] = pd.to_datetime(cov_hist["ds"], utc=True, errors="coerce")
                dataset = main_df.merge(cov_hist, on="ds", how="left")
                dataset.ffill(inplace=True)
            report_nans("merged-dataset", dataset)

            # Auto-disable yearly if short history (< 30 days) unless explicitly forced on/off
            history_minutes = len(dataset) * period
            auto_disable_yearly = history_minutes < 30 * 24 * 60
            disable_yearly = disable_yearly_cfg or auto_disable_yearly

            await prophet.train(
                dataset,
                future_periods,
                regressor_meta=cov_meta,
                future_frames=future_frames,
                target_n_lags=n_lags_target,
                country=country,
                disable_yearly=disable_yearly,
                learning_rate=learning_rate,
            )
            await prophet.save_prediction(
                f"{target}_prediction",
                now,
                iface,
                incrementing=incrementing,
                reset_daily=reset_daily,
                units=units,
                history_days=export_days,
            )

        # Heartbeat
        await iface.set_state(
            "sensor.predai_last_run",
            str(datetime.now(timezone.utc)),
            attributes={"unit_of_measurement": "time"},
        )
        print(f"Sleeping {interval_run} minutes…")
        for _ in range(interval_run):
            last_run = await iface.get_state("sensor.predai_last_run")
            if last_run is None:
                print("sensor.predai_last_run removed – restarting loop early")
                break
            await asyncio.sleep(60)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
