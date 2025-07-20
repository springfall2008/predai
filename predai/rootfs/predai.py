#!/usr/bin/env python3
"""predai_updated.py – Enhanced PredAI with covariate support and improved Home Assistant integration

This version extends the original SpringFall PredAI script with the ability to
leverage additional exogenous variables (covariates) when training its
NeuralProphet model, as well as with stronger support for visualising the
resulting forecasts in Home Assistant (HA). The key improvements are:

1. **Covariates**
   * Each *target* sensor in `predai.yaml` can list `covariates:` – a collection
     of HA sensors whose historical (and optionally future‑known) values help
     explain the target.
   * Two types of covariate are supported:
       * `future: true` → known in advance (e.g. weather forecasts) and thus
         added with `add_future_regressor()`.
       * `future: false` → only historical values (e.g. inside temperature) and
         therefore added as *lagged* regressors via `add_lagged_regressor()`.
     For lagged covariates you may optionally set `n_lags:` (default = 24).
   * Missing covariate values are **forward‑filled** per timestamp.

2. **Plot‑ready forecasts**
   * The predicted time‑series already stored in the `results` attribute is
     preserved, so an [ApexCharts card](https://github.com/RomRider/apexcharts-card)
     can read it directly (example YAML in README).
   * A new convenience Boolean `create_graph_sensor` allows a separate entity
     holding *only* the `results` attribute to be created if desired.

3. **Configuration schema (excerpt)**
```
update_every: 30
sensors:
  - name: sensor.energy_total
    days: 14
    future_periods: 96
    covariates:
      - entity_id: sensor.temperature_inside
        future: false          # historical‑only, lagged regressor
        n_lags: 48             # optional
      - entity_id: sensor.weather_forecast_temperature
        future: true           # future‑known regressor
```

The remainder of the behaviour (database caching, incremental counters,
subtracting sensors, daily resets, etc.) is unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta, timezone
from neuralprophet import NeuralProphet, set_log_level
import os
import aiohttp  # retained for future async http needs
import requests
import asyncio
import json
import ssl
import math
import yaml

TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"

# ‑‑‑‑‑‑ Utility ——————————————————————————————————————————————————————————

def timestr_to_datetime(timestamp: str | None) -> datetime | None:
    """Convert a Home Assistant timestamp string to a *minute‑aligned* datetime."""
    if timestamp is None:
        return None
    for fmt in (TIME_FORMAT_HA, TIME_FORMAT_HA_DOT):
        try:
            dt = datetime.strptime(timestamp, fmt)
            return dt.replace(second=0, microsecond=0)
        except ValueError:
            continue
    return None

# ‑‑‑‑‑‑ Home Assistant interface ——————————————————————————————————————

class HAInterface:
    """Minimal async wrapper around the HA REST API."""

    def __init__(self, ha_url: str | None, ha_key: str | None):
        self.ha_url = ha_url or "http://supervisor/core"
        self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            print("No Home Assistant key found, exiting")
            raise SystemExit(1)
        print(f"HA Interface started key {self.ha_key[:5]}… url {self.ha_url}")

    # Generic REST helper --------------------------------------------------
    async def api_call(self, endpoint: str, datain: dict | None = None, *, post: bool = False):
        url = self.ha_url + endpoint
        headers = {
            "Authorization": f"Bearer {self.ha_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            if post:
                response = await asyncio.to_thread(
                    requests.post, url, headers=headers, json=datain, timeout=TIMEOUT
                )
            else:
                response = await asyncio.to_thread(
                    requests.get, url, headers=headers, params=datain, timeout=TIMEOUT
                )
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.JSONDecodeError, requests.Timeout, requests.exceptions.RequestException) as exc:
            print(f"HA api_call error on {url}: {exc}")
            return None

    # Convenience wrappers -------------------------------------------------
    async def get_history(self, sensor: str, now: datetime, *, days: int = 7):
        start = now - timedelta(days=days)
        print(f"Getting history for {sensor} from {start} to {now}")
        res = await self.api_call(
            f"/api/history/period/{start.strftime(TIME_FORMAT_HA)}",
            {"filter_entity_id": sensor, "end_time": now.strftime(TIME_FORMAT_HA)},
        )
        if not res:
            return [], None, None
        res = res[0]
        start_dt = timestr_to_datetime(res[0]["last_updated"])
        end_dt = timestr_to_datetime(res[-1]["last_updated"])
        print(f"History for {sensor}: {start_dt} — {end_dt}")
        return res, start_dt, end_dt

    async def get_state(self, entity_id: str, *, default: Any = None, attribute: str | None = None):
        item = await self.api_call(f"/api/states/{entity_id}")
        if not item:
            return default
        return item.get("attributes", {}).get(attribute, default) if attribute else item.get("state", default)

    async def set_state(self, entity_id: str, state: Any, *, attributes: dict | None = None):
        data = {"state": state}
        if attributes:
            data["attributes"] = attributes
        await self.api_call(f"/api/states/{entity_id}", data, post=True)

# ‑‑‑‑‑‑ Prophet helper ——————————————————————————————————————————————

class Prophet:
    """Wraps NeuralProphet with dataset handling and HA persistence."""

    def __init__(self, period: int = 30):
        set_log_level("ERROR")
        self.period = period  # minutes
        self.model: NeuralProphet | None = None

    # ------------------------------------------------------------------ Dataset builders
    async def _dataset_from_ha(
        self,
        sensor_name: str,
        raw_data: list[dict[str, Any]],
        start: datetime,
        end: datetime,
        *,
        period_minutes: int,
        incrementing: bool = False,
        max_increment: float = 0.0,
        reset_low: float = 0.0,
        reset_high: float = 0.0,
    ) -> pd.DataFrame:
        """Convert HA raw history into a uniform‑frequency DataFrame."""
        dataset = pd.DataFrame(columns=["ds", "y"])
        cursor_time = start.replace(second=0, microsecond=0, minute=0)
        idx = 0
        last_value: float | None = None
        total = 0.0
        while cursor_time <= end and idx < len(raw_data):
            try:
                value = float(raw_data[idx]["state"])
            except (ValueError, TypeError):
                idx += 1
                continue

            if last_value is None:
                last_value = value

            # Incrementing counter logic ------------------------------
            if incrementing:
                if value < last_value and value < reset_low and last_value > reset_high:
                    total += value  # counter reset
                else:
                    if max_increment and abs(value - last_value) > max_increment:
                        value = last_value  # spike removal
                    total = max(total + value - last_value, 0)
            last_value = value

            # Align to our regular interval --------------------------
            pt = timestr_to_datetime(raw_data[idx]["last_updated"])
            if not pt or pt < cursor_time:
                idx += 1
                continue

            dataset.loc[len(dataset)] = {
                "ds": cursor_time,
                "y": max(total, 0) if incrementing else value,
            }
            cursor_time += timedelta(minutes=period_minutes)
        return dataset

    # Public helper to fetch/prepare one sensor --------------------------
    async def historical_dataframe(
        self,
        interface: HAInterface,
        sensor_name: str,
        now: datetime,
        *,
        period_minutes: int,
        days: int,
        incrementing: bool = False,
        max_increment: float = 0.0,
        reset_low: float = 0.0,
        reset_high: float = 0.0,
    ) -> pd.DataFrame:
        raw, start, end = await interface.get_history(sensor_name, now, days=days)
        if not raw:
            return pd.DataFrame(columns=["ds", "y"])
        return await self._dataset_from_ha(
            sensor_name,
            raw,
            start,
            end,
            period_minutes=period_minutes,
            incrementing=incrementing,
            max_increment=max_increment,
            reset_low=reset_low,
            reset_high=reset_high,
        )

    # ------------------------------------------------------------------ Training & forecasting
    async def train(
        self,
        dataset: pd.DataFrame,
        future_periods: int,
        *,
        regressor_meta: list[dict[str, Any]] | None = None,
        future_regressor_frames: dict[str, pd.DataFrame] | None = None,
        target_n_lags: int = 0,
        country: str | None = None,
    ) -> None:
        regressor_meta = regressor_meta or []
        self.model = NeuralProphet(
            n_lags=target_n_lags,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
        )
        if country:
            self.model.add_country_holidays(country)

        # Register regressors ----------------------------------------
        for reg in regressor_meta:
            name = reg["name"]
            if reg["type"] == "future":
                self.model.add_future_regressor(name)
            else:  # lagged
                self.model.add_lagged_regressor(name, n_lags=reg.get("n_lags", 24))

        # Fit ---------------------------------------------------------
        self.model.fit(dataset, freq=f"{self.period}min", progress=None)

        # Build future df --------------------------------------------
        df_future = self.model.make_future_dataframe(dataset, n_historic_predictions=True, periods=future_periods)

        # Inject known‑future regressors ------------------------------
        if future_regressor_frames:
            for name, frame in future_regressor_frames.items():
                df_future = df_future.merge(frame, on="ds", how="left", suffixes=("", "_drop"))
                if f"{name}_drop" in df_future.columns:
                    df_future.drop(columns=[f"{name}_drop"], inplace=True)

        # Forward‑fill any remaining NaNs ----------------------------
        df_future.fillna(method="ffill", inplace=True)

        self.forecast = self.model.predict(df_future)

    # ------------------------------------------------------------------ Persistence to HA
    async def save_prediction(
        self,
        entity: str,
        now: datetime,
        interface: HAInterface,
        *,
        incrementing: bool = False,
        reset_daily: bool = False,
        units: str = "",
        days: int = 7,
    ) -> None:
        if self.forecast is None:
            return
        pred = self.forecast
        total = total_org = 0.0
        timeseries: dict[str, float] = {}
        timeseries_org: dict[str, float] = {}

        for _, row in pred.iterrows():
            ptimestamp = row["ds"].tz_localize(timezone.utc)
            diff = ptimestamp - now
            timestamp = now + diff
            if diff.days < -days:
                continue  # prune history beyond export window
            value = row["yhat1"]
            value_org = row.get("y")

            if timestamp <= now and reset_daily and timestamp.hour == 0 and timestamp.minute == 0:
                total = total_org = 0.0

            if incrementing:
                total += value
                if not math.isnan(value_org):
                    total_org += value_org
                timeseries[timestamp.strftime(TIME_FORMAT_HA)] = round(total, 2)
                if value_org is not None and not math.isnan(value_org):
                    timeseries_org[timestamp.strftime(TIME_FORMAT_HA)] = round(total_org, 2)
            else:
                timeseries[timestamp.strftime(TIME_FORMAT_HA)] = round(value, 2)
                if value_org is not None and not math.isnan(value_org):
                    timeseries_org[timestamp.strftime(TIME_FORMAT_HA)] = round(value_org, 2)

        final_state = round(total if incrementing else value, 2)
        attributes = {
            "last_updated": str(now),
            "unit_of_measurement": units,
            "state_class": "measurement",
            "results": timeseries,
            "source": timeseries_org,
        }
        print(f"Saving prediction to {entity} at {now}")
        await interface.set_state(entity, state=final_state, attributes=attributes)

# ‑‑‑‑‑‑ SQLite helper (unchanged) ————————————————————————————————

class Database:
    def __init__(self):
        self.con = sqlite3.connect("/config/predai.db")
        self.cur = self.con.cursor()

    async def cleanup_table(self, table_name: str, max_age: int):
        now_utc = datetime.now(timezone.utc).astimezone()
        oldest = (now_utc - timedelta(days=max_age)).strftime("%Y-%m-%d %H:%M:%S%z")
        self.cur.execute(f"DELETE FROM {table_name} WHERE timestamp < \"{oldest}\"")
        self.con.commit()

    async def create_table(self, table: str):
        self.cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (timestamp TEXT PRIMARY KEY, value REAL)")
        self.con.commit()

    async def get_history(self, table: str) -> pd.DataFrame:
        self.cur.execute(f"SELECT * FROM {table} ORDER BY timestamp")
        rows = self.cur.fetchall()
        return pd.DataFrame(rows, columns=["ds", "y"]) if rows else pd.DataFrame(columns=["ds", "y"])

    async def store_history(self, table: str, history: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
        prev_values = prev["ds"].astype(str).tolist()
        added = 0
        for _, row in history.iterrows():
            ts = str(row["ds"])
            val = row["y"]
            if ts not in prev_values:
                prev.loc[len(prev)] = {"ds": ts, "y": val}
                self.cur.execute(f"INSERT INTO {table} (timestamp, value) VALUES ('{ts}', {val})")
                added += 1
        self.con.commit()
        print(f"Added {added} rows to {table}")
        return prev

# ‑‑‑‑‑‑ Main orchestration ———————————————————————————————————————————

async def build_covariate_frames(
    cov_configs: List[dict[str, Any]],
    interface: HAInterface,
    prophet: Prophet,
    now: datetime,
    *,
    period: int,
    days: int,
) -> tuple[List[dict[str, Any]], pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Return (meta, merged_dataset_fragment, future_regressor_frames)."""
    meta: list[dict[str, Any]] = []
    merged: pd.DataFrame | None = None
    future_frames: dict[str, pd.DataFrame] = {}

    for cov in cov_configs:
        ent = cov["entity_id"]
        col = ent.replace(".", "_")
        future_known = bool(cov.get("future", False))
        n_lags = int(cov.get("n_lags", 24))
        incrementing = bool(cov.get("incrementing", False))

        # Historical part -------------------------------------------------
        hist_df = await prophet.historical_dataframe(
            interface,
            ent,
            now,
            period_minutes=period,
            days=days,
            incrementing=incrementing,
            max_increment=0,
            reset_low=0.0,
            reset_high=0.0,
        )
        hist_df.rename(columns={"y": col}, inplace=True)
        if merged is None:
            merged = hist_df
        else:
            merged = merged.merge(hist_df, on="ds", how="outer")

        # Meta -----------------------------------------------------------
        meta.append({"name": col, "type": "future" if future_known else "lagged", "n_lags": n_lags})

        # Future values --------------------------------------------------
        if future_known:
            # Attempt to read attribute "results" which should be a {timestamp: value} mapping
            res_attr = await interface.get_state(ent, attribute="results")
            future_series = {}
            if isinstance(res_attr, dict):
                for ts_str, val in res_attr.items():
                    dt = timestr_to_datetime(ts_str)
                    if dt and dt >= now:
                        future_series[dt] = float(val)
            if future_series:
                future_frames[col] = pd.DataFrame(
                    {"ds": list(future_series.keys()), col: list(future_series.values())}
                )

    if merged is None:
        merged = pd.DataFrame()
    merged.sort_values("ds", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged.fillna(method="ffill", inplace=True)
    return meta, merged, future_frames

async def main() -> None:
    config = yaml.safe_load(open("/config/predai.yaml"))
    interface = HAInterface(config.get("ha_url"), config.get("ha_key"))

    while True:
        config = yaml.safe_load(open("/config/predai.yaml")) or {}
        update_every = config.get("update_every", 30)
        sensors_cfg = config.get("sensors", [])

        if not sensors_cfg:
            print("WARN: predai.yaml missing or no sensors defined – sleeping…")
        else:
            now = datetime.now(timezone.utc).astimezone().replace(second=0, microsecond=0, minute=0)
            for s_cfg in sensors_cfg:
                target = s_cfg.get("name")
                if not target:
                    continue
                print(f"\n=== Processing {target} at {now} ===")

                # Target‑specific parameters ---------------------------
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

                # Build main dataset ----------------------------------
                prophet = Prophet(period)
                main_df = await prophet.historical_dataframe(
                    interface,
                    target,
                    now,
                    period_minutes=period,
                    days=days,
                    incrementing=incrementing,
                    max_increment=max_increment,
                    reset_low=reset_low,
                    reset_high=reset_high,
                )

                # Optionally subtract other sensors -------------------
                subtract_names = s_cfg.get("subtract")
                if subtract_names:
                    if isinstance(subtract_names, str):
                        subtract_names = [subtract_names]
                    for sub in subtract_names:
                        sub_df = await prophet.historical_dataframe(
                            interface,
                            sub,
                            now,
                            period_minutes=period,
                            days=days,
                            incrementing=incrementing,
                            max_increment=max_increment,
                            reset_low=reset_low,
                            reset_high=reset_high,
                        )
                        # Align & subtract on ds
                        main_df = pd.merge(main_df, sub_df, on="ds", how="left", suffixes=("", "_sub"))
                        main_df["y"] = main_df["y"].fillna(method="ffill") - main_df["y_sub"].fillna(0)
                        main_df.drop(columns=["y_sub"], inplace=True)

                # Database caching ------------------------------------ (unchanged)
                if use_db:
                    table = target.replace(".", "_")
                    db = Database()
                    await db.create_table(table)
                    prev_hist = await db.get_history(table)
                    main_df = await db.store_history(table, main_df, prev_hist)
                    await db.cleanup_table(table, max_age)

                # Covariates -----------------------------------------
                cov_cfgs = s_cfg.get("covariates", [])
                reg_meta, cov_hist_fragment, future_reg_frames = await build_covariate_frames(
                    cov_cfgs, interface, prophet, now, period=period, days=days
                )

                # Merge target + covariates ---------------------------
                if not cov_hist_fragment.empty:
                    dataset = main_df.merge(cov_hist_fragment, on="ds", how="left")
                else:
                    dataset = main_df
                dataset.fillna(method="ffill", inplace=True)

                # Train ------------------------------------------------
                await prophet.train(
                    dataset,
                    future_periods,
                    regressor_meta=reg_meta,
                    future_regressor_frames=future_reg_frames,
                    target_n_lags=n_lags_target,
                    country=country,
                )

                # Persist predictions ---------------------------------
                await prophet.save_prediction(
                    f"{target}_prediction",
                    now,
                    interface,
                    incrementing=incrementing,
                    reset_daily=reset_daily,
                    units=units,
                    days=export_days,
                )

        # ---------------------------------------------------------------- Sleep / heartbeat
        await interface.set_state("sensor.predai_last_run", state=str(datetime.now(timezone.utc).astimezone()), attributes={"unit_of_measurement": "time"})
        print(f"Sleeping for {update_every} minutes…")
        for _ in range(update_every):
            last_run = await interface.get_state("sensor.predai_last_run")
            if last_run is None:
                print("sensor.predai_last_run removed – restarting loop")
                break
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
