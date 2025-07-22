"""
PredAI ― Home Assistant prediction service
Re‑revised to
* guarantee NeuralProphet receives a pure‑datetime `ds` column
* honour YAML key `cov_n_lags` when adding lagged regressors
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta, timezone
from neuralprophet import NeuralProphet, set_log_level
import os
import aiohttp
import requests
import asyncio
import math
import yaml

TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"


# ────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────
def timestr_to_datetime(timestamp: str) -> datetime | None:
    """Convert an HA timestamp string to timezone‑aware datetime (seconds, µs floored)."""
    try:
        start_time = datetime.strptime(timestamp, TIME_FORMAT_HA)
    except ValueError:
        try:
            start_time = datetime.strptime(timestamp, TIME_FORMAT_HA_DOT)
        except ValueError:
            return None
    return start_time.replace(second=0, microsecond=0)


def sanitise_name(entity_id: str) -> str:
    """Produce a SQLite/column‑safe string from an HA entity_id."""
    return entity_id.replace(".", "_").replace("-", "_")


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Force df['ds'] to be datetime (timezone‑naïve); drop rows where conversion fails."""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], utc=True, errors="coerce").dt.tz_convert(None)
    return df.dropna(subset=["ds"])


# ────────────────────────────────────────────────────────────────
#  Home Assistant interface
# ────────────────────────────────────────────────────────────────
class HAInterface:
    def __init__(self, ha_url: str | None, ha_key: str | None):
        self.ha_url = ha_url or "http://supervisor/core"
        self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            print("No Home Assistant key found, exiting")
            exit(1)
        print(f"HA Interface started (key ****, url {self.ha_url})")

    async def get_history(self, sensor: str, now: datetime, *, days: int = 7):
        start = now - timedelta(days=days)
        end = now
        res = await self.api_call(
            f"/api/history/period/{start.strftime(TIME_FORMAT_HA)}",
            {"filter_entity_id": sensor, "end_time": end.strftime(TIME_FORMAT_HA)},
        )
        if res:
            res = res[0]
            start = timestr_to_datetime(res[0]["last_updated"])
            end = timestr_to_datetime(res[-1]["last_updated"])
        return res, start, end

    async def get_state(self, entity_id: str, default=None, attribute: str | None = None):
        item = await self.api_call(f"/api/states/{entity_id}")
        if not item:
            return default
        return item.get("attributes", {}).get(attribute, default) if attribute else item.get("state", default)

    async def set_state(self, entity_id: str, state: Any, attributes: dict | None = None):
        data = {"state": state}
        if attributes:
            data["attributes"] = attributes
        await self.api_call(f"/api/states/{entity_id}", data, post=True)

    async def api_call(self, endpoint: str, datain: dict | None = None, *, post: bool = False):
        url = self.ha_url + endpoint
        headers = {
            "Authorization": "Bearer " + self.ha_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if post:
            response = await asyncio.to_thread(
                requests.post, url, headers=headers, json=datain, timeout=TIMEOUT
            )
        else:
            response = await asyncio.to_thread(
                requests.get, url, headers=headers, params=datain, timeout=TIMEOUT
            )
        try:
            return response.json()
        except (requests.exceptions.JSONDecodeError, requests.Timeout, requests.exceptions.ReadTimeout):
            print(f"Failed/timeout whilst calling {url}")
            return None


# ────────────────────────────────────────────────────────────────
#  Prophet wrapper
# ────────────────────────────────────────────────────────────────
class Prophet:
    def __init__(self, period: int = 30):
        set_log_level("ERROR")
        self.period = period  # minutes between rows

    async def process_dataset(
        self,
        sensor_name: str,
        new_data: list,
        start_time: datetime,
        end_time: datetime,
        *,
        incrementing: bool = False,
        max_increment: float = 0,
        reset_low: float = 0.0,
        reset_high: float = 0.0,
    ) -> Tuple[pd.DataFrame, float]:
        dataset = pd.DataFrame(columns=["ds", "y"])
        timenow = start_time.replace(second=0, microsecond=0, minute=0)
        idx = 0
        last_val = None
        total = 0
        while timenow <= end_time and idx < len(new_data):
            try:
                value = float(new_data[idx]["state"])
                if last_val is None:
                    last_val = value
            except ValueError:
                idx += 1
                continue

            last_updated = timestr_to_datetime(new_data[idx]["last_updated"])
            if incrementing:
                if value < last_val and value < reset_low and last_val > reset_high:
                    total += value
                else:
                    if max_increment and abs(value - last_val) > max_increment:
                        value = last_val
                    total = max(total + value - last_val, 0)
            last_val = value

            if not last_updated or last_updated < timenow:
                idx += 1
                continue

            dataset.loc[len(dataset)] = {
                "ds": timenow,
                "y": max(0, total) if incrementing else value,
            }
            if incrementing:
                total = 0
            timenow += timedelta(minutes=self.period)
        return dataset, value

    async def train(
        self,
        dataset: pd.DataFrame,
        future_periods: int,
        *,
        n_lags: int = 0,
        country: str | None = None,
        regressors: List[str] | None = None,
        reg_n_lags: int = 0,
        future_regressors: List[str] | None = None,
        future_values: Dict[str, float] | None = None,
    ):
        self.model = NeuralProphet(
            n_lags=n_lags,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
        )
        if country:
            self.model.add_country_holidays(country)
        if regressors:
            for reg in regressors:
                self.model.add_lagged_regressor(reg, n_lags=reg_n_lags)
        if future_regressors:
            for reg in future_regressors:
                self.model.add_future_regressor(reg)

        # ─── ensure pure datetime, then drop any duplicate ds rows ───
        dataset = ensure_datetime(dataset)
        dataset = (
            dataset
            .sort_values("ds")
            .drop_duplicates(subset="ds", keep="last")
            .reset_index(drop=True)
        )

        self.metrics = self.model.fit(dataset, freq=f"{self.period}min", progress=None)
        self.df_future = self.model.make_future_dataframe(
            dataset, n_historic_predictions=True, periods=future_periods
        )
        if future_regressors:
            fut_mask = self.df_future["ds"] > dataset["ds"].max()
            for reg in future_regressors:
                val = None
                if future_values and reg in future_values:
                    val = future_values[reg]
                else:
                    val = dataset[reg].iloc[-1] if reg in dataset.columns else 0.0
                self.df_future.loc[fut_mask, reg] = val
        self.forecast = self.model.predict(self.df_future)


    async def save_prediction(
        self,
        entity: str,
        now: datetime,
        interface: "HAInterface",
        *,
        start: datetime,
        incrementing: bool = False,
        reset_daily: bool = False,
        units: str = "",
        days: int = 7,
    ):
        pred = self.forecast
        total = total_org = 0
        timeseries: Dict[str, float] = {}
        timeseries_org: Dict[str, float] = {}

        for _, row in pred.iterrows():
            ptimestamp = row["ds"].tz_localize(timezone.utc)
            diff = ptimestamp - now
            timestamp = now + diff
            if diff.days < -days:
                continue

            val = row["yhat1"]
            val_org = row["y"]
            if timestamp <= now and reset_daily and timestamp.hour == 0:
                total = total_org = 0
            total += val
            if not math.isnan(val_org):
                total_org += val_org

            key = timestamp.strftime(TIME_FORMAT_HA)
            if incrementing:
                timeseries[key] = round(total, 2)
                timeseries_org[key] = round(total_org, 2)
            else:
                timeseries[key] = round(val, 2)
                timeseries_org[key] = round(val_org, 2) if not math.isnan(val_org) else None

        final = total if incrementing else val
        attrs = {
            "last_updated": str(now),
            "unit_of_measurement": units,
            "state_class": "measurement",
            "results": timeseries,
            "source": timeseries_org,
        }
        await interface.set_state(entity, round(final, 2), attrs)


# ────────────────────────────────────────────────────────────────
#  SQLite helper
# ────────────────────────────────────────────────────────────────
class Database:
    def __init__(self):
        self.con = sqlite3.connect("/config/predai.db")
        self.cur = self.con.cursor()

    async def cleanup_table(self, table_name: str, max_age: int):
        cutoff = (
            datetime.now(timezone.utc).astimezone() - timedelta(days=max_age)
        ).strftime("%Y-%m-%d %H:%M:%S%z")
        self.cur.execute(f'DELETE FROM {table_name} WHERE timestamp < "{cutoff}"')
        self.con.commit()

    async def create_table(self, table: str):
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (timestamp TEXT PRIMARY KEY, value REAL)"
        )
        self.con.commit()

    async def get_history(self, table: str) -> pd.DataFrame:
        self.cur.execute(f"SELECT * FROM {table} ORDER BY timestamp")
        rows = self.cur.fetchall()
        hist = pd.DataFrame(rows, columns=["ds", "y"])
        return ensure_datetime(hist)

    async def store_history(self, table: str, new_rows: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
        """
        Store the history in the database, skipping duplicates.
        Normalises both DataFrames so that 'ds' is tz‑naïve datetime64[ns].
        """
        # ── Step 1: force both DataFrames to tz‑naïve datetime64 ──
        prev = prev.copy()
        prev["ds"] = (
            pd.to_datetime(prev["ds"], utc=True, errors="coerce")
              .dt.tz_convert(None)
        )

        new_rows = new_rows.copy()
        new_rows["ds"] = (
            pd.to_datetime(new_rows["ds"], utc=True, errors="coerce")
              .dt.tz_convert(None)
        )

        added = 0
        prev_stamps = set(prev["ds"].astype(str))
        for _, row in new_rows.iterrows():
            ts: pd.Timestamp = row["ds"]
            val: float = row["y"]
            # use INSERT OR IGNORE to skip existing timestamps
            self.cur.execute(
                f"INSERT OR IGNORE INTO {table} (timestamp, value) VALUES (?, ?)",
                (ts.strftime("%Y-%m-%d %H:%M:%S"), val),
            )
            if ts.strftime("%Y-%m-%d %H:%M:%S") not in prev_stamps and self.cur.rowcount > 0:
                # now that prev['ds'] is datetime64, assigning ts (a pd.Timestamp) is safe
                prev.loc[len(prev)] = {"ds": ts, "y": val}
                added += 1

        self.con.commit()
        if added:
            print(f"Added {added} new rows to {table}")
        return prev



# ────────────────────────────────────────────────────────────────
#  History acquisition
# ────────────────────────────────────────────────────────────────
async def get_history(
    interface: HAInterface,
    nw: Prophet,
    sensor_name: str,
    now: datetime,
    *,
    incrementing: bool,
    max_increment: float,
    days: int,
    use_db: bool,
    reset_low: float,
    reset_high: float,
    max_age: int,
) -> Tuple[pd.DataFrame, datetime, datetime]:
    raw, start, end = await interface.get_history(sensor_name, now, days=days)
    dataset, _ = await nw.process_dataset(
        sensor_name,
        raw,
        start,
        end,
        incrementing=incrementing,
        max_increment=max_increment,
        reset_low=reset_low,
        reset_high=reset_high,
    )

    if use_db:
        table = sanitise_name(sensor_name)
        db = Database()
        await db.create_table(table)
        prev = await db.get_history(table)
        dataset = await db.store_history(table, dataset, prev)
        await db.cleanup_table(table, max_age)
    return ensure_datetime(dataset), start, end


# ────────────────────────────────────────────────────────────────
#  Main loop
# ────────────────────────────────────────────────────────────────
async def main():
    cfg = yaml.safe_load(open("/config/predai.yaml"))
    interface = HAInterface(cfg.get("ha_url"), cfg.get("ha_key"))
    while True:
        cfg = yaml.safe_load(open("/config/predai.yaml"))
        if not cfg:
            await asyncio.sleep(60)
            continue

        update_every = cfg.get("update_every", 30)
        for sensor in cfg.get("sensors", []):
            name = sensor.get("name")
            if not name:
                continue

            subtract_names = sensor.get("subtract") or []
            if isinstance(subtract_names, str):
                subtract_names = [subtract_names]

            covars = sensor.get("covariates", []) or []
            future_covars = sensor.get("future_covariates", []) or []
            cov_n_lags = sensor.get("cov_n_lags", 0)

            nw = Prophet(sensor.get("interval", 30))
            now = datetime.now(timezone.utc).astimezone().replace(second=0, microsecond=0, minute=0)

            base_ds, *_ = await get_history(
                interface,
                nw,
                name,
                now,
                incrementing=sensor.get("incrementing", False),
                max_increment=sensor.get("max_increment", 0),
                days=sensor.get("days", 7),
                use_db=sensor.get("database", True),
                reset_low=sensor.get("reset_low", 1.0),
                reset_high=sensor.get("reset_high", 2.0),
                max_age=sensor.get("max_age", 365),
            )

            for sub in subtract_names:
                sub_ds, *_ = await get_history(
                    interface,
                    nw,
                    sub,
                    now,
                    incrementing=sensor.get("incrementing", False),
                    max_increment=sensor.get("max_increment", 0),
                    days=sensor.get("days", 7),
                    use_db=sensor.get("database", True),
                    reset_low=sensor.get("reset_low", 1.0),
                    reset_high=sensor.get("reset_high", 2.0),
                    max_age=sensor.get("max_age", 365),
                )
                base_ds = await subtract_set(
                    base_ds, sub_ds, incrementing=sensor.get("incrementing", False)
                )

            cov_cols: List[str] = []
            future_cols: List[str] = []
            future_vals: Dict[str, float] = {}

            for cov in covars:
                cov_ds, *_ = await get_history(
                    interface,
                    nw,
                    cov,
                    now,
                    incrementing=False,
                    max_increment=0,
                    days=sensor.get("days", 7),
                    use_db=sensor.get("database", True),
                    reset_low=0,
                    reset_high=0,
                    max_age=sensor.get("max_age", 365),
                )
                col = sanitise_name(cov)
                cov_cols.append(col)
                base_ds = pd.merge(
                    base_ds,
                    cov_ds.rename(columns={"y": col}),
                    on="ds",
                    how="left",
                )

            for cov in future_covars:
                cov_ds, *_ = await get_history(
                    interface,
                    nw,
                    cov,
                    now,
                    incrementing=False,
                    max_increment=0,
                    days=sensor.get("days", 7),
                    use_db=sensor.get("database", True),
                    reset_low=0,
                    reset_high=0,
                    max_age=sensor.get("max_age", 365),
                )
                col = sanitise_name(cov)
                future_cols.append(col)
                base_ds = pd.merge(
                    base_ds,
                    cov_ds.rename(columns={"y": col}),
                    on="ds",
                    how="left",
                )
                val = await interface.get_state(cov)
                try:
                    future_vals[col] = float(val)
                except (TypeError, ValueError):
                    future_vals[col] = 0.0
            if cov_cols:
                base_ds.sort_values("ds", inplace=True)
                base_ds[cov_cols] = base_ds[cov_cols].ffill().bfill()
            if future_cols:
                base_ds.sort_values("ds", inplace=True)
                base_ds[future_cols] = base_ds[future_cols].ffill().bfill()

            await nw.train(
                base_ds,
                sensor.get("future_periods", 96),
                n_lags=sensor.get("n_lags", 0),
                country=sensor.get("country"),
                regressors=cov_cols,
                reg_n_lags=cov_n_lags,
                future_regressors=future_cols,
                future_values=future_vals,
            )

            await nw.save_prediction(
                name + "_prediction",
                now,
                interface,
                start=now,
                incrementing=sensor.get("incrementing", False),
                reset_daily=sensor.get("reset_daily", False),
                units=sensor.get("units", ""),
                days=sensor.get("export_days", sensor.get("days", 7)),
            )

        await interface.set_state(
            "sensor.predai_last_run",
            str(datetime.now(timezone.utc).astimezone()),
            {"unit_of_measurement": "time"},
        )
        for _ in range(update_every):
            if await interface.get_state("sensor.predai_last_run") is None:
                break
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
