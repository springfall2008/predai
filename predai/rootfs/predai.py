"""
PredAI ― Home Assistant prediction service
Altered to support *covariates* (additional sensors with historical values)
so that NeuralProphet can exploit them as lagged regressors.

New in this version
-------------------
* YAML: each main sensor may list `covariates`, e.g.

    sensors:
      - name: sensor.energy_total
        covariates:
          - sensor.temperature_outdoor
          - sensor.humidity_indoor

  — every covariate is fetched, resampled to the same period, merged with
  the target dataset and passed to the model.

* NeuralProphet: every covariate is added with `add_lagged_regressor`.
  This means **no future values are required** at prediction time.

Nothing else in the original workflow has been removed or renamed, so
existing configurations keep working.
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
import json
import ssl
import math
import yaml

TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"


# ────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────
def timestr_to_datetime(timestamp: str) -> datetime | None:
    """Convert a Home Assistant timestamp string to a timezone‑aware datetime."""
    try:
        start_time = datetime.strptime(timestamp, TIME_FORMAT_HA)
    except ValueError:
        try:
            start_time = datetime.strptime(timestamp, TIME_FORMAT_HA_DOT)
        except ValueError:
            start_time = None
    if start_time:
        start_time = start_time.replace(second=0, microsecond=0)
    return start_time


def sanitise_name(entity_id: str) -> str:
    """Produce a SQLite/column‑safe string from an HA entity_id."""
    return entity_id.replace(".", "_").replace("-", "_")


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

    async def get_events(self):
        return await self.api_call("/api/events")

    async def get_history(self, sensor: str, now: datetime, *, days: int = 7):
        """Return history plus its first/last timestamps for *sensor*."""
        start = now - timedelta(days=days)
        end = now
        print(
            f"Getting history for {sensor} start {start.strftime(TIME_FORMAT_HA)} "
            f"end {end.strftime(TIME_FORMAT_HA)}"
        )
        res = await self.api_call(
            f"/api/history/period/{start.strftime(TIME_FORMAT_HA)}",
            {"filter_entity_id": sensor, "end_time": end.strftime(TIME_FORMAT_HA)},
        )
        if res:
            res = res[0]
            start = timestr_to_datetime(res[0]["last_updated"])
            end = timestr_to_datetime(res[-1]["last_updated"])
        print(f"History for {sensor} starts {start} ends {end}")
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

    # ────────────────────────────────────────────────────────────
    #  Data preparation
    # ────────────────────────────────────────────────────────────
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
        """Convert raw HA history into a uniform DataFrame for training."""
        dataset = pd.DataFrame(columns=["ds", "y"])
        timenow = start_time.replace(second=0, microsecond=0, minute=0)
        data_index = 0
        data_len = len(new_data)
        total = 0
        last_value = None

        print(
            f"Process {sensor_name} start {start_time} end {end_time} "
            f"incrementing {incrementing} reset_low {reset_low} reset_high {reset_high}"
        )

        while timenow <= end_time and data_index < data_len:
            try:
                value = float(new_data[data_index]["state"])
                if last_value is None:
                    last_value = value
            except ValueError:
                if last_value is not None:
                    value = last_value
                else:
                    data_index += 1
                    continue

            last_updated = new_data[data_index]["last_updated"]
            start_time = timestr_to_datetime(last_updated)

            if incrementing:
                # Handle meter reset
                if value < last_value and value < reset_low and last_value > reset_high:
                    total += value
                else:
                    if max_increment and abs(value - last_value) > max_increment:
                        value = last_value
                    total = max(total + value - last_value, 0)
            last_value = value

            if not start_time or start_time < timenow:
                data_index += 1
                continue

            real_value = max(0, total) if incrementing else value
            if incrementing:
                total = 0
            dataset.loc[len(dataset)] = {"ds": timenow, "y": real_value}
            timenow += timedelta(minutes=self.period)

        # dataset.to_csv(f"/config/{sensor_name}.csv", index=False)
        return dataset, value

    # ────────────────────────────────────────────────────────────
    #  Model training & prediction
    # ────────────────────────────────────────────────────────────
    async def train(
        self,
        dataset: pd.DataFrame,
        future_periods: int,
        *,
        n_lags: int = 0,
        country: str | None = None,
        regressors: List[str] | None = None,
    ):
        """Fit NeuralProphet with optional lagged regressors."""
        self.model = NeuralProphet(
            n_lags=n_lags,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
        )
        if country:
            print(f"Adding country holidays for {country}")
            self.model.add_country_holidays(country)

        if regressors:
            for reg in regressors:
                # Lagged regressors allow future prediction without future covariate values
                self.model.add_lagged_regressor(reg)
                print(f"Added lagged regressor: {reg}")

        self.metrics = self.model.fit(dataset, freq=f"{self.period}min", progress=None)
        self.df_future = self.model.make_future_dataframe(
            dataset, n_historic_predictions=True, periods=future_periods
        )
        self.forecast = self.model.predict(self.df_future)
        print(self.forecast.tail())

    # ────────────────────────────────────────────────────────────
    #  Save prediction back to Home Assistant
    # ────────────────────────────────────────────────────────────
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

            time_str = timestamp.strftime(TIME_FORMAT_HA)
            value = row["yhat1"]
            value_org = row["y"]

            if timestamp <= now and reset_daily and timestamp.hour == 0 and timestamp.minute == 0:
                total = total_org = 0

            total += value
            if not math.isnan(value_org):
                total_org += value_org
            else:
                value_org = None

            if diff.days < -days:
                continue

            if incrementing:
                timeseries[time_str] = round(total, 2)
                if value_org is not None:
                    timeseries_org[time_str] = round(total_org, 2)
            else:
                timeseries[time_str] = round(value, 2)
                if value_org is not None:
                    timeseries_org[time_str] = round(value_org, 2)

        final = total if incrementing else value
        attributes = {
            "last_updated": str(now),
            "unit_of_measurement": units,
            "state_class": "measurement",
            "results": timeseries,
            "source": timeseries_org,
        }
        print(f"Saving prediction to {entity} at {now}")
        await interface.set_state(entity, state=round(final, 2), attributes=attributes)


# ────────────────────────────────────────────────────────────────
#  Utility functions for dataset manipulation
# ────────────────────────────────────────────────────────────────
async def subtract_set(
    dataset: pd.DataFrame,
    subset: pd.DataFrame,
    *,
    incrementing: bool = False,
) -> pd.DataFrame:
    """Subtract *subset* from *dataset* row by row (matched by timestamp)."""
    pruned = pd.DataFrame(columns=["ds", "y"])
    count = 0
    for _, row in dataset.iterrows():
        ds = row["ds"]
        value = row["y"]
        car_row = subset.loc[subset["ds"] == ds]
        car_value = car_row["y"].values[0] if not car_row.empty else 0
        count += int(not car_row.empty)

        value = max(value - car_value, 0) if incrementing else value - car_value
        pruned.loc[len(pruned)] = {"ds": ds, "y": value}

    print(f"Subtracted {count} rows into new set (len={len(pruned)})")
    return pruned


# ────────────────────────────────────────────────────────────────
#  SQLite helper
# ────────────────────────────────────────────────────────────────
class Database:
    def __init__(self):
        self.con = sqlite3.connect("/config/predai.db")
        self.cur = self.con.cursor()

    async def cleanup_table(self, table_name: str, max_age: int):
        oldest_stamp = (
            datetime.now(timezone.utc).astimezone() - timedelta(days=max_age)
        ).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"Cleaning {table_name}: deleting rows older than {oldest_stamp}")
        self.cur.execute(f'DELETE FROM {table_name} WHERE timestamp < "{oldest_stamp}"')
        self.con.commit()

    async def create_table(self, table: str):
        print(f"Ensuring table {table} exists")
        self.cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (timestamp TEXT PRIMARY KEY, value REAL)")
        self.con.commit()

    async def get_history(self, table: str) -> pd.DataFrame:
        self.cur.execute(f"SELECT * FROM {table} ORDER BY timestamp")
        rows = self.cur.fetchall()
        history = pd.DataFrame(columns=["ds", "y"])
        for ts, val in rows:
            history.loc[len(history)] = {"ds": ts, "y": val}
        return history

    async def store_history(self, table: str, history: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
        added = 0
        prev_stamps = set(prev["ds"].values.tolist())
        for _, row in history.iterrows():
            ts, val = str(row["ds"]), row["y"]
            if ts not in prev_stamps:
                prev.loc[len(prev)] = {"ds": ts, "y": val}
                self.cur.execute(f"INSERT INTO {table} (timestamp, value) VALUES ('{ts}', {val})")
                added += 1
        self.con.commit()
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
    """Fetch & (optionally) persist history for a sensor."""
    raw_set, start, end = await interface.get_history(sensor_name, now, days=days)
    dataset, _ = await nw.process_dataset(
        sensor_name,
        raw_set,
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
        print(f"{sensor_name}: history length now {len(dataset)} (database combined)")
    return dataset, start, end


# ────────────────────────────────────────────────────────────────
#  Main loop
# ────────────────────────────────────────────────────────────────
async def main():
    interface = HAInterface(None, None)  # Defaults taken from environment/the YAML
    while True:
        config = yaml.safe_load(open("/config/predai.yaml"))
        if not config:
            print("WARN: predai.yaml missing; sleeping one minute.")
            await asyncio.sleep(60)
            continue

        update_every = config.get("update_every", 30)
        sensors = config.get("sensors", [])
        print(f"Loaded configuration for {len(sensors)} sensors")

        for sensor in sensors:
            sensor_name: str | None = sensor.get("name")
            if not sensor_name:
                continue

            subtract_names = sensor.get("subtract")
            covariate_names = sensor.get("covariates", []) or []

            # Baseline parameters
            days = sensor.get("days", 7)
            export_days = sensor.get("export_days", days)
            incrementing = sensor.get("incrementing", False)
            reset_daily = sensor.get("reset_daily", False)
            interval = sensor.get("interval", 30)
            units = sensor.get("units", "")
            future_periods = sensor.get("future_periods", 96)
            use_db = sensor.get("database", True)
            reset_low = sensor.get("reset_low", 1.0)
            reset_high = sensor.get("reset_high", 2.0)
            max_increment = sensor.get("max_increment", 0)
            n_lags = sensor.get("n_lags", 0)
            country = sensor.get("country")
            max_age = sensor.get("max_age", 365)

            nw = Prophet(interval)
            now = datetime.now(timezone.utc).astimezone().replace(second=0, microsecond=0, minute=0)

            print(
                f"\n=== [{now}] Processing {sensor_name} "
                f"(incrementing={incrementing}, interval={interval}m) ==="
            )

            # ────────────── fetch history for target
            dataset, start, end = await get_history(
                interface,
                nw,
                sensor_name,
                now,
                incrementing=incrementing,
                max_increment=max_increment,
                days=days,
                use_db=use_db,
                reset_low=reset_low,
                reset_high=reset_high,
                max_age=max_age,
            )

            # ────────────── subtract datasets (e.g. charging energy)
            if subtract_names:
                if isinstance(subtract_names, str):
                    subtract_names = [subtract_names]
                for name in subtract_names:
                    sub_ds, *_ = await get_history(
                        interface,
                        nw,
                        name,
                        now,
                        incrementing=incrementing,
                        max_increment=max_increment,
                        days=days,
                        use_db=use_db,
                        reset_low=reset_low,
                        reset_high=reset_high,
                        max_age=max_age,
                    )
                    dataset = await subtract_set(dataset, sub_ds, incrementing=incrementing)

            # ────────────── fetch & merge covariates
            covariate_columns: List[str] = []
            for cov in covariate_names:
                cov_ds, *_ = await get_history(
                    interface,
                    nw,
                    cov,
                    now,
                    incrementing=False,  # covariates default to absolute
                    max_increment=0,
                    days=days,
                    use_db=use_db,
                    reset_low=0,
                    reset_high=0,
                    max_age=max_age,
                )
                col_name = sanitise_name(cov)
                covariate_columns.append(col_name)
                cov_ds = cov_ds.rename(columns={"y": col_name})
                dataset = pd.merge(dataset, cov_ds, on="ds", how="left")

            # Forward‑fill any gaps in regressors (target *y* is assumed complete)
            if covariate_columns:
                dataset.sort_values("ds", inplace=True)
                dataset[covariate_columns] = dataset[covariate_columns].ffill().bfill()

            # ────────────── train & predict
            await nw.train(
                dataset,
                future_periods,
                n_lags=n_lags,
                country=country,
                regressors=covariate_columns,
            )

            # ────────────── save prediction
            await nw.save_prediction(
                sensor_name + "_prediction",
                now,
                interface,
                start=end,
                incrementing=incrementing,
                reset_daily=reset_daily,
                units=units,
                days=export_days,
            )

        # Book‑keeping
        time_now = datetime.now(timezone.utc).astimezone()
        await interface.set_state(
            "sensor.predai_last_run",
            state=str(time_now),
            attributes={"unit_of_measurement": "time"},
        )
        print(f"Sleeping {update_every} minutes (until next cycle)\n")
        for _ in range(update_every):
            # Allows graceful restart if HA clears the last_run sensor
            if await interface.get_state("sensor.predai_last_run") is None:
                print("Last‑run state vanished; restarting loop.")
                break
            await asyncio.sleep(60)


# ────────────────────────────────────────────────────────────────
#  Entrypoint
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
