from typing import Any
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

# Constants
TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"


def timestr_to_datetime(timestamp: str) -> datetime:
    """
    Convert a Home Assistant timestamp string to a datetime object.
    """
    try:
        dt = datetime.strptime(timestamp, TIME_FORMAT_HA)
    except ValueError:
        try:
            dt = datetime.strptime(timestamp, TIME_FORMAT_HA_DOT)
        except ValueError:
            return None
    # Round down to minute precision
    return dt.replace(second=0, microsecond=0)


class HAInterface:
    def __init__(self, ha_url: str = None, ha_key: str = None):
        self.ha_url = ha_url or "http://supervisor/core"
        self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            print("No Home Assistant key found, exiting")
            exit(1)
        print(f"HA Interface started key {self.ha_key} url {self.ha_url}")

    async def api_call(self, endpoint: str, datain: Any = None, post: bool = False) -> Any:
        url = self.ha_url + endpoint
        headers = {
            "Authorization": f"Bearer {self.ha_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            if post:
                resp = await asyncio.to_thread(requests.post, url, headers=headers, json=datain, timeout=TIMEOUT)
            else:
                resp = await asyncio.to_thread(requests.get, url, headers=headers, params=datain, timeout=TIMEOUT)
            return resp.json()
        except Exception as e:
            print(f"API call error at {url}: {e}")
            return None

    async def get_history(self, sensor: str, now: datetime, days: int = 7):
        """
        Fetch raw history events for a sensor from Home Assistant over the past 'days'.
        Returns (events_list, start_dt, end_dt).
        """
        start = now - timedelta(days=days)
        end = now
        print(f"Getting history for {sensor} from {start} to {end}")
        res = await self.api_call(
            f"/api/history/period/{start.strftime(TIME_FORMAT_HA)}",
            {"filter_entity_id": sensor, "end_time": end.strftime(TIME_FORMAT_HA)}
        )
        if not res:
            return [], start, end
        events = res[0]
        # Determine true start/end from data
        start_dt = timestr_to_datetime(events[0]["last_updated"])
        end_dt = timestr_to_datetime(events[-1]["last_updated"])
        return events, start_dt, end_dt

    async def get_state(self, entity_id: str, default: Any = None, attribute: str = None) -> Any:
        item = await self.api_call(f"/api/states/{entity_id}")
        if not item:
            return default
        if attribute:
            return item.get("attributes", {}).get(attribute, default)
        return item.get("state", default)

    async def set_state(self, entity_id: str, state: Any, attributes: dict = None):
        data = {"state": state}
        if attributes:
            data["attributes"] = attributes
        await self.api_call(f"/api/states/{entity_id}", data, post=True)


class Prophet:
    def __init__(self, period: int = 30):
        set_log_level("ERROR")
        self.period = period

    async def process_dataset(
        self,
        sensor_name: str,
        raw_events: list,
        start_time: datetime,
        end_time: datetime,
        incrementing: bool = False,
        max_increment: float = 0,
        reset_low: float = 0.0,
        reset_high: float = 0.0
    ) -> (pd.DataFrame, float):
        """
        Build a uniformly spaced (ds, y) time series from raw HA events.
        Returns (dataset_df, last_value).
        """
        df = pd.DataFrame(columns=["ds", "y"])
        t_curr = start_time.replace(second=0, microsecond=0, minute=0)
        idx = 0
        total = 0.0
        last_val = None

        while t_curr <= end_time and idx < len(raw_events):
            ev = raw_events[idx]
            try:
                val = float(ev.get("state", 0))
                if last_val is None:
                    last_val = val
            except ValueError:
                val = last_val or 0.0

            t_event = timestr_to_datetime(ev.get("last_updated"))
            if not t_event or t_event < t_curr:
                idx += 1
                continue

            if incrementing:
                if val < last_val and val < reset_low and last_val > reset_high:
                    total += val
                else:
                    if max_increment and abs(val - last_val) > max_increment:
                        val = last_val
                    total = max(total + val - last_val, 0)
                real_val = total
                total = 0.0
            else:
                real_val = val

            df.loc[len(df)] = {"ds": t_curr, "y": real_val}
            last_val = val
            t_curr += timedelta(minutes=self.period)
            idx += 1

        return df, last_val

    async def train(
        self,
        dataset: pd.DataFrame,
        future_periods: int,
        n_lags: int = 0,
        country: str = None,
        covariates: list = None
    ):
        """Train NeuralProphet with optional holidays and regressors."""
        self.model = NeuralProphet(
            n_lags=n_lags,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        if country:
            print(f"Adding country holidays for {country}")
            self.model.add_country_holidays(country)
        if covariates:
            for cov in covariates:
                print(f"Adding regressor {cov}")
                self.model.add_regressor(cov)

        self.metrics = self.model.fit(dataset, freq=f"{self.period}min", progress=None)
        self.df_future = self.model.make_future_dataframe(
            dataset, n_historic_predictions=True, periods=future_periods
        )
        # Fill future covariate values
        if covariates:
            for cov in covariates:
                last_val = dataset[cov].iloc[-1] if cov in dataset.columns else 0
                self.df_future[cov] = self.df_future[cov].fillna(last_val)

        self.forecast = self.model.predict(self.df_future)

    async def save_prediction(
        self,
        entity: str,
        now: datetime,
        interface: HAInterface,
        start: datetime,
        incrementing: bool = False,
        reset_daily: bool = False,
        units: str = "",
        days: int = 7
    ):
        """Push forecast back into Home Assistant."""
        results = {}
        source = {}
        total = 0.0
        total_org = 0.0
        for _, row in self.forecast.iterrows():
            pt = row["ds"].tz_localize(timezone.utc)
            diff = pt - now
            if diff.days < -days:
                continue
            ts = (now + diff).strftime(TIME_FORMAT_HA)
            yhat = row.get("yhat1", np.nan)
            y_orig = row.get("y", np.nan)
            if incrementing:
                total += yhat
                if not math.isnan(y_orig):
                    total_org += y_orig
                results[ts] = round(total, 2)
                source[ts] = round(total_org, 2)
            else:
                results[ts] = round(yhat, 2)
                if not math.isnan(y_orig):
                    source[ts] = round(y_orig, 2)
        final = total if incrementing else yhat
        attrs = {
            "last_updated": str(now),
            "unit_of_measurement": units,
            "state_class": "measurement",
            "results": results,
            "source": source
        }
        print(f"Saving prediction to {entity}")
        await interface.set_state(entity, state=round(final, 2), attributes=attrs)


class Database:
    def __init__(self, path: str = "/config/predai.db"):
        self.con = sqlite3.connect(path)
        self.cur = self.con.cursor()

    async def create_table(self, table: str):
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (timestamp TEXT PRIMARY KEY, value REAL)"
        )
        self.con.commit()

    async def get_history(self, table: str) -> pd.DataFrame:
        self.cur.execute(f"SELECT * FROM {table} ORDER BY timestamp")
        rows = self.cur.fetchall()
        df = pd.DataFrame(columns=["ds", "y"])
        for ts, val in rows:
            df.loc[len(df)] = {"ds": timestr_to_datetime(ts), "y": val}
        return df

    async def store_history(
        self, table: str, new_data: pd.DataFrame, prev: pd.DataFrame
    ) -> pd.DataFrame:
        added = 0
        existing = prev["ds"].astype(str).tolist()
        for _, row in new_data.iterrows():
            ts = str(row["ds"])
            if ts not in existing:
                prev.loc[len(prev)] = {"ds": row["ds"], "y": row["y"]}
                self.cur.execute(
                    f"INSERT INTO {table} (timestamp, value) VALUES ('{ts}', {row['y']})"
                )
                added += 1
        self.con.commit()
        print(f"Added {added} rows to {table}")
        return prev

    async def cleanup_table(self, table: str, max_age: int):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age)).strftime("%Y-%m-%d %H:%M:%S%z")
        self.cur.execute(
            f"DELETE FROM {table} WHERE timestamp < '{cutoff}'"
        )
        self.con.commit()
        print(f"Cleaned {table}, older than {cutoff}")


async def subtract_set(
    dataset: pd.DataFrame,
    subset: pd.DataFrame,
    incrementing: bool = False
) -> pd.DataFrame:
    """Subtract subset y-values from dataset y-values on matching ds."""
    pruned = pd.DataFrame(columns=["ds", "y"])
    for _, row in dataset.iterrows():
        ds = row["ds"]
        y = row["y"]
        subrow = subset[subset["ds"] == ds]
        if not subrow.empty:
            y_sub = subrow["y"].iloc[0]
        else:
            y_sub = 0.0
        y_new = max(y - y_sub, 0) if incrementing else y - y_sub
        pruned.loc[len(pruned)] = {"ds": ds, "y": y_new}
    return pruned


async def print_dataset(name: str, dataset: pd.DataFrame):
    print(f"Dataset {name}:")
    for idx, row in dataset.head(24).iterrows():
        print(row["ds"], row["y"])


async def get_history(
    interface: HAInterface,
    nw: Prophet,
    sensor_name: str,
    now: datetime,
    incrementing: bool,
    max_increment: float,
    days: int,
    use_db: bool,
    reset_low: float,
    reset_high: float,
    max_age: int
):
    # Fetch raw events
    raw, start, end = await interface.get_history(sensor_name, now, days=days)
    # Build uniform dataset
    dataset, _ = await nw.process_dataset(
        sensor_name,
        raw,
        start,
        end,
        incrementing=incrementing,
        max_increment=max_increment,
        reset_low=reset_low,
        reset_high=reset_high
    )
    # Optionally merge DB history
    if use_db:
        table = sensor_name.replace('.', '_')
        db = Database()
        await db.create_table(table)
        prev = await db.get_history(table)
        dataset = await db.store_history(table, dataset, prev)
        await db.cleanup_table(table, max_age)
    return dataset, start, end


async def main():
    config = yaml.safe_load(open("/config/predai.yaml"))
    interface = HAInterface(config.get("ha_url"), config.get("ha_key"))
    while True:
        config = yaml.safe_load(open("/config/predai.yaml"))
        sensors = config.get("sensors", [])

        now = datetime.now(timezone.utc).replace(second=0, microsecond=0, minute=0)
        for sensor in sensors:
            name = sensor.get("name")
            if not name:
                continue
            interval = sensor.get("interval", 30)
            days = sensor.get("days", 7)
            incrementing = sensor.get("incrementing", False)
            max_increment = sensor.get("max_increment", 0)
            reset_low = sensor.get("reset_low", 0.0)
            reset_high = sensor.get("reset_high", 0.0)
            use_db = sensor.get("database", True)
            n_lags = sensor.get("n_lags", 0)
            country = sensor.get("country")
            future_periods = sensor.get("future_periods", 96)
            units = sensor.get("units", "")
            export_days = sensor.get("export_days", days)
            max_age = sensor.get("max_age", 365)
            covariates = sensor.get("covariates", [])

            nw = Prophet(interval)
            # Main history
            dataset, start, end = await get_history(
                interface, nw, name, now,
                incrementing, max_increment,
                days, use_db,
                reset_low, reset_high,
                max_age
            )
            # Subtract sensors
            subtract_names = sensor.get("subtract", [])
            if isinstance(subtract_names, str):
                subtract_names = [subtract_names]
            for sub in subtract_names:
                df_sub, _, _ = await get_history(
                    interface, nw, sub, now,
                    incrementing, max_increment,
                    days, use_db,
                    reset_low, reset_high,
                    max_age
                )
                dataset = await subtract_set(dataset, df_sub, incrementing=incrementing)
            # Covariate merge
            if covariates:
                cov_frames = {}
                for cov in covariates:
                    df_cov, _, _ = await get_history(
                        interface, nw, cov, now,
                        incrementing, max_increment,
                        days, use_db,
                        reset_low, reset_high,
                        max_age
                    )
                    df_cov = df_cov.rename(columns={"y": cov})
                    cov_frames[cov] = df_cov[["ds", cov]]
                for cov, df_cov in cov_frames.items():
                    dataset = pd.merge(dataset, df_cov, on="ds", how="left")
                    dataset[cov].fillna(method="ffill", inplace=True)
                    dataset[cov].fillna(0, inplace=True)
            # Train & predict
            await nw.train(dataset, future_periods, n_lags=n_lags, country=country, covariates=covariates)
            await nw.save_prediction(
                f"{name}_prediction", now, interface,
                start=end, incrementing=incrementing,
                reset_daily=sensor.get("reset_daily", False),
                units=units, days=export_days
            )
        # Update last run and sleep
        time_str = datetime.now(timezone.utc).strftime(TIME_FORMAT_HA)
        await interface.set_state("sensor.predai_last_run", state=time_str)
        print(f"Sleeping for {config.get('update_every', 30)} minutes...")
        await asyncio.sleep(config.get('update_every', 30) * 60)

if __name__ == "__main__":
    asyncio.run(main())
