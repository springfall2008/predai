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


def timestr_to_datetime(timestamp):
    """
    Convert a Home Assistant timestamp string to a datetime object.
    """
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


class HAInterface():
    def __init__(self, ha_url, ha_key):
        # Initialise Home Assistant interface
        self.ha_url = ha_url or "http://supervisor/core"
        self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            print("No Home Assistant key found, exiting")
            exit(1)
        print(f"HA Interface started key {self.ha_key} url {self.ha_url}")

    async def get_events(self):
        return await self.api_call("/api/events")

    async def get_history(self, sensor, now, days=7):
        """
        Get the history for a sensor from Home Assistant.
        Returns list of events, and start/end datetimes.
        """
        start = now - timedelta(days=days)
        end = now
        print(f"Getting history for sensor {sensor} from {start.strftime(TIME_FORMAT_HA)} to {end.strftime(TIME_FORMAT_HA)}")
        res = await self.api_call(
            f"/api/history/period/{start.strftime(TIME_FORMAT_HA)}",
            {"filter_entity_id": sensor, "end_time": end.strftime(TIME_FORMAT_HA)}
        )
        if res:
            res = res[0]
            start = timestr_to_datetime(res[0]["last_updated"])
            end = timestr_to_datetime(res[-1]["last_updated"])
        print(f"History for sensor {sensor} starts at {start} ends at {end}")
        return res, start, end

    async def get_state(self, entity_id=None, default=None, attribute=None):
        item = await self.api_call(f"/api/states/{entity_id}")
        if not item:
            return default
        if attribute:
            return item.get("attributes", {}).get(attribute, default)
        return item.get("state", default)

    async def set_state(self, entity_id, state, attributes=None):
        data = {"state": state}
        if attributes:
            data["attributes"] = attributes
        await self.api_call(f"/api/states/{entity_id}", data, post=True)

    async def api_call(self, endpoint, datain=None, post=False):
        url = self.ha_url + endpoint
        headers = {
            "Authorization": f"Bearer {self.ha_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            if post:
                response = await asyncio.to_thread(requests.post, url, headers=headers, json=datain, timeout=TIMEOUT)
            else:
                response = await asyncio.to_thread(requests.get, url, headers=headers, params=datain, timeout=TIMEOUT)
            return response.json()
        except Exception as e:
            print(f"API call error at {url}: {e}")
            return None


class Prophet:
    def __init__(self, period=30):
        set_log_level("ERROR")
        self.period = period

    async def process_dataset(self, sensor_name, new_data, start_time, end_time, 
                              incrementing=False, max_increment=0,
                              reset_low=0.0, reset_high=0.0):
        """
        Build a regular time series (ds, y) from raw HA history.
        """
        dataset = pd.DataFrame(columns=["ds", "y"])
        timenow = start_time.replace(second=0, microsecond=0, minute=0)
        data_index = 0
        total = 0
        last_value = None

        while timenow <= end_time and data_index < len(new_data):
            entry = new_data[data_index]
            try:
                value = float(entry["state"])
                if last_value is None:
                    last_value = value
            except ValueError:
                value = last_value or 0.0

            t = timestr_to_datetime(entry["last_updated"])
            if not t or t < timenow:
                data_index += 1
                continue

            if incrementing:
                if value < last_value and value < reset_low and last_value > reset_high:
                    total += value
                else:
                    if max_increment and abs(value - last_value) > max_increment:
                        value = last_value
                    total = max(total + value - last_value, 0)
                real_value = total
                total = 0
            else:
                real_value = value

            dataset.loc[len(dataset)] = {"ds": timenow, "y": real_value}
            last_value = value
            timenow += timedelta(minutes=self.period)
            data_index += 1

        return dataset, last_value

    async def train(self, dataset, future_periods, n_lags=0, country=None, covariates=None):
        """
        Train model, optionally adding holiday and covariate regressors.
        """
        self.model = NeuralProphet(
            n_lags=n_lags,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        # Add holidays if requested
        if country:
            print(f"Adding country holidays for {country}")
            self.model.add_country_holidays(country)
        # Add external regressors
        if covariates:
            for cov in covariates:
                print(f"Adding covariate regressor: {cov}")
                self.model.add_regressor(cov)

        # Fit model
        self.metrics = self.model.fit(dataset, freq=f"{self.period}min", progress=None)
        # Create future dataframe
        self.df_future = self.model.make_future_dataframe(
            dataset, n_historic_predictions=True, periods=future_periods
        )
        # Fill future covariate values with last known
        if covariates:
            for cov in covariates:
                last_val = dataset[cov].iloc[-1]
                self.df_future[cov] = self.df_future[cov].fillna(last_val)

        # Predict
        self.forecast = self.model.predict(self.df_future)

    async def save_prediction(self, entity, now, interface, start, incrementing=False, reset_daily=False, units="", days=7):
        """Save forecast back to Home Assistant."""
        pred = self.forecast
        results, source = {}, {}
        total = total_org = 0
        for _, row in pred.iterrows():
            pt = row["ds"].tz_localize(timezone.utc)
            diff = pt - now
            if diff.days < -days:
                continue
            ts = (now + diff).strftime(TIME_FORMAT_HA)
            val = row["yhat1"]
            orig = row.get("y", None)
            if incrementing:
                total += val
                if not math.isnan(orig): total_org += orig
                results[ts] = round(total, 2)
                source[ts] = round(total_org, 2)
            else:
                results[ts] = round(val, 2)
                if not math.isnan(orig): source[ts] = round(orig, 2)
        final = total if incrementing else val
        attrs = {
            "last_updated": str(now),
            "unit_of_measurement": units,
            "state_class": "measurement",
            "results": results,
            "source": source
        }
        print(f"Saving prediction to {entity}")
        await interface.set_state(entity, state=round(final, 2), attributes=attrs)

# Database and utility classes unchanged...
# ... [retain Database, subtract_set, print_dataset as before] ...

async def get_history(interface, nw, sensor_name, now, incrementing, max_increment, days, use_db, reset_low, reset_high, max_age):
    # identical to original get_history, returns dataset, start, end
    # ...
    pass  # retain original implementation

async def main():
    config = yaml.safe_load(open("/config/predai.yaml"))
    interface = HAInterface(config.get("ha_url"), config.get("ha_key"))
    while True:
        config = yaml.safe_load(open("/config/predai.yaml"))
        sensors = config.get("sensors", [])
        for sensor in sensors:
            # Core sensor config
            sensor_name = sensor.get("name")
            days = sensor.get("days", 7)
            update_every = sensor.get("update_every", 30)
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

            # New: covariate sensors
            covariates = sensor.get("covariates", [])  # list of entity_ids

            now = datetime.now(timezone.utc).replace(second=0, microsecond=0, minute=0)
            nw = Prophet(sensor.get("interval", 30))

            # Fetch main sensor history
            dataset, start, end = await get_history(
                interface, nw, sensor_name, now,
                incrementing, max_increment, days,
                use_db, reset_low, reset_high, max_age
            )

            # Fetch subtract sensors as before...
            # ... (unchanged) ...

            # New: fetch and merge covariates
            cov_data = {}
            for cov in covariates:
                df_cov, _, _ = await get_history(
                    interface, nw, cov, now,
                    incrementing, max_increment, days,
                    use_db, reset_low, reset_high, max_age
                )
                cov_data[cov] = df_cov.rename(columns={"y": cov})
            # Merge into main dataset
            if cov_data:
                for cov, df_cov in cov_data.items():
                    dataset = pd.merge(dataset, df_cov[["ds", cov]], on="ds", how="left")
                    dataset[cov].fillna(method="ffill", inplace=True)
                    dataset[cov].fillna(0, inplace=True)

            # Train and predict including covariates
            await nw.train(
                dataset, future_periods,
                n_lags=n_lags,
                country=country,
                covariates=covariates
            )

            # Save prediction
            await nw.save_prediction(
                f"{sensor_name}_prediction",
                now, interface, start=end,
                incrementing=incrementing,
                reset_daily=sensor.get("reset_daily", False),
                units=units,
                days=export_days
            )

        # Update last run and sleep as before...
        # ... (unchanged) ...

if __name__ == "__main__":
    asyncio.run(main())
