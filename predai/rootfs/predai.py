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

"""
PredAI – extended to support *covariate* sensors (a.k.a. regressors)
===================================================================

Key additions
-------------
1. **Configuration**::

       sensors:
         - name: sensor.energy_total
           covariates:
             - sensor.outdoor_temperature  # will be used as a lagged regressor
             - sensor.humidity
           cov_incrementing: false          # optional override per‑covariate
           cov_n_lags: 24                   # number of historic lags to use

   If *cov_n_lags* is omitted we default to the parent sensor's *n_lags*.

2. **Data ingestion** – we fetch history for every covariate, resample it on the
   same time‑grid as the target and merge into a single DataFrame.

3. **Model** – for each covariate we call ``add_lagged_regressor`` so only
   historic values are required at prediction time (no need for future values).

4. **Utility helpers**: ``merge_covariates`` and ``align_series``.
"""

###############################################################################
# Constants & helpers                                                          #
###############################################################################

TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"


def timestr_to_datetime(timestamp: str) -> datetime | None:
    """Convert Home‑Assistant timestamp string to *minute‑aware* ``datetime``."""
    for fmt in (TIME_FORMAT_HA, TIME_FORMAT_HA_DOT):
        try:
            dt = datetime.strptime(timestamp, fmt)
            return dt.replace(second=0, microsecond=0)
        except ValueError:
            continue
    return None

###############################################################################
# Home‑Assistant interface                                                    #
###############################################################################

class HAInterface:
    """Thin async wrapper around HA REST API."""

    def __init__(self, ha_url: str | None, ha_key: str | None) -> None:
        self.ha_url = ha_url or "http://supervisor/core"
        self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            raise RuntimeError("No Home Assistant key found, exiting")
        print(f"HA Interface started key {self.ha_key[:5]}… url {self.ha_url}")

    # unchanged: get_events, get_state, set_state, api_call, get_history ……
    # (Paste the bodies from the original script without modification.)

    async def get_history(self, sensor: str, now: datetime, days: int = 7):
        """Fetch history for *sensor* and return raw list plus (start, end)"""
        start = now - timedelta(days=days)
        print(
            f"Getting history for sensor {sensor} start {start:%Y‑%m‑%dT%H:%M%z}"
            f" end {now:%Y‑%m‑%dT%H:%M%z}"
        )
        res = await self.api_call(
            f"/api/history/period/{start.strftime(TIME_FORMAT_HA)}",
            {"filter_entity_id": sensor, "end_time": now.strftime(TIME_FORMAT_HA)},
        )
        if res:
            res = res[0]
            start = timestr_to_datetime(res[0]["last_updated"])
            end = timestr_to_datetime(res[-1]["last_updated"])
        else:
            end = start  # empty
        return res or [], start, end

###############################################################################
# Prophet wrapper                                                             #
###############################################################################

def align_series(base_times: pd.Series, other: pd.DataFrame, name: str) -> pd.Series:
    """Map *other* sensor dataframe on to *base_times* index, using forward‑fill."""
    ser = other.set_index("ds")["y"]
    aligned = base_times.map(ser)
    return aligned.ffill()


def merge_covariates(
    base_df: pd.DataFrame,
    cov_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return *base_df* with additional columns for every covariate dataset."""
    out = base_df.copy()
    for name, cov in cov_dfs.items():
        out[name] = align_series(out["ds"], cov, name)
    return out


class Prophet:
    def __init__(self, period: int = 30):
        set_log_level("ERROR")
        self.period = period
        self.model: NeuralProphet | None = None

    # ---------------------------------------------------------------------
    # unchanged: process_dataset … (identical to original)
    # ---------------------------------------------------------------------

    async def process_dataset(
        self,
        sensor_name: str,
        new_data: list[dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
        *,
        incrementing: bool = False,
        max_increment: float = 0,
        reset_low: float = 0.0,
        reset_high: float = 0.0,
    ) -> Tuple[pd.DataFrame, float]:
        """Build a (ds,y) dataframe on a fixed *period* grid (unchanged)."""
        # ----- paste original body here UNCHANGED -----

    # ---------------------------------------------------------------------
    # New: training with optional covariates
    # ---------------------------------------------------------------------
    async def train(
        self,
        dataset: pd.DataFrame,
        future_periods: int,
        *,
        n_lags: int = 0,
        country: str | None = None,
        covariate_cols: List[str] | None = None,
        cov_n_lags: int | None = None,
    ) -> None:
        """Train model, optionally adding *lagged* covariates."""
        self.model = NeuralProphet(
            n_lags=n_lags,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
        )
        if country:
            self.model.add_country_holidays(country)

        if covariate_cols:
            for cov in covariate_cols:
                self.model.add_lagged_regressor(cov, n_lags=cov_n_lags or n_lags or 24)

        self.metrics = self.model.fit(dataset, freq=f"{self.period}min", progress=None)
        self.df_future = self.model.make_future_dataframe(
            dataset, n_historic_predictions=True, periods=future_periods
        )
        self.forecast = self.model.predict(self.df_future)

    # ---------------------------------------------------------------------
    # unchanged: save_prediction …
    # ---------------------------------------------------------------------

###############################################################################
# Existing helper functions: subtract_set, Database… remain *unchanged*       #
###############################################################################

###############################################################################
# Main loop                                                                   #
###############################################################################

async def get_history(
    interface: "HAInterface",
    nw: "Prophet",
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
    """Wrapper kept identical – uses nw.process_dataset internally."""
    # --- paste original body UNCHANGED ---


async def main() -> None:
    config = yaml.safe_load(open("/config/predai.yaml"))
    interface = HAInterface(config.get("ha_url"), config.get("ha_key"))

    while True:
        config = yaml.safe_load(open("/config/predai.yaml"))
        if not config:
            print("WARN: predai.yaml is missing, no work to do")
        else:
            update_every = config.get("update_every", 30)
            sensors = config.get("sensors", [])

            for sensor in sensors:
                sensor_name = sensor.get("name")
                if not sensor_name:
                    continue

                # Original options ------------------------------------------------
                subtract_names = sensor.get("subtract")
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

                # New options -----------------------------------------------------
                covariates: list[str] = sensor.get("covariates", [])
                cov_incrementing: bool = sensor.get("cov_incrementing", False)
                cov_n_lags: int | None = sensor.get("cov_n_lags")

                nw = Prophet(interval)
                now = datetime.now(timezone.utc).astimezone().replace(
                    second=0, microsecond=0, minute=0
                )

                print(
                    f"[{now}] Processing {sensor_name} (covariates={covariates})"
                )

                # ----------------------------------------------------------------
                # Target dataset
                # ----------------------------------------------------------------
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

                # ----------------------------------------------------------------
                # Subtract logic (unchanged)
                # ----------------------------------------------------------------
                subtract_data_list = []
                if subtract_names:
                    if isinstance(subtract_names, str):
                        subtract_names = [subtract_names]
                    for sub_name in subtract_names:
                        sub_data, *_ = await get_history(
                            interface,
                            nw,
                            sub_name,
                            now,
                            incrementing=incrementing,
                            max_increment=max_increment,
                            days=days,
                            use_db=use_db,
                            reset_low=reset_low,
                            reset_high=reset_high,
                            max_age=max_age,
                        )
                        subtract_data_list.append(sub_data)

                for sub_data in subtract_data_list:
                    dataset = await subtract_set(dataset, sub_data, now, incrementing=incrementing)

                # ----------------------------------------------------------------
                # Fetch covariate datasets
                # ----------------------------------------------------------------
                cov_dfs: Dict[str, pd.DataFrame] = {}
                for cov_name in covariates:
                    cov_ds, *_ = await get_history(
                        interface,
                        nw,
                        cov_name,
                        now,
                        incrementing=cov_incrementing,
                        max_increment=0,
                        days=days,
                        use_db=use_db,
                        reset_low=0.0,
                        reset_high=0.0,
                        max_age=max_age,
                    )
                    cov_dfs[cov_name] = cov_ds

                if cov_dfs:
                    dataset = merge_covariates(dataset, cov_dfs)

                # ----------------------------------------------------------------
                # Train & predict
                # ----------------------------------------------------------------
                await nw.train(
                    dataset,
                    future_periods,
                    n_lags=n_lags,
                    country=country,
                    covariate_cols=list(cov_dfs.keys()) or None,
                    cov_n_lags=cov_n_lags,
                )

                await nw.save_prediction(
                    f"{sensor_name}_prediction",
                    now,
                    interface,
                    start=end,
                    incrementing=incrementing,
                    reset_daily=reset_daily,
                    units=units,
                    days=export_days,
                )

        # --------------------------------------------------------------------
        # book‑keeping & scheduler
        # --------------------------------------------------------------------
        time_now = datetime.now(timezone.utc).astimezone()
        await interface.set_state(
            "sensor.predai_last_run", state=str(time_now), attributes={"unit_of_measurement": "time"}
        )
        print(f"Waiting {update_every} minutes… (ctrl‑c to stop)")
        for _ in range(update_every):
            last_run = await interface.get_state("sensor.predai_last_run")
            if last_run is None:
                print("Restarting PredAI – last_run disappeared")
                break
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
