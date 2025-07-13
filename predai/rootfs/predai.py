# -*- coding: utf-8 -*-
"""PredAI – covariate‑enabled fork (July 2025)
===========================================
Based on upstream `springfall2008/predai` **commit 6eb34d**.
Adds:
* **Covariate / regressor support** (`covariates:` block in YAML)
* Future‑known vs lagged regressors (`known_in_advance: true`)
* Seasonality flags & mode surfaced from YAML
* Verbose logging around covariate handling
* Loose ends fixed (holiday token, interval flooring, safe tz)

Replace the original `predai.py` in your fork with this file or point
`run.sh` at it.  Works with NeuralProphet ≥ 0.6.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import asyncio
import math
import os
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
from neuralprophet import NeuralProphet, set_log_level

# ---------------------------------------------------------------------------
# Globals & helpers
# ---------------------------------------------------------------------------

TIMEOUT = 240
FMT_HA = "%Y-%m-%dT%H:%M:%S%z"
FMT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"


def ts_to_dt(ts: str | None):
    for fmt in (FMT_HA, FMT_HA_DOT):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(second=0, microsecond=0)
        except (TypeError, ValueError):
            pass
    return None


def floor_to_period(dt: datetime, period: int) -> datetime:
    """Round *down* to nearest *period*‑minute boundary."""
    delta = dt.minute % period
    return dt - timedelta(minutes=delta, seconds=dt.second, microseconds=dt.microsecond)

# ---------------------------------------------------------------------------
# Home‑Assistant REST interface
# ---------------------------------------------------------------------------


class HAInterface:
    def __init__(self, url: str | None, token: str | None):
        self.ha_url = url or "http://supervisor/core"
        self.ha_key = token or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            raise SystemExit("No Home‑Assistant auth token found!")
        print(f"HA interface → {self.ha_url}")

    # low‑level REST helper
    async def _call(self, endpoint: str, *, params=None, json_in=None, post=False):
        url = self.ha_url + endpoint
        headers = {"Authorization": f"Bearer {self.ha_key}", "Content-Type": "application/json", "Accept": "application/json"}
        fn = requests.post if post else requests.get
        try:
            resp = await asyncio.to_thread(fn, url, headers=headers, params=params, json=json_in, timeout=TIMEOUT)
            return resp.json()
        except (requests.exceptions.JSONDecodeError, requests.Timeout, requests.exceptions.ReadTimeout):
            print(f"REST error → {url}")
            return None

    async def get_history(self, entity: str, now: datetime, *, days: int):
        start = now - timedelta(days=days)
        ep = f"/api/history/period/{start.strftime(FMT_HA)}"
        res = await self._call(ep, params={"filter_entity_id": entity, "end_time": now.strftime(FMT_HA)})
        if not res:
            return [], None, None
        res = res[0]
        s, e = ts_to_dt(res[0]["last_updated"]), ts_to_dt(res[-1]["last_updated"])
        print(f"History {entity}: {s} → {e} ({len(res)} rows)")
        return res, s, e

    async def set_state(self, entity: str, state, attrs=None):
        await self._call(f"/api/states/{entity}", json_in={"state": state, "attributes": attrs or {}}, post=True)

# ---------------------------------------------------------------------------
# SQLite cache (unchanged except for minor prints)
# ---------------------------------------------------------------------------


class Database:
    def __init__(self):
        self.con = sqlite3.connect("/config/predai.db")
        self.cur = self.con.cursor()

    async def ensure_table(self, name: str):
        self.cur.execute(f"CREATE TABLE IF NOT EXISTS {name} (timestamp TEXT PRIMARY KEY, value REAL)")
        self.con.commit()

    async def get(self, name: str) -> pd.DataFrame:
        self.cur.execute(f"SELECT * FROM {name} ORDER BY timestamp")
        rows = self.cur.fetchall()
        return pd.DataFrame(rows, columns=["ds", "y"])

    async def upsert(self, name: str, data: pd.DataFrame, prev: pd.DataFrame | None):
        existing = set(prev["ds"].tolist()) if prev is not None and not prev.empty else set()
        added = 0
        for _, row in data.iterrows():
            ts, val = str(row["ds"]), row["y"]
            if ts in existing:
                continue
            self.cur.execute(f"INSERT INTO {name} VALUES (?, ?)", (ts, val))
            added += 1
        self.con.commit()
        if added:
            print(f"DB • {name}: added {added}")
        return pd.concat([prev, data]).drop_duplicates("ds") if prev is not None else data

    async def cleanup(self, name: str, max_age: int):
        cut = (datetime.now(timezone.utc) - timedelta(days=max_age)).strftime("%Y-%m-%d %H:%M:%S%z")
        self.cur.execute(f"DELETE FROM {name} WHERE timestamp < ?", (cut,))
        self.con.commit()

# ---------------------------------------------------------------------------
# NeuralProphet wrapper with covariates
# ---------------------------------------------------------------------------


class ProphetNP:
    def __init__(self, period_min: int):
        set_log_level("ERROR")
        self.period = period_min
        self.model: NeuralProphet | None = None
        self.forecast: pd.DataFrame | None = None

    async def to_dataframe(self, raw: list, start: datetime, end: datetime, *, incrementing=False, max_increment=0, reset_low=0, reset_high=0):
        df = pd.DataFrame(columns=["ds", "y"])
        t = floor_to_period(start, self.period)
        idx, total, last = 0, 0.0, None
        while t <= end and idx < len(raw):
            try:
                val = float(raw[idx]["state"])
            except ValueError:
                idx += 1; continue
            if last is None:
                last = val
            upd = ts_to_dt(raw[idx]["last_updated"])
            if not upd or upd < t:
                idx += 1; continue
            if incrementing:
                if val < last < reset_high and val < reset_low:
                    total += val  # rollover
                else:
                    if max_increment and abs(val - last) > max_increment:
                        val = last
                    total += max(val - last, 0)
            last = val
            df.loc[len(df)] = {"ds": t, "y": total if incrementing else val}
            t += timedelta(minutes=self.period)
        return df

    async def train(
        self,
        base: pd.DataFrame,
        periods: int,
        *,
        n_lags=0,
        country=None,
        seasonality_mode="additive",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        covariates: Dict[str, Tuple[pd.DataFrame, bool]] | None = None,
    ):
        self.model = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=1,
            seasonality_mode=seasonality_mode,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
        )
        if country:
            self.model.add_country_holidays(country)

        if covariates:
            print(f"Covariates ({len(covariates)}): {list(covariates.keys())}")
            for name, (df, known_adv) in covariates.items():
                if known_adv:
                    self.model.add_future_regressor(name)
                else:
                    self.model.add_lagged_regressor(name)
                df2 = df.rename(columns={"y": name})
                base = base.merge(df2, on="ds", how="left")
                na_pre = base[name].isna().sum()
                base[name] = base[name].ffill()
                print(f" • {name}: {na_pre} → {base[name].isna().sum()} NaNs after ffill")

        self.model.fit(base, freq=f"{self.period}min", progress=None)
        fut = self.model.make_future_dataframe(base, periods=periods, n_historic_predictions=True)
        if covariates:
            for name in covariates.keys():
                if name in fut.columns:
                    fut[name] = fut[name].ffill()
        self.forecast = self.model.predict(fut)

    async def publish(self, iface: HAInterface, entity: str, now: datetime, *, incrementing=False, reset_daily=False, units="", history_days=7):
        if self.forecast is None:
            return
        res, src, tot, tot_src = {}, {}, 0.0, 0.0
        for _, row in self.forecast.iterrows():
            ts: datetime = row["ds"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            diff = ts - now
            if diff.days < -history_days:
                continue
            label = (now + diff).strftime(FMT_HA)
            yhat, y_true = float(row["yhat1"]), row.get("y", math.nan)
            if ts <= now and reset_daily and ts.hour == ts.minute == 0:
                tot = tot_src = 0.0
            if incrementing:
                tot += yhat
                res[label] = round(tot, 2)
                if not math.isnan(y_true):
                    tot_src += y_true
                    src[label] = round(tot_src, 2)
            else:
                res[label] = round(yhat, 2)
                if not math.isnan(y_true):
                    src[label] = round(y_true, 2)
        final = tot if incrementing else yhat
        attrs = {"last_updated": str(now), "unit_of_measurement": units, "state_class": "measurement", "results": res, "source": src}
        await iface.set_state(f"{entity}_prediction", state=round(final, 2), attrs=attrs)
        print(f"Saved {len(res)} forecast pts → {entity}_prediction")

# ---------------------------------------------------------------------------
# Data acquisition helpers
# ---------------------------------------------------------------------------

async def build_series(iface: HAInterface, np_wrap: ProphetNP, cfg: dict, now: datetime, *, use_db: bool, max_age: int):
    inc = cfg.get("incrementing", False)
    raw, s, e = await iface.get_history(cfg["name"], now, days=cfg.get("days", 7))
    df = await np_wrap.to_dataframe(raw, s, e, incrementing=inc, max_increment=cfg.get("max_increment", 0), reset_low=cfg.get("reset_low", 0), reset_high=cfg.get("reset_high", 0))
    if use_db:
        tbl = cfg["name"].replace(".", "_")
        db = Database()
        await db.ensure_table(tbl)
        prev = await db.get(tbl)
        df = await db.upsert(tbl, df, prev)
        await db.cleanup(tbl, max_age)
    return df, s, e

async def subtract_df(base: pd.DataFrame, sub: pd.DataFrame, *, inc: bool):
    sub = sub.set_index("ds")
    out = []
    for _, row in base.iterrows():
        delta = sub.at[row["ds"], "y"] if row["ds"] in sub.index else 0
        out.append({"ds": row["ds"], "y": max(row["y"] - delta, 0) if inc else row["y"] - delta})
    return pd.DataFrame(out)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main():
    cfg_path = "/config/predai.yaml"
    base_cfg = yaml.safe_load(open(cfg_path))
    iface = HAInterface(base
