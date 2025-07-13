# -*- coding: utf-8 -*-
"""PredAI – covariate‑enabled fork (July 2025)
Fully self‑contained; place as `/config/predai.py` in your HA add-on fork.
Features:
 • Future & lagged covariates via `covariates:` block
 • YAML‑driven seasonality flags & mode
 • SQLite cache for restart resilience
 • Verbose logging for traceability
Requires NeuralProphet ≥ 0.6.3, HA Supervisor token in ENV or YAML.
"""
from __future__ import annotations
import asyncio
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple, List

import pandas as pd
import requests
import yaml
from neuralprophet import NeuralProphet, set_log_level

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

TIMEOUT = 240
FMT = "%Y-%m-%dT%H:%M:%S%z"
FMT_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"
DB_FILE = "/config/predai.db"
CONF_FILE = "/config/predai.yaml"


def ha_ts_to_dt(ts: str | None) -> datetime | None:
    for fmt in (FMT, FMT_DOT):
        try:
            return datetime.strptime(ts or "", fmt).replace(second=0, microsecond=0)
        except (ValueError, TypeError):
            continue
    return None


def floor_period(dt: datetime, period: int) -> datetime:
    mins = dt.minute - (dt.minute % period)
    return dt.replace(minute=mins, second=0, microsecond=0)

# ---------------------------------------------------------------------------
# Home‑Assistant REST helper
# ---------------------------------------------------------------------------

class HAInterface:
    def __init__(self, url: str | None, token: str | None):
        self.ha_url = url or "http://supervisor/core"
        self.ha_key = token or os.getenv("SUPERVISOR_TOKEN")
        if not self.ha_key:
            raise SystemExit("No Home‑Assistant token provided.")
        print(f"HA URL → {self.ha_url}")

    async def _call(self, ep: str, *, params=None, json_in=None, post=False) -> Any:
        url = self.ha_url + ep
        hdr = {"Authorization": f"Bearer {self.ha_key}", "Content-Type": "application/json", "Accept": "application/json"}
        fn = requests.post if post else requests.get
        try:
            resp = await asyncio.to_thread(fn, url, headers=hdr, params=params, json=json_in, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"REST {url} → {exc}")
            return None

    async def get_history(self, entity: str, now: datetime, *, days: int) -> Tuple[List[dict], datetime, datetime]:
        start = now - timedelta(days=days)
        data = await self._call(
            f"/api/history/period/{start.strftime(FMT)}",
            params={"filter_entity_id": entity, "end_time": now.strftime(FMT)}
        )
        if not data:
            return [], None, None
        arr = data[0]
        s = ha_ts_to_dt(arr[0]["last_updated"])
        e = ha_ts_to_dt(arr[-1]["last_updated"])
        print(f"History {entity}: {s} → {e} ({len(arr)} pts)")
        return arr, s, e

    async def set_state(self, entity: str, state: Any, attrs: dict | None = None) -> None:
        await self._call(
            f"/api/states/{entity}",
            json_in={"state": state, "attributes": attrs or {}},
            post=True
        )

# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

class Database:
    def __init__(self):
        self.con = sqlite3.connect(DB_FILE)
        self.cur = self.con.cursor()

    async def ensure(self, table: str) -> None:
        self.cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (timestamp TEXT PRIMARY KEY, value REAL)")
        self.con.commit()

    async def read(self, table: str) -> pd.DataFrame:
        self.cur.execute(f"SELECT * FROM {table} ORDER BY timestamp")
        rows = self.cur.fetchall()
        return pd.DataFrame(rows, columns=["ds", "y"])

    async def merge(self, table: str, df: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
        existing = set(prev["ds"].tolist()) if prev is not None and not prev.empty else set()
        added = 0
        for _, r in df.iterrows():
            ts, val = str(r["ds"]), r["y"]
            if ts in existing:
                continue
            self.cur.execute(f"INSERT INTO {table} VALUES (?, ?)", (ts, val))
            added += 1
        self.con.commit()
        if added:
            print(f"DB {table}: +{added} rows")
        return pd.concat([prev, df]).drop_duplicates("ds") if prev is not None else df

    async def prune(self, table: str, max_age: int) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age)).strftime("%Y-%m-%d %H:%M:%S%z")
        self.cur.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))
        self.con.commit()

# ---------------------------------------------------------------------------
# NeuralProphet wrapper with covariates
# ---------------------------------------------------------------------------

class NPWrapper:
    def __init__(self, period: int):
        set_log_level("ERROR")
        self.period = period
        self.model: NeuralProphet | None = None
        self.forecast: pd.DataFrame | None = None

    async def build_df(
        self,
        raw: List[dict],
        start: datetime,
        end: datetime,
        *,
        inc: bool = False,
        max_inc: float = 0,
        reset_low: float = 0,
        reset_high: float = 0
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=["ds", "y"])
        t = floor_period(start, self.period)
        idx, tot, last = 0, 0.0, None
        while t <= end and idx < len(raw):
            try:
                val = float(raw[idx]["state"])
            except ValueError:
                idx += 1
                continue
            if last is None:
                last = val
            upd = ha_ts_to_dt(raw[idx]["last_updated"])
            if not upd or upd < t:
                idx += 1
                continue
            if inc:
                if val < last < reset_high and val < reset_low:
                    tot += val
                else:
                    if max_inc and abs(val - last) > max_inc:
                        val = last
                    tot += max(val - last, 0)
            last = val
            df.loc[len(df)] = {"ds": t, "y": tot if inc else val}
            t += timedelta(minutes=self.period)
        return df

    async def train(
        self,
        base: pd.DataFrame,
        periods: int,
        *,
        n_lags: int = 0,
        country: str | None = None,
        seasonality_mode: str = "additive",
        daily: bool = True,
        weekly: bool = True,
        yearly: bool = False,
        covars: Dict[str, Tuple[pd.DataFrame, bool]] | None = None
    ) -> None:
        self.model = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=1,
            seasonality_mode=seasonality_mode,
            daily_seasonality=daily,
            weekly_seasonality=weekly,
            yearly_seasonality=yearly
        )
        if country:
            self.model.add_country_holidays(country)
        if covars:
            print(f"Covariates: {list(covars.keys())}")
            for name, (df_c, known) in covars.items():
                reg_fn = self.model.add_future_regressor if known else self.model.add_lagged_regressor
                reg_fn(name)
                df2 = df_c.rename(columns={"y": name})
                base = base.merge(df2, on="ds", how="left")
                na0 = base[name].isna().sum()
                base[name] = base[name].ffill()
                print(f" • {name}: NaN {na0} → {base[name].isna().sum()}")
        self.model.fit(base, freq=f"{self.period}min", progress=None)
        fut = self.model.make_future_dataframe(base, periods=periods, n_historic_predictions=True)
        if covars:
            for name in covars.keys():
                if name in fut.columns:
                    fut[name] = fut[name].ffill()
        self.forecast = self.model.predict(fut)

    async def publish(
        self,
        iface: HAInterface,
        entity: str,
        now: datetime,
        *,
        inc: bool = False,
        reset_daily: bool = False,
        units: str = "",
        history_days: int = 7
    ) -> None:
        if self.forecast is None:
            return
        res, src, tot, tot_src = {}, {}, 0.0, 0.0
        for _, r in self.forecast.iterrows():
            ts: datetime = r["ds"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            diff = ts - now
            if diff.days < -history_days:
                continue
            label = (now + diff).strftime(FMT)
            yh = float(r["yhat1"])
            y_true = r.get("y", math.nan)
            if ts <= now and reset_daily and ts.hour == ts.minute == 0:
                tot, tot_src = 0.0, 0.0
            if inc:
                tot += yh
                res[label] = round(tot, 2)
                if not math.isnan(y_true):
                    tot_src += y_true
                    src[label] = round(tot_src, 2)
            else:
                res[label] = round(yh, 2)
                if not math.isnan(y_true):
                    src[label] = round(y_true, 2)
        final = tot if inc else yh
        attrs = {"last_updated": str(now), "unit_of_measurement": units, "state_class": "measurement", "results": res, "source": src}
        await iface.set_state(f"{entity}_prediction", final, attrs)
        print(f"Publish → {entity}_prediction ({len(res)} points)")

# ---------------------------------------------------------------------------
# Helpers: subtract & sensor build
# ---------------------------------------------------------------------------

async def subtract_set(base: pd.DataFrame, sub: pd.DataFrame, *, inc: bool = False) -> pd.DataFrame:
    merged = base.merge(sub, on="ds", how="left", suffixes=("", "_sub"))
    merged["y_sub"].fillna(0, inplace=True)
    merged["y"] = merged.apply(lambda row: max(row["y"] - row["y_sub"], 0) if inc else row["y"] - row["y_sub"], axis=1)
    return merged[["ds", "y"]]

async def build_sensor_history(
    iface: HAInterface,
    npw: NPWrapper,
    cfg: Dict[str, Any],
    now: datetime,
    use_db: bool,
) -> Tuple[pd.DataFrame, datetime, datetime]:
    raw, start, end = await iface.get_history(cfg["name"], now, days=cfg.get("days", 7))
    df = await npw.build_df(
        raw, start, end,
        inc=cfg.get("incrementing", False),
        max_inc=cfg.get("max_increment", 0),
        reset_low=cfg.get("reset_low", 0),
        reset_high=cfg.get("reset_high", 0)
    )
    if use_db:
        table = cfg["name"].replace(".", "_")
        db = Database()
        await db.ensure(table)
        prev = await db.read(table)
        df = await db.merge(table, df, prev)
        await db.prune(table, cfg.get("max_age", 365))
    return df, start, end

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main():
    config = yaml.safe_load(open(CONF_FILE)) or {}
    iface = HAInterface(config.get("ha_url"), config.get("ha_key"))
    while True:
        print("Configuration loaded")
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        for cfg in config.get("sensors", []):
            name = cfg.get("name")
            if not name:
                continue
            print(f"\n=== {name} @ {now} ===")
            npw = NPWrapper(cfg.get("interval", 30))
            df_base, start, end = await build_sensor_history(iface, npw, cfg, now, use_db=cfg.get("database", True))
            covars: Dict[str, Tuple[pd.DataFrame, bool]] = {}
            for cov in cfg.get("covariates", []):
                df_cov, _, _ = await build_sensor_history(iface, npw, cov, now, use_db=False)
                covars[cov["name"]] = (df_cov, cov.get("known_in_advance", False))
            # subtraction
            if cfg.get("subtract"):
                subs = cfg["subtract"] if isinstance(cfg["subtract"], list) else [cfg["subtract"]]
                for sub in subs:
                    df_sub, _, _ = await build_sensor_history(iface, npw, {"name": sub, **cfg}, now, use_db=False)
                    df_base = await subtract_set(df_base, df_sub, inc=cfg.get("incrementing", False))
            # train & predict
            await npw.train(
                df_base,
                cfg.get("future_periods", 96),
                n_lags=cfg.get("n_lags", 0),
                country=cfg.get("country"),
                seasonality_mode=cfg.get("seasonality_mode", "additive"),
                daily=cfg.get("daily_seasonality", True),
                weekly=cfg.get("weekly_seasonality", True),
                yearly=cfg.get("yearly_seasonality", False),
                covars=covars
            )
            await npw.publish(
                iface,
                name,
                now,
                inc=cfg.get("incrementing", False),
                reset_daily=cfg.get("reset_daily", False),
                units=cfg.get("units", ""),
                history_days=cfg.get("export_days", cfg.get("days", 7))
            )
        # mark run
        await iface.set_state("sensor.predai_last_run", str(now), {"unit_of_measurement": "time"})
        # sleep
        await asyncio.sleep(config.get("update_every", 30) * 60)

if __name__ == "__main__":
    asyncio.run(main())
