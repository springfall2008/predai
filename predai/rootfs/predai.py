print("Test started")

from typing import Any
import pandas as pd
from datetime import datetime, timedelta, timezone
from neuralprophet import NeuralProphet, set_log_level
import os
import aiohttp
import requests
import asyncio
import json
import ssl

#  - torch
#  - neuralprophet==0.8.0

TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"

class HAInterface():
    def __init__(self):
        self.ha_key = os.environ.get("SUPERVISOR_TOKEN")
        self.ha_url = "http://supervisor/core"
        print("HA Interface started key {} url {}".format(self.ha_key, self.ha_url))

    async def get_history(self, sensor):
        """
        Get the history for a sensor from Home Assistant.

        :param sensor: The sensor to get the history for.
        :return: The history for the sensor.
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=7)
        end = now
        print("Getting history for sensor {}".format(sensor))
        res = await self.api_call("/api/history/period/{}".format(start.strftime(TIME_FORMAT_HA)), {"filter_entity_id": sensor, "end_time": end.strftime(TIME_FORMAT_HA)})
        if res:
            res = res[0]
        return res, start, end

    async def api_call(self, endpoint, datain=None, post=False):
        """
        Make an API call to Home Assistant.

        :param endpoint: The API endpoint to call.
        :param datain: The data to send in the body of the request.
        :param post: True if this is a POST request, False for GET.
        :return: The response from the API.
        """
        url = self.ha_url + endpoint
        print("Making API call to {}".format(url))
        headers = {
            "Authorization": "Bearer " + self.ha_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if post:
            if datain:
                response = await asyncio.to_thread(requests.post, url, headers=headers, json=datain, timeout=TIMEOUT)
            else:
                response = await asyncio.to_thread(requests.post, url, headers=headers, timeout=TIMEOUT)
        else:
            if datain:
                response = await asyncio.to_thread(requests.get, url, headers=headers, params=datain, timeout=TIMEOUT)
            else:
                response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=TIMEOUT)
        print("Response {}".format(response))
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Failed to decode response from {}".format(url))
            data = None
        except (requests.Timeout, requests.exceptions.ReadTimeout):
            print("Timeout from {}".format(url))
            data = None
        return data

class Prophet:
    def __init__(self):
        set_log_level("ERROR")

    def store_data(self, new_data, start_time, end_time, incrementing=False):
        """
        Store the data in the dataset for training.
        """
        self.dataset = pd.DataFrame(columns=["ds", "y"])
        
        timenow = start_time
        data_index = 0
        value = 0
        last_value = 0
        if incrementing:
            last_value = float(new_data[0]["state"])

        while timenow < end_time:
            try:
                value = float(new_data[data_index]["state"])
            except ValueError:
                pass

            last_updated = new_data[data_index]["last_updated"]
            try:
                start_time = datetime.strptime(last_updated, TIME_FORMAT_HA)
            except ValueError:
                try:
                    start_time = datetime.strptime(last_updated, TIME_FORMAT_HA_DOT)
                except ValueError:
                    start_time = None
        
            if not start_time or start_time < timenow:
                data_index += 1
                continue

            real_value = value
            if incrementing:
                real_value = value - last_value
            self.dataset.loc[len(self.dataset)] = {"ds": timenow, "y": real_value}
            last_value = value
            timenow = timenow + timedelta(minutes=30)

        print(self.dataset.head())
    
    def train(self):
        """
        Train the model on the dataset.
        """
        self.model = NeuralProphet()
        # Fit the model on the dataset (this might take a bit)
        self.metrics = self.model.fit(self.dataset)
        # Create a new dataframe reaching 96 into the future for our forecast, n_historic_predictions also shows historic data
        self.df_future = self.model.make_future_dataframe(self.dataset, n_historic_predictions=True, periods=96)
        self.forecast = self.model.predict(self.df_future)
        print(self.forecast.head())
    
    def save_prediction(self, entity, incrementing=False):
        """
        Save the prediction to Home Assistant.
        """
        pred = self.forecast
        total = 0
        timeseries = {}
        for index, row in pred.iterrows():
            time = row["ds"]
            value = row["y"]
            total += value
            if incrementing:
                timeseries[time] = total
            else:
                timeseries[time] = value
        data = {"state": 0, "attributes": {"unit_of_measurement": "kWh", "results" : timeseries}}
        print("Saving prediction to {}".format(entity))
        self.interface.api_call("/api/states/{}".format(entity), data, post=True)

async def main():
    interface = HAInterface()
    while True:
        dataset, start, end = await interface.get_history("sensor.givtcp_sa2243g277_load_energy_total_kwh")
        if dataset:
            nw = Prophet()
            nw.store_data(dataset, start, end, incrementing=True)
            nw.train()
            nw.save_prediction("sensor.givtcp_sa2243g277_load_energy_total_kwh_prediction", incrementing=True)

        print("Waiting")
        await asyncio.sleep(60 * 60)

asyncio.run(main())
