print("Test started")

from typing import Any
import pandas as pd
from datetime import datetime, timedelta
from neuralprophet import NeuralProphet, set_log_level

#  - torch
#  - neuralprophet==0.8.0

TIME_FORMAT_OCTOPUS = "%Y-%m-%dT%H:%M:%S%z"

class Prophet:
    def __init__(self):
        set_log_level("ERROR")

    def store_data(self, dataset):
        self.dataset = pd.DataFrame(columns=["ds", "y"])
        for item in dataset:
            start_time = datetime.strptime(item["Start"], TIME_FORMAT_OCTOPUS)
            value = item["Consumption (kWh)"]
            self.dataset = self.dataset.append({"ds": start_time, "y": value}, ignore_index=True)
    
    def train(self):
        self.model = NeuralProphet()
        # Fit the model on the dataset (this might take a bit)
        self.metrics = self.model.fit(self.dataset)
        # Create a new dataframe reaching 96 into the future for our forecast, n_historic_predictions also shows historic data
        self.df_future = self.model.make_future_dataframe(self.dataset, n_historic_predictions=True, periods=96)
        self.forecast = self.model.predict(self.df_future)
        print(self.forecast.head())
    
    def get_prediction(self, time: datetime):
        # Get the prediction for a specific time
        return 0

  test = Prophet()
  print("Here")
