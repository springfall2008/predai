import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os

# Fix for PyTorch 2.6+ weights_only=True default - must be done BEFORE importing pred ai
try:
    import torch
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        # Force weights_only=False for all checkpoint loads
        # This is safe because we're loading locally-created NeuralProphet checkpoints
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except (ImportError, AttributeError):
    pass

# Add the rootfs directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'predai', 'rootfs'))

from predai import HAInterface, Prophet, timestr_to_datetime


class TestPredAI(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for PredAI with mocked HAInterface and sine wave data.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.ha_url = "http://test-ha-url"
        self.ha_key = "test-key"
        self.sensor_name = "sensor.test_sine_wave"
        self.period = 30  # 30 minute intervals
        
    def generate_sine_wave_data(self, start_time, hours=168, amplitude=50, offset=100):
        """
        Generate synthetic sine wave data for testing.
        
        Args:
            start_time: Starting datetime
            hours: Number of hours of data to generate
            amplitude: Amplitude of the sine wave
            offset: Vertical offset of the sine wave
            
        Returns:
            List of dictionaries mimicking HA history format
        """
        data = []
        current_time = start_time
        
        # Generate one data point per hour
        for i in range(hours):
            # Create sine wave: one complete cycle per 24 hours
            value = offset + amplitude * np.sin(2 * np.pi * i / 24)
            
            timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%S%z")
            data.append({
                "state": str(value),
                "last_updated": timestamp
            })
            current_time += timedelta(hours=1)
            
        return data

    async def test_sine_wave_prediction(self):
        """Test Prophet prediction with sine wave data."""
        # Create Prophet instance
        prophet = Prophet(period=self.period)
        
        # Generate test data (7 days of hourly sine wave data)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0, minute=0)
        start_time = now - timedelta(days=7)
        end_time = now
        
        sine_data = self.generate_sine_wave_data(start_time, hours=7*24, amplitude=50, offset=100)
        
        # Process the dataset
        dataset, last_value = await prophet.process_dataset(
            sensor_name=self.sensor_name,
            new_data=sine_data,
            start_time=start_time,
            end_time=end_time,
            incrementing=False
        )
        
        # Verify dataset was created
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
        self.assertIn("ds", dataset.columns)
        self.assertIn("y", dataset.columns)
        
        # Verify sine wave pattern (values should oscillate around 100)
        mean_value = dataset["y"].mean()
        self.assertAlmostEqual(mean_value, 100, delta=10)
        
        # Verify data range matches sine wave bounds
        min_value = dataset["y"].min()
        max_value = dataset["y"].max()
        self.assertGreater(min_value, 40)  # offset - amplitude - tolerance
        self.assertLess(max_value, 160)     # offset + amplitude + tolerance
        
        print(f"Dataset created with {len(dataset)} rows")
        print(f"Value range: {min_value:.2f} to {max_value:.2f}, mean: {mean_value:.2f}")
        
        # Train the model
        future_periods = 48  # Predict 24 hours ahead (48 x 30min periods)
        await prophet.train(dataset, future_periods, n_lags=0)
        
        # Verify model was trained
        self.assertIsNotNone(prophet.model)
        self.assertIsNotNone(prophet.forecast)
        
        # Verify forecast contains predictions
        self.assertGreater(len(prophet.forecast), len(dataset))
        
        print(f"Forecast created with {len(prophet.forecast)} rows")
        print(f"Forecast head:\n{prophet.forecast.head()}")
        
    async def test_mocked_ha_interface_prediction(self):
        """Test full prediction flow with mocked HAInterface."""
        # Mock HAInterface
        mock_interface = AsyncMock(spec=HAInterface)
        
        # Setup time
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0, minute=0)
        start_time = now - timedelta(days=7)
        
        # Generate sine wave test data
        sine_data = self.generate_sine_wave_data(start_time, hours=7*24, amplitude=30, offset=75)
        
        # Mock get_history to return our sine wave data
        mock_interface.get_history.return_value = (
            sine_data,
            start_time,
            now
        )
        
        # Mock set_state to capture the prediction
        captured_state = {}
        
        async def capture_set_state(entity_id, state, attributes=None):
            captured_state['entity_id'] = entity_id
            captured_state['state'] = state
            captured_state['attributes'] = attributes
            
        mock_interface.set_state.side_effect = capture_set_state
        
        # Create Prophet and process
        prophet = Prophet(period=self.period)
        
        # Get history (in real code this calls interface.get_history)
        history_data, start, end = await mock_interface.get_history(
            self.sensor_name, now, days=7
        )
        
        # Process dataset
        dataset, last_value = await prophet.process_dataset(
            sensor_name=self.sensor_name,
            new_data=history_data,
            start_time=start,
            end_time=end,
            incrementing=False
        )
        
        # Train model
        await prophet.train(dataset, future_periods=48)
        
        # Save prediction
        await prophet.save_prediction(
            entity=f"{self.sensor_name}_prediction",
            now=now,
            interface=mock_interface,
            start=end,
            incrementing=False,
            units="W",
            days=7
        )
        
        # Verify set_state was called
        mock_interface.set_state.assert_called_once()
        
        # Verify captured prediction
        self.assertIn('entity_id', captured_state)
        self.assertEqual(captured_state['entity_id'], f"{self.sensor_name}_prediction")
        self.assertIsNotNone(captured_state['state'])
        self.assertIsInstance(captured_state['state'], (int, float))
        
        # Verify attributes
        attrs = captured_state['attributes']
        self.assertIn('results', attrs)
        self.assertIn('source', attrs)
        self.assertIn('unit_of_measurement', attrs)
        self.assertEqual(attrs['unit_of_measurement'], "W")
        
        # Verify timeseries data
        results = attrs['results']
        self.assertGreater(len(results), 0)
        
        print(f"Prediction saved: entity={captured_state['entity_id']}, state={captured_state['state']}")
        print(f"Timeseries contains {len(results)} data points")
        
    async def test_incrementing_sensor(self):
        """Test Prophet with incrementing sensor (like energy meters)."""
        prophet = Prophet(period=self.period)
        
        # Generate incrementing data (simulating an energy meter)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0, minute=0)
        start_time = now - timedelta(days=7)
        
        data = []
        current_time = start_time
        cumulative = 0
        
        # Generate 7 days of hourly increments with sine wave variation
        for i in range(7*24):
            # Increment varies in a sine pattern (simulating daily usage pattern)
            increment = 0.5 + 0.3 * np.sin(2 * np.pi * i / 24)
            cumulative += increment
            
            timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%S%z")
            data.append({
                "state": str(cumulative),
                "last_updated": timestamp
            })
            current_time += timedelta(hours=1)
        
        # Process as incrementing sensor
        dataset, last_value = await prophet.process_dataset(
            sensor_name="sensor.energy_total",
            new_data=data,
            start_time=start_time,
            end_time=now,
            incrementing=True,
            reset_low=0.1,
            reset_high=1.0
        )
        
        # Verify dataset
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
        
        # All values should be positive increments
        self.assertTrue((dataset["y"] >= 0).all())
        
        # Train and verify
        await prophet.train(dataset, future_periods=48)
        self.assertIsNotNone(prophet.model)
        
        print(f"Incrementing sensor test: {len(dataset)} rows processed")
        print(f"Value range: {dataset['y'].min():.2f} to {dataset['y'].max():.2f}")


if __name__ == '__main__':
    unittest.main()
