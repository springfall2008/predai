# PredAI

Predai is a NeuralProphet integration for Home Assistant than can be used to predict future values of sensors given the history of that sensor.

See: https://neuralprophet.com/contents.html for documentation on the base AI library

## Installation:

PredAI is a Home Assistant Add On, install it by going to:
- Settings, Add-ons, Add-on Store, '...' (top right), Repositories,
- Put 'https://github.com/springfall2008/predai' into the bottom box and Click Add, Close
- Select PredAI from the list and select install.
- Wait quite a while (could be 15-20 minutes) to install
- Select Start

## Configuration
- Edit '/addon_configs/93e0bb20_predai/predai.yaml' to configure the tool.
- Then click Restart on the Add-on to ensure the configuration is read again
- Check the log for any errors or warnings

Example configuration

```yaml
update_every: 30
#ha_url: "http://homeassistant.local:8123"
#ha_key: "xxxxxxxxxxxx"
sensors:
  - name: sensor.givtcp_sa2243g277_load_energy_today_kwh
    subtract: sensor.wallbox_portal_added_energy
    days: 14
    incrementing: True
    max_increment: 10
    reset_daily: True
    interval: 30
    units: kWh
    future_periods: 100
    database: True
    reset_low: 1.0
    reset_high: 2.0
    country: UK
    max_age: 365
  - name: sensor.external_temperature
    days: 14
    interval: 30
    incrementing: False
    units: c
    future_periods: 100
    country: UK
```

**update_every** Sets the frequency of updates in minutes
**ha_url** Set this to your Home Assistant URL (e.g. http://homeassistant.local:8123) you are not running as a Home Assistant addon (e.g. in Docker)
**ha_key** Set this to your Home Assistant API key if you are not running as a Home Assistant addon (e.g. in Docker)

**Sensors** This is an array of entities to predict the future on

  - **Name** Give the name of the entity exactly as in Home Assistant
  - **Subtact** can be used to subtract another numerical value from the first entity, mostly used to remove things like car charging from energy data. Can also be a list of sensor names to subtract.
  - **days** Sets how many days in the past to take the history from
  - **incrementing** - When true the sensor is always incrementing (e.g. energy used), but can include resets. When False they are individual values.
  - **max_increment** - Defines the maximum increase or decrease an incrementing sensor can have before its considered a spike and ignored
  - **reset_daily** - When true the sensor value is reset to 0 at midnight (e.g. energy per day)
  - **interval** - Sets the prediction inverval, should divide into 60 e.g. 5, 10 , 15 , 30
  - **Units** - Sets the output Unit to report in HA
  - **future_periods** - Sets the number of periods (of interval minutes) to predict into the future, 100 is recommended for Predbat use to cover 48 hours + time until the next PredAI run.
  - **database** - When True (default) all data is stored in a sqllite3 database in the addon directory, this will keep a full history beyond what HA keeps and use
that history for training. You can browse the data using an SQL Lite viewer on your computer.
  - **export_days** - Sets how many days of history to include in the HA entities that are created, recommended values are 7-14. The default is **days**
  - **reset_low/reset_high** - For incrementing sensors if the sensor goes above **reset_high** and then falls below **reset_low** then its considered a reset even
  if it never goes to 0.
  - **country** - When set adds in the specified countries holidays (see https://python-holidays.readthedocs.io/en/latest/)

A new sensor with the name **name**_prediction will be created, this will contain two series:
  - **results** contains the time series of the predictions, starts in the past so you can plot corrolation
  - **source** contains the original source data that was used to make the prediction

## Charting

You can use Apex charts to plot the predictions for example for the above sensor we plot a corrolation chart

<img width="494" alt="image" src="https://github.com/springfall2008/predai/assets/48591903/070ae165-f242-4ce9-a7e1-aef9294c82af">


```yaml
type: custom:apexcharts-card
header:
  show: true
  title: Data prediction
  show_states: true
  colorize_states: true
graph_span: 8d
span:
  start: day
  offset: '-6d'
now:
  show: true
yaxis:
  - min: 0
series:
  - entity: sensor.givtcp_sa2243g277_load_energy_today_kwh
    stroke_width: 1
    curve: smooth
    name: Load
  - entity: sensor.givtcp_sa2243g277_load_energy_today_kwh_prediction
    stroke_width: 1
    curve: smooth
    name: AI Load
    show:
      in_header: raw
    data_generator: >
      let res = []; for (const [key, value] of
      Object.entries(entity.attributes.results)) { res.push([new
      Date(key).getTime(), value]); } return res.sort((a, b) => { return a[0] -
      b[0]  })
  - entity: sensor.givtcp_sa2243g277_load_energy_today_kwh_prediction
    stroke_width: 1
    curve: smooth
    name: AI Load Source
    show:
      in_header: raw
    data_generator: >
      let res = []; for (const [key, value] of
      Object.entries(entity.attributes.source)) { res.push([new
      Date(key).getTime(), value]); } return res.sort((a, b) => { return a[0] -
      b[0]  })
```

## Use within Predbat

You can use the AI load prediction in Predbat by setting it in the apps.yaml

This will disable the use of Predbat internal predictions and instead use the AI based one for load forecasts.

e.g.

```yaml
  load_forecast_only: True
  load_forecast:
     - sensor.givtcp_{geserial}_load_energy_today_kwh_prediction$results
```
