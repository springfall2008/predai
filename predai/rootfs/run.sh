#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "PredAI Custom add‑on starting…"

TARGET_SENSOR="$(bashio::config 'target_sensor')"
FORECAST_HOURS="$(bashio::config 'forecast_hours')"

bashio::log.info "Forecasting ${FORECAST_HOURS:-24} h for ${TARGET_SENSOR}"

exec python3 predai_main.py \
     --target "${TARGET_SENSOR}" \
     --hours "${FORECAST_HOURS}"
