#!/usr/bin/env bash
# Daily Market Data Agent on Azure: Redis cache + a scheduled Container Apps Job (cron).
#
#   export SUFFIX=24680
#   COSMOS_NAME=cosmos-dhm-24680-se bash infra/deploy_data_agent.sh            # job only (no Redis)
#   ENABLE_REDIS=1 COSMOS_NAME=cosmos-dhm-24680-se bash infra/deploy_data_agent.sh
#
# The job runs `python -m scripts.data_agent` every day at 07:00 Dubai (03:00 UTC):
# it appends only the new DLD deals to Cosmos DB, rebuilds homes, publishes them to Cosmos
# (+ Redis), and the web app hot-swaps them within HOMES_SYNC_SECONDS. Safe to re-run.
set -euo pipefail

RG=${RG:-rg-dubai-home-match}
LOCATION=${LOCATION:-uaenorth}
SUFFIX=${SUFFIX:?export SUFFIX=24680 first}
APP=${APP:-baytak-ai}
JOB=${JOB:-baytak-data-agent}
IMAGE=${IMAGE:-ghcr.io/athira31-ally/baytak-ai:latest}
CRON=${CRON:-"0 3 * * *"}                 # UTC; 03:00 UTC = 07:00 Dubai
COSMOS=${COSMOS_NAME:-cosmos-dhm-$SUFFIX}
AOAI=aoai-dhm-$SUFFIX
ENABLE_REDIS=${ENABLE_REDIS:-0}
exists() { "$@" -o none >/dev/null 2>&1; }

echo ">> Reading existing resources"
COSMOS_ENDPOINT=$(az cosmosdb show -n "$COSMOS" -g "$RG" --query documentEndpoint -o tsv)
COSMOS_KEY=$(az cosmosdb keys list -n "$COSMOS" -g "$RG" --query primaryMasterKey -o tsv)
ENV_ID=$(az containerapp show -n "$APP" -g "$RG" --query properties.managedEnvironmentId -o tsv)
SECRETS=(cosmos-key="$COSMOS_KEY")
ENVV=(COSMOS_ENDPOINT="$COSMOS_ENDPOINT" COSMOS_KEY=secretref:cosmos-key HOMES_SYNC_SECONDS=300)
if exists az cognitiveservices account show -n "$AOAI" -g "$RG"; then
  AOAI_KEY=$(az cognitiveservices account keys list -n "$AOAI" -g "$RG" --query key1 -o tsv)
  AOAI_ENDPOINT=$(az cognitiveservices account show -n "$AOAI" -g "$RG" --query properties.endpoint -o tsv)
  SECRETS+=(aoai-key="$AOAI_KEY")
  ENVV+=(AZURE_OPENAI_ENDPOINT="$AOAI_ENDPOINT" AZURE_OPENAI_API_KEY=secretref:aoai-key AZURE_OPENAI_CHAT_DEPLOYMENT=chat)
fi
if [ -n "${DLD_FILE_URLS:-}" ]; then ENVV+=(DLD_FILE_URLS="$DLD_FILE_URLS"); fi
if [ -n "${DUBAI_PULSE_API_KEY:-}" ]; then
  SECRETS+=(pulse-key="$DUBAI_PULSE_API_KEY" pulse-secret="$DUBAI_PULSE_API_SECRET")
  ENVV+=(DUBAI_PULSE_API_KEY=secretref:pulse-key DUBAI_PULSE_API_SECRET=secretref:pulse-secret)
fi

if [ -n "${REDIS_URL:-}" ]; then
  SECRETS+=(redis-url="$REDIS_URL"); ENVV+=(REDIS_URL=secretref:redis-url)
elif [ "$ENABLE_REDIS" = "1" ]; then
  REDIS=redis-dhm-$SUFFIX
  echo ">> Azure Cache for Redis ($REDIS, Basic C0 - creation takes ~15 minutes)"
  if ! exists az redis show -n "$REDIS" -g "$RG"; then
    if ! az redis create -n "$REDIS" -g "$RG" -l "$LOCATION" --sku Basic --vm-size c0 --minimum-tls-version 1.2 -o none; then
      echo "!! Azure Cache for Redis could not be created here (region capacity or the offering is being"
      echo "   replaced by Azure Managed Redis). Re-run without ENABLE_REDIS=1: the app and agent use Cosmos"
      echo "   alone, or create any Redis yourself and pass REDIS_URL=rediss://... to this script."
      exit 1
    fi
  fi
  REDIS_HOST=$(az redis show -n "$REDIS" -g "$RG" --query hostName -o tsv)
  REDIS_KEY=$(az redis list-keys -n "$REDIS" -g "$RG" --query primaryKey -o tsv)
  SECRETS+=(redis-url="rediss://:${REDIS_KEY}@${REDIS_HOST}:6380/0")
  ENVV+=(REDIS_URL=secretref:redis-url)
fi

echo ">> Web app: same Cosmos/Redis settings so it can hot-swap published homes"
az containerapp secret set -n "$APP" -g "$RG" --secrets "${SECRETS[@]}" -o none
az containerapp update -n "$APP" -g "$RG" -o none --set-env-vars "${ENVV[@]}"

echo ">> Scheduled job $JOB (cron '$CRON' UTC)"
if exists az containerapp job show -n "$JOB" -g "$RG"; then
  az containerapp job secret set -n "$JOB" -g "$RG" --secrets "${SECRETS[@]}" -o none
  az containerapp job update -n "$JOB" -g "$RG" --image "$IMAGE" --cron-expression "$CRON" \
    --set-env-vars "${ENVV[@]}" -o none
else
  az containerapp job create -n "$JOB" -g "$RG" --environment "$ENV_ID" \
    --trigger-type Schedule --cron-expression "$CRON" \
    --image "$IMAGE" --cpu 1.0 --memory 2.0Gi \
    --replica-timeout 3600 --replica-retry-limit 1 --parallelism 1 --replica-completion-count 1 \
    --command "python" --args "-m" "scripts.data_agent" \
    --secrets "${SECRETS[@]}" --env-vars "${ENVV[@]}" -o none
fi

echo ""
echo "Done. Run it now instead of waiting for 07:00:"
echo "  az containerapp job start -n $JOB -g $RG"
echo "See runs and logs:"
echo "  az containerapp job execution list -n $JOB -g $RG -o table"
echo "  Logs: Azure portal -> $JOB -> Execution history -> Console logs"
echo "Check the app picked it up:  https://<your-app>/health  and  /market-brief"
