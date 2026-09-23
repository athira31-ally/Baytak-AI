#!/usr/bin/env bash
# Deploy Baytak AI to Azure with the az CLI.
#
#   az login --tenant <tenant-id>
#   export SUFFIX=24680                  # keep the same SUFFIX when re-running
#   bash infra/deploy.sh                 # Azure OpenAI + Cosmos + App Insights + Container Apps
#   ENABLE_SEARCH=1 bash infra/deploy.sh # also Azure AI Search (Basic tier costs money - see README)
#
# Safe to re-run: anything that already exists is reused, not recreated.
# The chat model version is picked automatically (newest one that isn't being
# retired). Override with CHAT_MODEL=... CHAT_MODEL_VERSION=...
set -euo pipefail

RG=${RG:-rg-dubai-home-match}
LOCATION=${LOCATION:-uaenorth}
AOAI_LOCATION=${AOAI_LOCATION:-swedencentral}
COSMOS_LOCATION=${COSMOS_LOCATION:-$LOCATION}
SUFFIX=${SUFFIX:?Set SUFFIX first, e.g. export SUFFIX=24680 (reuse the same value on re-runs)}
APP=${APP:-baytak-ai}
CHAT_MODEL=${CHAT_MODEL:-gpt-4.1-mini}
CHAT_MODEL_VERSION=${CHAT_MODEL_VERSION:-}
CHAT_DEPLOYMENT=chat
CHAT_SKU=${CHAT_SKU:-GlobalStandard}
CHAT_CAPACITY=${CHAT_CAPACITY:-10}   # thousands of tokens per minute
ENABLE_SEARCH=${ENABLE_SEARCH:-0}
SKIP_AOAI=${SKIP_AOAI:-0}   # 1 = deploy without Azure OpenAI (agent runs in offline mode)

exists() { "$@" -o none >/dev/null 2>&1; }

echo ">> Registering resource providers"
for p in Microsoft.App Microsoft.OperationalInsights Microsoft.CognitiveServices Microsoft.DocumentDB Microsoft.Insights Microsoft.Search; do
  az provider register -n "$p" --wait >/dev/null
done
az extension add -n containerapp --upgrade -y --only-show-errors >/dev/null
az extension add -n application-insights --upgrade -y --only-show-errors >/dev/null

echo ">> Resource group $RG ($LOCATION)"
az group create -n "$RG" -l "$LOCATION" -o none

AOAI_ENV=()
AOAI_SECRET=()
if [ "$SKIP_AOAI" = "1" ]; then
  echo ">> Skipping Azure OpenAI (SKIP_AOAI=1): the agent will run in offline mode"
else
  echo ">> Azure OpenAI ($AOAI_LOCATION)"
  AOAI=aoai-dhm-$SUFFIX
  if exists az cognitiveservices account show -n "$AOAI" -g "$RG"; then
    echo "   reusing $AOAI"
  else
    az cognitiveservices account create -n "$AOAI" -g "$RG" -l "$AOAI_LOCATION" --kind OpenAI --sku S0 \
      --custom-domain "$AOAI" -o none
  fi

  if [ -z "$CHAT_MODEL_VERSION" ]; then
    CHAT_MODEL_VERSION=$(az cognitiveservices account list-models -n "$AOAI" -g "$RG" \
      --query "[?name=='$CHAT_MODEL' && format=='OpenAI' && lifecycleStatus!='Deprecating' && lifecycleStatus!='Deprecated' && lifecycleStatus!='Legacy'].version" \
      -o tsv | sort | tail -1)
  fi
  if [ -z "$CHAT_MODEL_VERSION" ]; then
    echo "!! No deployable version of $CHAT_MODEL in $AOAI_LOCATION. Chat models you can use here:"
    az cognitiveservices account list-models -n "$AOAI" -g "$RG" \
      --query "[?format=='OpenAI' && lifecycleStatus!='Deprecating' && lifecycleStatus!='Deprecated' && lifecycleStatus!='Legacy' && contains(name,'gpt')].{model:name, version:version, status:lifecycleStatus}" -o table
    echo "Re-run with e.g.: CHAT_MODEL=<model> bash infra/deploy.sh"
    exit 1
  fi
  echo "   model: $CHAT_MODEL $CHAT_MODEL_VERSION (deployment name: $CHAT_DEPLOYMENT)"
  if exists az cognitiveservices account deployment show -n "$AOAI" -g "$RG" --deployment-name "$CHAT_DEPLOYMENT"; then
    echo "   reusing deployment '$CHAT_DEPLOYMENT'"
  else
    if ! az cognitiveservices account deployment create -n "$AOAI" -g "$RG" --deployment-name "$CHAT_DEPLOYMENT" \
      --model-name "$CHAT_MODEL" --model-version "$CHAT_MODEL_VERSION" --model-format OpenAI \
      --sku-name "$CHAT_SKU" --sku-capacity "$CHAT_CAPACITY" -o none; then
      echo "!! Deployment failed. Models with quota in $AOAI_LOCATION (limit > 0):"
      az cognitiveservices usage list -l "$AOAI_LOCATION" \
        --query "[?limit > \`0\` && starts_with(name.value, 'OpenAI.')].{quota:name.value, used:currentValue, limit:limit}" -o table
      echo "Quota names read OpenAI.<SKU>.<model>. Re-run with e.g.: CHAT_MODEL=<model> CHAT_SKU=<SKU> bash infra/deploy.sh"
      exit 1
    fi
  fi
  AOAI_ENDPOINT=$(az cognitiveservices account show -n "$AOAI" -g "$RG" --query properties.endpoint -o tsv)
  AOAI_KEY=$(az cognitiveservices account keys list -n "$AOAI" -g "$RG" --query key1 -o tsv)
  AOAI_ENV=(AZURE_OPENAI_ENDPOINT="$AOAI_ENDPOINT" AZURE_OPENAI_API_KEY=secretref:aoai-key AZURE_OPENAI_CHAT_DEPLOYMENT="$CHAT_DEPLOYMENT")
  AOAI_SECRET=(aoai-key="$AOAI_KEY")
fi

echo ">> Cosmos DB (serverless, $COSMOS_LOCATION)"
COSMOS=${COSMOS_NAME:-cosmos-dhm-$SUFFIX}
if exists az cosmosdb show -n "$COSMOS" -g "$RG"; then
  STATE=$(az cosmosdb show -n "$COSMOS" -g "$RG" --query provisioningState -o tsv)
  if [ "$STATE" != "Succeeded" ]; then
    echo "!! $COSMOS exists but is in state '$STATE'. Use a new name, e.g.: COSMOS_NAME=$COSMOS-2 COSMOS_LOCATION=swedencentral"
    exit 1
  fi
  echo "   reusing $COSMOS"
else
  az cosmosdb create -n "$COSMOS" -g "$RG" --locations regionName="$COSMOS_LOCATION" --capabilities EnableServerless -o none
fi
COSMOS_ENDPOINT=$(az cosmosdb show -n "$COSMOS" -g "$RG" --query documentEndpoint -o tsv)
COSMOS_KEY=$(az cosmosdb keys list -n "$COSMOS" -g "$RG" --query primaryMasterKey -o tsv)

echo ">> Application Insights"
APPI=appi-dhm-$SUFFIX
if exists az monitor app-insights component show --app "$APPI" -g "$RG"; then
  echo "   reusing $APPI"
else
  az monitor app-insights component create --app "$APPI" -g "$RG" -l "$LOCATION" --kind web --application-type web -o none
fi
AI_CONN=$(az monitor app-insights component show --app "$APPI" -g "$RG" --query connectionString -o tsv)


SEARCH_ENV=()
if [ "$ENABLE_SEARCH" = "1" ]; then
  echo ">> Azure AI Search (basic)"
  SEARCH=srch-dhm-$SUFFIX
  exists az search service show -n "$SEARCH" -g "$RG" || az search service create -n "$SEARCH" -g "$RG" -l "$LOCATION" --sku basic -o none
  SEARCH_KEY=$(az search admin-key show --service-name "$SEARCH" -g "$RG" --query primaryKey -o tsv)
  echo "   Next: deploy text-embedding-3-small, set EMBEDDING_PROVIDER=azure, run bootstrap + scripts.index_azure_search,"
  echo "   then rebuild with --build-arg BOOTSTRAP=0. See README > 'Switching retrieval to Azure AI Search'."
  SEARCH_ENV=(AZURE_SEARCH_ENDPOINT="https://$SEARCH.search.windows.net" AZURE_SEARCH_API_KEY=secretref:search-key)
fi

echo ">> Container App (builds the Dockerfile in Azure Container Registry - takes a few minutes)"
# Azure allows only 1 Container Apps environment per region on many subscriptions,
# so reuse an existing one in $LOCATION (e.g. from another project) if there is one.
if [ -z "${CONTAINERAPP_ENV:-}" ]; then
  WANT=$(echo "$LOCATION" | tr -d ' ' | tr '[:upper:]' '[:lower:]')
  CONTAINERAPP_ENV=$(az containerapp env list --query "[].[id, location]" -o tsv \
    | while IFS=$'\t' read -r id loc; do
        [ "$(echo "$loc" | tr -d ' ' | tr '[:upper:]' '[:lower:]')" = "$WANT" ] && echo "$id"
      done | head -1 || true)
fi
ENV_ARGS=()
if [ -n "$CONTAINERAPP_ENV" ]; then
  echo "   reusing Container Apps environment: ${CONTAINERAPP_ENV##*/}"
  ENV_ARGS=(--environment "$CONTAINERAPP_ENV")
fi
# How the image gets built:
#   default         Azure builds the Dockerfile (ACR Tasks) - blocked on some free/trial subscriptions
#   BUILD=local     build with Docker Desktop on this Mac and push to your Azure Container Registry
#   IMAGE=<ref>     use a ready-made public image, e.g. ghcr.io/<you>/baytak-ai:latest
BUILD=${BUILD:-acr}
if [ -n "${IMAGE:-}" ]; then
  echo "   using prebuilt image $IMAGE"
  az containerapp up -n "$APP" -g "$RG" -l "$LOCATION" --image "$IMAGE" --ingress external --target-port 8000 \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"}
elif [ "$BUILD" = "local" ]; then
  command -v docker >/dev/null || { echo "!! Docker not found. Install Docker Desktop, or use the GitHub build (see README)."; exit 1; }
  ACR=${ACR_NAME:-acrdhm$SUFFIX}
  exists az acr show -n "$ACR" -g "$RG" || az acr create -n "$ACR" -g "$RG" -l "$LOCATION" --sku Basic --admin-enabled true -o none
  az acr update -n "$ACR" --admin-enabled true -o none
  LOGIN_SERVER=$(az acr show -n "$ACR" --query loginServer -o tsv)
  TAG="$LOGIN_SERVER/$APP:$(date +%Y%m%d%H%M)"
  az acr login -n "$ACR"
  echo "   building $TAG for linux/amd64 (Apple Silicon needs this flag)"
  docker buildx build --platform linux/amd64 -t "$TAG" --push .
  ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
  ACR_PASS=$(az acr credential show -n "$ACR" --query "passwords[0].value" -o tsv)
  az containerapp up -n "$APP" -g "$RG" -l "$LOCATION" --image "$TAG" --ingress external --target-port 8000 \
    --registry-server "$LOGIN_SERVER" --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"}
else
  if ! az containerapp up -n "$APP" -g "$RG" -l "$LOCATION" --source . --ingress external --target-port 8000 \
      ${ENV_ARGS[@]+"${ENV_ARGS[@]}"}; then
    echo "!! Cloud build failed. If the error says 'TasksOperationsNotAllowed', your subscription blocks ACR Tasks."
    echo "   Re-run with BUILD=local (needs Docker Desktop) or IMAGE=ghcr.io/<you>/baytak-ai:latest"
    exit 1
  fi
fi

echo ">> Secrets + environment"
SECRETS=(cosmos-key="$COSMOS_KEY" appi-conn="$AI_CONN" ${AOAI_SECRET[@]+"${AOAI_SECRET[@]}"})
if [ "$ENABLE_SEARCH" = "1" ]; then SECRETS+=(search-key="$SEARCH_KEY"); fi
LIVE_ENV=()
if [ -n "${DUBAI_PULSE_API_KEY:-}" ] && [ -n "${DUBAI_PULSE_API_SECRET:-}" ]; then
  echo "   live DLD data: ON (Dubai Pulse API)"
  SECRETS+=(pulse-key="$DUBAI_PULSE_API_KEY" pulse-secret="$DUBAI_PULSE_API_SECRET")
  LIVE_ENV=(DUBAI_PULSE_API_KEY=secretref:pulse-key DUBAI_PULSE_API_SECRET=secretref:pulse-secret)
fi
az containerapp secret set -n "$APP" -g "$RG" --secrets "${SECRETS[@]}" -o none
az containerapp update -n "$APP" -g "$RG" --min-replicas 0 --max-replicas 2 -o none --set-env-vars \
  ${AOAI_ENV[@]+"${AOAI_ENV[@]}"} \
  COSMOS_ENDPOINT="$COSMOS_ENDPOINT" COSMOS_KEY=secretref:cosmos-key \
  APPLICATIONINSIGHTS_CONNECTION_STRING=secretref:appi-conn ${SEARCH_ENV[@]+"${SEARCH_ENV[@]}"} \
  ${LIVE_ENV[@]+"${LIVE_ENV[@]}"}

URL=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo ""
echo "Deployed: https://$URL"
echo "Health:   https://$URL/health   (expect \"llm\": \"azure-openai\"; first request after idle takes ~20s)"
echo "API docs: https://$URL/docs"
echo ""
echo "Tear down everything: az group delete -n $RG --yes --no-wait"
echo "Daily data agent + Redis: bash infra/deploy_data_agent.sh  (see README)"
