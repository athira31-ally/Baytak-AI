#!/usr/bin/env bash
# Turn on the new AI features on the live web app (safe to re-run):
#   - Azure AI Search (Free tier) for hybrid retrieval  -> the app publishes the homes index itself
#   - Azure AI Content Safety Prompt Shields             -> uses the Foundry resource you already have
#   - LangGraph multi-agent engine, MCP endpoint (/mcp/), model capability hint
#
#   export SUFFIX=24680
#   bash infra/deploy_ai_upgrades.sh
#   MCP_API_KEY=$(openssl rand -hex 16) bash infra/deploy_ai_upgrades.sh     # optional: protect /mcp
set -euo pipefail

RG=${RG:-rg-dubai-home-match}
SUFFIX=${SUFFIX:?export SUFFIX=24680 first}
APP=${APP:-baytak-ai}
SEARCH=${SEARCH_NAME:-srch-dhm-$SUFFIX}
SEARCH_SKU=${SEARCH_SKU:-free}
FOUNDRY=${FOUNDRY_NAME:-foundry-dhm-$SUFFIX}
exists() { "$@" -o none >/dev/null 2>&1; }

echo ">> Azure AI Search $SEARCH ($SEARCH_SKU)"
if exists az search service show -n "$SEARCH" -g "$RG"; then
  echo "   reusing"
else
  created=""
  for loc in ${SEARCH_LOCATIONS:-uaenorth swedencentral westeurope}; do
    if az search service create -n "$SEARCH" -g "$RG" -l "$loc" --sku "$SEARCH_SKU" -o none 2>/tmp/search_err; then
      echo "   created in $loc"; created=1; break
    fi
    echo "   $loc: $(head -c 200 /tmp/search_err)"
  done
  [ -n "$created" ] || { echo "!! Could not create Azure AI Search (a subscription can have only ONE free service)."; exit 1; }
fi
SEARCH_ENDPOINT="https://$SEARCH.search.windows.net"
SEARCH_KEY=$(az search admin-key show --service-name "$SEARCH" -g "$RG" --query primaryKey -o tsv)

echo ">> Content Safety (Prompt Shields) via Foundry resource $FOUNDRY"
CS_ENDPOINT="https://$FOUNDRY.cognitiveservices.azure.com"
CS_KEY=$(az cognitiveservices account keys list -n "$FOUNDRY" -g "$RG" --query key1 -o tsv)

SECRETS=(search-key="$SEARCH_KEY" cs-key="$CS_KEY")
ENVV=(AZURE_SEARCH_ENDPOINT="$SEARCH_ENDPOINT" AZURE_SEARCH_API_KEY=secretref:search-key
      CONTENT_SAFETY_ENDPOINT="$CS_ENDPOINT" CONTENT_SAFETY_KEY=secretref:cs-key
      AGENT_ENGINE=langgraph AZURE_OPENAI_CHAT_MODEL=${CHAT_MODEL:-gpt-5-mini})
if [ -n "${MCP_API_KEY:-}" ]; then SECRETS+=(mcp-key="$MCP_API_KEY"); ENVV+=(MCP_API_KEY=secretref:mcp-key); fi

echo ">> Web app $APP: secrets + settings (creates a new revision)"
az containerapp secret set -n "$APP" -g "$RG" --secrets "${SECRETS[@]}" -o none
az containerapp update -n "$APP" -g "$RG" --image ${IMAGE:-ghcr.io/athira31-ally/baytak-ai:latest} \
  --set-env-vars "${ENVV[@]}" IMAGE_SHA=$(git rev-parse --short HEAD 2>/dev/null || date +%s) -o none

URL=https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo ""
echo "Done. In ~1 minute check:"
echo "  $URL/health        -> agent_engine: langgraph, retrieval.backend: azure-ai-search"
echo "  $URL/docs          -> try POST /chat (response lists the agents that ran)"
echo "  MCP endpoint:        $URL/mcp/"
echo "  Evaluate live:       python -m scripts.run_evals --url $URL --no-gate"
