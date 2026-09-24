#!/usr/bin/env bash
# Move the Market Data Agent into Microsoft Foundry Agent Service.
#
#   export SUFFIX=24680
#   bash infra/deploy_foundry.sh
#
# Creates (safe to re-run):
#   - a Foundry resource + project        (the agent, its versions and conversations live here)
#   - a gpt-5-mini deployment in it
#   - "Foundry User" role for you (to run it from your Mac) and for the daily job's managed identity
#   - FOUNDRY_PROJECT_ENDPOINT on the daily job -> the job now runs the Foundry agent
set -euo pipefail

RG=${RG:-rg-dubai-home-match}
SUFFIX=${SUFFIX:?export SUFFIX=24680 first}
FLOC=${FOUNDRY_LOCATION:-swedencentral}
FOUNDRY=${FOUNDRY_NAME:-foundry-dhm-$SUFFIX}
PROJECT=${FOUNDRY_PROJECT:-baytak}
MODEL=${MODEL:-gpt-5-mini}
MODEL_VERSION=${MODEL_VERSION:-2025-08-07}
JOB=${JOB:-baytak-data-agent}
FOUNDRY_USER_ROLE=53ca6127-db72-4b80-b1b0-d745d6d5456d     # "Foundry User"
exists() { "$@" -o none >/dev/null 2>&1; }

echo ">> Foundry resource $FOUNDRY ($FLOC)"
if exists az cognitiveservices account show -n "$FOUNDRY" -g "$RG"; then
  echo "   reusing"
else
  az cognitiveservices account create -n "$FOUNDRY" -g "$RG" -l "$FLOC" --kind AIServices --sku S0 \
    --custom-domain "$FOUNDRY" --assign-identity --allow-project-management true -o none
fi

echo ">> Foundry project $PROJECT"
if exists az cognitiveservices account project show -n "$FOUNDRY" -g "$RG" --project-name "$PROJECT"; then
  echo "   reusing"
else
  az cognitiveservices account project create -n "$FOUNDRY" -g "$RG" --project-name "$PROJECT" -l "$FLOC" -o none
fi

echo ">> Model deployment $MODEL"
if exists az cognitiveservices account deployment show -n "$FOUNDRY" -g "$RG" --deployment-name "$MODEL"; then
  echo "   reusing"
else
  az cognitiveservices account deployment create -n "$FOUNDRY" -g "$RG" --deployment-name "$MODEL" \
    --model-name "$MODEL" --model-version "$MODEL_VERSION" --model-format OpenAI \
    --sku-name GlobalStandard --sku-capacity 10 -o none
fi

PROJECT_ID=$(az cognitiveservices account project show -n "$FOUNDRY" -g "$RG" --project-name "$PROJECT" --query id -o tsv)
ENDPOINT=$(az cognitiveservices account project show -n "$FOUNDRY" -g "$RG" --project-name "$PROJECT" \
  --query 'properties.endpoints."AI Foundry API"' -o tsv)

echo ">> Role: Foundry User for you"
ME=$(az ad signed-in-user show --query id -o tsv)
az role assignment create --role "$FOUNDRY_USER_ROLE" --assignee-object-id "$ME" \
  --assignee-principal-type User --scope "$PROJECT_ID" -o none 2>/dev/null || echo "   (already assigned)"

if exists az containerapp job show -n "$JOB" -g "$RG"; then
  echo ">> Role: Foundry User for the daily job's managed identity"
  JOB_PRINCIPAL=$(az containerapp job identity assign -n "$JOB" -g "$RG" --system-assigned --query principalId -o tsv)
  az role assignment create --role "$FOUNDRY_USER_ROLE" --assignee-object-id "$JOB_PRINCIPAL" \
    --assignee-principal-type ServicePrincipal --scope "$PROJECT_ID" -o none 2>/dev/null || echo "   (already assigned)"
  echo ">> Pointing the job at Foundry"
  az containerapp job update -n "$JOB" -g "$RG" -o none \
    --set-env-vars FOUNDRY_PROJECT_ENDPOINT="$ENDPOINT" FOUNDRY_MODEL_DEPLOYMENT="$MODEL"
fi

echo ""
echo "Foundry project endpoint: $ENDPOINT"
echo "Add to your .env for local runs:"
echo "  FOUNDRY_PROJECT_ENDPOINT=$ENDPOINT"
echo "  FOUNDRY_MODEL_DEPLOYMENT=$MODEL"
echo "Role assignments can take ~5 minutes to apply. Then open https://ai.azure.com -> project '$PROJECT' -> Agents."
