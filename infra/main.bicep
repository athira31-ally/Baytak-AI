// Baytak AI - everything in one Bicep file.
//
//   az deployment group create -g rg-dubai-home-match -f infra/main.bicep -p infra/main.bicepparam
//
// What it deploys (names default to the resources you already have, so re-deploying adopts them):
//   AI        Microsoft Foundry resource + project + gpt-5-mini deployment. Serves BOTH agents:
//             - Home-search agent (web app): Azure OpenAI function calling, 5 tools, in app code
//             - Market Data Agent (daily job): Foundry Agent Service agent, 7 function tools
//   Data      Cosmos DB serverless: `feedback`, `market_tx` (per-deal TTL), `agent_memory`
//   Monitor   Log Analytics + Application Insights
//   Compute   Container App (web) + scheduled Container Apps Job (daily agent, 07:00 Dubai)
//   Access    "Foundry User" role for the job's managed identity on the Foundry project
//
// Not in Bicep, by design: the agents' prompts and tool definitions. They are data-plane objects
// that the code creates and versions itself (app/agents/*, app/data/foundry_agent.py), so the
// prompt and tools always match the code that implements the tools.

targetScope = 'resourceGroup'

@description('Suffix used in resource names (matches your existing resources).')
param suffix string = '24680'
param location string = 'uaenorth'
@description('Region for the Foundry resource (model availability/quota).')
param foundryLocation string = 'swedencentral'
@description('Region for Cosmos DB (UAE North had no capacity).')
param cosmosLocation string = 'swedencentral'
param cosmosName string = 'cosmos-dhm-${suffix}-se'
param foundryName string = 'foundry-dhm-${suffix}'
param foundryProjectName string = 'baytak'
param modelName string = 'gpt-5-mini'
param modelVersion string = '2025-08-07'
@description('Thousands of tokens per minute for the model deployment.')
param modelCapacity int = 10
param appName string = 'baytak-ai'
param jobName string = 'baytak-data-agent'
param image string = 'ghcr.io/athira31-ally/baytak-ai:latest'
@description('Existing Container Apps environment ID (your subscription allows one per region). Leave empty to create one.')
param containerAppsEnvironmentId string = ''
@description('UTC cron. 03:00 UTC = 07:00 Dubai.')
param cronExpression string = '0 3 * * *'
@description('Comma-separated data.dubai CSV links for the daily agent (optional).')
param dldFileUrls string = ''

var foundryUserRoleId = '53ca6127-db72-4b80-b1b0-d745d6d5456d' // "Foundry User"

// ------------------------------------------------------------------ monitoring
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-dhm-${suffix}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-dhm-${suffix}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

// --------------------------------------------------------------- Foundry (AI)
resource foundry 'Microsoft.CognitiveServices/accounts@2025-06-01' = {
  name: foundryName
  location: foundryLocation
  kind: 'AIServices'
  sku: { name: 'S0' }
  identity: { type: 'SystemAssigned' }
  properties: {
    customSubDomainName: foundryName
    allowProjectManagement: true
    publicNetworkAccess: 'Enabled'
  }
}

resource project 'Microsoft.CognitiveServices/accounts/projects@2025-06-01' = {
  parent: foundry
  name: foundryProjectName
  location: foundryLocation
  identity: { type: 'SystemAssigned' }
  properties: {
    displayName: 'Baytak AI'
    description: 'Agents for Baytak AI: home-search agent and daily Market Data Agent'
  }
}

resource model 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: foundry
  name: modelName
  sku: {
    name: 'GlobalStandard'
    capacity: modelCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: modelName
      version: modelVersion
    }
  }
}

// --------------------------------------------------------------------- Cosmos
resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2024-11-15' = {
  name: cosmosName
  location: cosmosLocation
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    capabilities: [ { name: 'EnableServerless' } ]
    locations: [ { locationName: cosmosLocation, failoverPriority: 0 } ]
    consistencyPolicy: { defaultConsistencyLevel: 'Session' }
  }
}

resource db 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-11-15' = {
  parent: cosmos
  name: 'homematch'
  properties: { resource: { id: 'homematch' } }
}

resource feedback 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = {
  parent: db
  name: 'feedback'
  properties: {
    resource: {
      id: 'feedback'
      partitionKey: { paths: [ '/session_id' ], kind: 'Hash' }
    }
  }
}

resource marketTx 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = {
  parent: db
  name: 'market_tx'
  properties: {
    resource: {
      id: 'market_tx'
      partitionKey: { paths: [ '/ym' ], kind: 'Hash' }
      defaultTtl: -1 // TTL on, set per deal (deal date + 400 days)
    }
  }
}

resource agentMemory 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = {
  parent: db
  name: 'agent_memory'
  properties: {
    resource: {
      id: 'agent_memory'
      partitionKey: { paths: [ '/key' ], kind: 'Hash' }
    }
  }
}

// ------------------------------------------------------------ Container Apps
resource newEnv 'Microsoft.App/managedEnvironments@2024-03-01' = if (empty(containerAppsEnvironmentId)) {
  name: 'cae-dhm-${suffix}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

var envId = empty(containerAppsEnvironmentId) ? newEnv.id : containerAppsEnvironmentId
var openAiEndpoint = 'https://${foundryName}.openai.azure.com/'

var commonSecrets = [
  { name: 'cosmos-key', value: cosmos.listKeys().primaryMasterKey }
  { name: 'aoai-key', value: foundry.listKeys().key1 }
  { name: 'appi-conn', value: appInsights.properties.ConnectionString }
]

var commonEnv = [
  { name: 'COSMOS_ENDPOINT', value: cosmos.properties.documentEndpoint }
  { name: 'COSMOS_KEY', secretRef: 'cosmos-key' }
  { name: 'AZURE_OPENAI_ENDPOINT', value: openAiEndpoint }
  { name: 'AZURE_OPENAI_API_KEY', secretRef: 'aoai-key' }
  { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT', value: model.name }
  { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appi-conn' }
  { name: 'HOMES_SYNC_SECONDS', value: '300' }
]

// Web app: UI + API + home-search agent (search_homes, check_affordability, check_golden_visa,
// estimate_commute, community_profile). Hot-swaps homes the Market Data Agent publishes.
resource web 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: envId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
      secrets: commonSecrets
    }
    template: {
      containers: [
        {
          name: appName
          image: image
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: commonEnv
          probes: [
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 2 }
    }
  }
}

// Daily job: Market Data Agent on Foundry Agent Service (recall_memory, fetch_new_deals,
// validate_batch, append_deals, rebuild_homes, remember, market_brief).
resource job 'Microsoft.App/jobs@2024-03-01' = {
  name: jobName
  location: location
  identity: { type: 'SystemAssigned' } // used to call Foundry (Entra ID, no key)
  properties: {
    environmentId: envId
    configuration: {
      triggerType: 'Schedule'
      scheduleTriggerConfig: {
        cronExpression: cronExpression
        parallelism: 1
        replicaCompletionCount: 1
      }
      replicaTimeout: 3600
      replicaRetryLimit: 1
      secrets: commonSecrets
    }
    template: {
      containers: [
        {
          name: jobName
          image: image
          command: [ 'python' ]
          args: [ '/app/scripts/data_agent.py' ]
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: concat(commonEnv, [
            { name: 'PYTHONPATH', value: '/app' }
            { name: 'FOUNDRY_PROJECT_ENDPOINT', value: project.properties.endpoints['AI Foundry API'] }
            { name: 'FOUNDRY_MODEL_DEPLOYMENT', value: model.name }
            { name: 'DLD_FILE_URLS', value: dldFileUrls }
          ])
        }
      ]
    }
  }
}

resource jobFoundryUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(project.id, job.id, foundryUserRoleId)
  scope: project
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', foundryUserRoleId)
    principalId: job.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output webUrl string = 'https://${web.properties.configuration.ingress.fqdn}'
output foundryProjectEndpoint string = project.properties.endpoints['AI Foundry API']
output foundryPortal string = 'https://ai.azure.com'
output cosmosEndpoint string = cosmos.properties.documentEndpoint
