using './main.bicep'

param suffix = '24680'
param location = 'uaenorth'
param foundryLocation = 'swedencentral'
param cosmosLocation = 'swedencentral'
param cosmosName = 'cosmos-dhm-24680-se'
param image = 'ghcr.io/athira31-ally/baytak-ai:latest'

// Your subscription allows one Container Apps environment per region; reuse the existing one.
// Get it with:  az containerapp env show -n trakheesi-env -g rg-trakheesi-demo --query id -o tsv
// Leave '' to create a new environment (e.g. after deleting rg-trakheesi-demo).
param containerAppsEnvironmentId = '/subscriptions/1d9604b6-3956-4570-a521-a2409f91c759/resourceGroups/rg-trakheesi-demo/providers/Microsoft.App/managedEnvironments/trakheesi-env'

// Paste the two data.dubai CSV links here (comma-separated) once you have them.
param dldFileUrls = ''
