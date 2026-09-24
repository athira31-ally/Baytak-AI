"""Run the Market Data Agent once - this is what the daily cron (Container Apps Job) runs.

    python -m scripts.data_agent                          # append deals since the last run
    python -m scripts.data_agent --files a.csv b.csv      # local files or URLs instead of discovery

Data + memory: Cosmos DB when COSMOS_ENDPOINT is set, else artifacts/market.db (SQLite).
Cache:         Redis when REDIS_URL is set, else in-process.
Exit code 0 = appended or nothing new; 1 = batch rejected or source unreachable (the job shows
as failed in Azure so you notice).
"""
import argparse
import sys

from app.config import get_settings
from app.data.agent import DataTools, MarketDataAgent
from app.observability import setup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", help="CSV paths or URLs (default: discover on data.dubai)")
    a = ap.parse_args()
    s = get_settings()
    setup(s)
    tools = DataTools(s, file_urls=a.files)
    if s.foundry_project_endpoint:                  # the agent lives in Microsoft Foundry
        from app.data.foundry_agent import FoundryMarketDataAgent
        result = FoundryMarketDataAgent(s, tools).run()
    else:                                           # plain Azure OpenAI function calling
        result = MarketDataAgent(s, tools).run()
    print(result["report"])
    print(f"\noutcome={result['outcome']} mode={result['mode']} watermark={result['watermark']} "
          f"new_deals={result['new_deals']} seconds={result['seconds']}")
    if result.get("foundry"):
        f = result["foundry"]
        print(f"foundry agent={f['agent']} version={f['version']} conversation={f['conversation_id']}")
    for t in result["trace"]:
        print(f"  {t['tool']:<16} {t['seconds']:>6}s  {t['result'][:150]}")
    sys.exit(0 if result["outcome"] in ("APPENDED", "REFRESHED", "NO NEW DATA") else 1)


if __name__ == "__main__":
    main()
