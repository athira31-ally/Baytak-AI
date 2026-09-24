"""MCP server: tool listing and calls over the in-process API, and the /mcp HTTP endpoint."""
import asyncio
import json

from app import mcp_server


def _call(name, args):
    res = asyncio.run(mcp_server.mcp.call_tool(name, args))
    blocks = res[0] if isinstance(res, tuple) else res
    return json.loads(blocks[0].text)


def test_tools_are_listed(client):
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"search_homes", "check_affordability", "check_golden_visa", "estimate_commute",
            "community_profile", "market_brief", "ask_baytak"} <= names


def test_search_and_ask_via_mcp(client):
    out = _call("search_homes", {"purpose": "sale", "budget_aed": 2_000_000, "min_bedrooms": 2})
    assert out["results"] and all(r["bedrooms"] >= 2 for r in out["results"])
    ans = _call("ask_baytak", {"question": "2 bed in Dubai Marina under 2M, golden visa"})
    assert ans["grounded"] and ans["homes"] and "search" in ans["agents"]


def test_mcp_http_endpoint(client):
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1"}}}
    r = client.post("/mcp/", json=init, headers=headers)
    assert r.status_code == 200 and r.json()["result"]["serverInfo"]["name"] == "Baytak AI"
    r = client.post("/mcp/", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, headers=headers)
    assert "search_homes" in {t["name"] for t in r.json()["result"]["tools"]}
