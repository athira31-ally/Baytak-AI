"""Market Data Agent on Microsoft Foundry Agent Service - against a fake Foundry project."""
import json
from types import SimpleNamespace as NS

import pandas as pd

from app.data.foundry_agent import FoundryMarketDataAgent
from tests.test_data_agent import env, history, run, server  # noqa: F401  (fixtures)


class FakeFoundry:
    """Stands in for AIProjectClient: agents.create_version + an OpenAI client with the Responses API.
    The fake 'agent' tries to append before validating (must be refused), then behaves."""

    def __init__(self):
        self.versions, self.requests, self.step = [], [], 0
        self.agents = NS(create_version=self._create_version)

    def _create_version(self, agent_name, definition, **kw):
        self.versions.append({"name": agent_name, "definition": definition, **kw})
        return NS(version=str(len(self.versions)), id=f"{agent_name}:{len(self.versions)}")

    def get_openai_client(self):
        return NS(conversations=NS(create=lambda **kw: NS(id="conv_1")),
                  responses=NS(create=self._respond))

    def _respond(self, conversation, input, extra_body):
        self.requests.append({"conversation": conversation, "input": input, "extra_body": extra_body})
        plan = ["recall_memory", "fetch_new_deals", "append_deals", "validate_batch", "append_deals",
                "rebuild_homes", "market_brief"]
        if self.step < len(plan):
            self.step += 1
            call = NS(type="function_call", name=plan[self.step - 1], arguments="{}", call_id=f"call_{self.step}")
            return NS(output=[call], output_text="")
        return NS(output=[NS(type="message")], output_text="APPENDED - report from the Foundry agent")


def test_foundry_agent_runs_tools_and_is_reused(server, env):
    root, base = server
    history().to_csv(root / "tx.csv", index=False)
    s, store, cache = env
    s = s.model_copy(update={"foundry_project_endpoint": "https://x.services.ai.azure.com/api/projects/p"})
    from app.data.agent import DataTools
    fake = FakeFoundry()
    r = FoundryMarketDataAgent(s, DataTools(s, store, cache, [f"{base}/tx.csv"]), project=fake).run()

    assert r["mode"] == "foundry-agent" and r["published"] and r["outcome"] == "APPENDED"
    assert r["foundry"] == {"agent": "baytak-market-data-agent", "version": "1", "conversation_id": "conv_1"}
    tools = [t.name for t in fake.versions[0]["definition"].tools]
    assert {"fetch_new_deals", "validate_batch", "append_deals", "rebuild_homes"} <= set(tools)
    # every call references the Foundry agent and stays in one conversation
    assert all(q["extra_body"]["agent_reference"]["name"] == "baytak-market-data-agent" for q in fake.requests)
    assert {q["conversation"] for q in fake.requests} == {"conv_1"}
    # tool results go back as function_call_output; the premature append was refused in code
    first_append = next(t for t in r["trace"] if t["tool"] == "append_deals")
    assert '"appended": false' in first_append["result"]
    outputs = [i for q in fake.requests if isinstance(q["input"], list) for i in q["input"]]
    assert outputs and all(o["type"] == "function_call_output" for o in outputs)

    # second run: same prompt/tools -> no new agent version
    fake2 = FakeFoundry()
    FoundryMarketDataAgent(s, DataTools(s, store, cache, [f"{base}/tx.csv"]), project=fake2).run()
    assert fake2.versions == []
