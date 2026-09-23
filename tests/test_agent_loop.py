"""Exercise the Azure OpenAI function-calling loop with a fake client (no network)."""
import json
from types import SimpleNamespace as NS

from app.agents.orchestrator import Agent
from app.schemas import ChatRequest


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        if self.calls == 1:  # step 1: model asks for a search
            tc = NS(id="call_1", type="function", function=NS(name="search_homes", arguments=json.dumps(
                {"purpose": "sale", "budget_aed": 2_000_000, "min_bedrooms": 2, "work_location": "DIFC"})))
            return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))])
        # step 2: model answers, citing one real ID from the tool result and one invented ID
        tool_msg = next(m for m in kw["messages"] if m["role"] == "tool")
        real_id = json.loads(tool_msg["content"])["results"][0]["listing_id"]
        return NS(choices=[NS(message=NS(content=f"Try [{real_id}] or [DHM-99999].", tool_calls=None))])


def test_llm_loop_and_grounding_check(client):
    from app.main import state
    agent = Agent(state["rec"], state["rec"].s)
    agent.client = NS(chat=NS(completions=FakeCompletions()))
    r = agent.chat(ChatRequest(message="2 bed near DIFC, 2M"))
    assert r.mode == "azure-openai"
    assert [t.tool for t in r.tool_trace] == ["search_homes"]
    assert not r.grounded and r.ungrounded_ids == ["DHM-99999"]   # hallucinated ID is caught


class ReasoningModelCompletions(FakeCompletions):
    """Mimics gpt-5 / o-series: any custom temperature is a 400 error."""
    def create(self, **kw):
        if "temperature" in kw:
            raise ValueError("Unsupported value: 'temperature' does not support 0.2 with this model.")
        return super().create(**kw)


def test_reasoning_model_without_temperature(client):
    from app.main import state
    agent = Agent(state["rec"], state["rec"].s)
    agent.client = NS(chat=NS(completions=ReasoningModelCompletions()))
    r = agent.chat(ChatRequest(message="2 bed near DIFC, 2M"))
    assert r.mode == "azure-openai" and not agent.temperature_ok
