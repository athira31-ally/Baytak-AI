"""The Market Data Agent as a Microsoft Foundry agent (Foundry Agent Service).

What lives where:
  Foundry   the agent itself: name, model (gpt-5-mini), instructions and the 7 function-tool
            definitions, versioned. Every daily run is a Foundry *conversation*, so you can open
            it in the Foundry portal and see each step: model output, tool call, tool result.
  This code the tool implementations (they touch Cosmos, Redis and the DLD files) and the
            guardrails. Foundry function tools are client-executed: Foundry decides WHICH tool
            to call; this code runs it and hands the result back.

Loop (Responses API):  input -> response with function_call items -> run tools here ->
function_call_output items -> ... until the agent answers with text (the daily report).
"""
from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from app.config import Settings
from app.data.agent import SYSTEM_PROMPT, TOOL_SPECS, DataTools, MarketDataAgent

log = logging.getLogger(__name__)
MAX_TURNS = 15


def _function_tools():
    from azure.ai.projects.models import FunctionTool
    return [FunctionTool(name=t["function"]["name"], description=t["function"]["description"],
                         parameters={**t["function"]["parameters"], "additionalProperties": False},
                         strict=False)
            for t in TOOL_SPECS]


def definition_hash(model: str) -> str:
    blob = json.dumps({"model": model, "instructions": SYSTEM_PROMPT, "tools": TOOL_SPECS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


class FoundryMarketDataAgent(MarketDataAgent):
    mode_name = "foundry-agent"

    def __init__(self, settings: Settings, tools: DataTools | None = None, project=None):
        super().__init__(settings, tools, client=False)       # False = don't build a chat client
        if project is None:
            from azure.ai.projects import AIProjectClient
            from azure.identity import DefaultAzureCredential
            project = AIProjectClient(endpoint=settings.foundry_project_endpoint, credential=DefaultAzureCredential())
        self.project = project
        self.client = project                                  # non-None -> run() uses _llm_run
        self.agent_version: str | None = None
        self.conversation_id: str | None = None

    # ----------------------------------------------------------- agent setup
    def ensure_agent(self) -> str:
        """Create a new agent version in Foundry only when the prompt/tools/model changed."""
        from azure.ai.projects.models import PromptAgentDefinition
        h = definition_hash(self.s.foundry_model_deployment)
        mem = self.tools.store.get_memory("foundry_agent") or {}
        if mem.get("hash") == h and mem.get("version"):
            return mem["version"]
        v = self.project.agents.create_version(
            agent_name=self.s.foundry_agent_name,
            definition=PromptAgentDefinition(model=self.s.foundry_model_deployment, instructions=SYSTEM_PROMPT,
                                             tools=_function_tools()),
            description="Appends new Dubai Land Department deals to Cosmos DB daily and republishes Baytak AI homes.",
            metadata={"definition_hash": h, "app": "baytak-ai"},
        )
        self.tools.store.set_memory("foundry_agent", {"hash": h, "version": str(v.version), "id": v.id})
        log.info("Created Foundry agent %s version %s", self.s.foundry_agent_name, v.version)
        return str(v.version)

    # --------------------------------------------------------------- the loop
    def _llm_run(self) -> str:
        self.agent_version = self.ensure_agent()
        ref = {"agent_reference": {"name": self.s.foundry_agent_name, "version": self.agent_version,
                                   "type": "agent_reference"}}
        oai = self.project.get_openai_client()
        conv = oai.conversations.create(metadata={"app": "baytak-ai", "job": "daily-refresh",
                                                  "date": str(pd.Timestamp.today().date())})
        self.conversation_id = conv.id
        inp = f"Run today's refresh. Today is {pd.Timestamp.today().date()}."
        for _ in range(MAX_TURNS):
            resp = oai.responses.create(conversation=conv.id, input=inp, extra_body=ref)
            calls = [o for o in resp.output if getattr(o, "type", None) == "function_call"]
            if not calls:
                return resp.output_text or ""
            inp = []
            for c in calls:
                try:
                    args = json.loads(c.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = self._call(c.name, args)
                inp.append({"type": "function_call_output", "call_id": c.call_id,
                            "output": json.dumps(out, default=str)[:8000]})
        return "Stopped after too many steps."

    def run(self) -> dict:
        result = super().run()
        return {**result, "foundry": {"agent": self.s.foundry_agent_name, "version": self.agent_version,
                                      "conversation_id": self.conversation_id}}
