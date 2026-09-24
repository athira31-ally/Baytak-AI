"""Open-source / on-prem LLM path: LLM_PROVIDER=openai_compatible talks to any OpenAI-compatible server
(Ollama, vLLM, NVIDIA NIM). A tiny local HTTP server plays the model, so the whole LangGraph team is
exercised over real HTTP with the same client code that talks to Ollama."""
import http.server
import json
import re
import threading

from app.agents.graph import GraphAgent
from app.agents.llm import chat_model, llm_client
from app.config import Settings
from app.schemas import ChatRequest

ID = re.compile(r"(?:DHM-\d{5}|DLD-[0-9A-F]{8})")


class FakeOllama(http.server.BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        msg = {"role": "assistant", "content": ""}
        choice = body.get("tool_choice")
        if isinstance(choice, dict):                         # forced tool, e.g. the supervisor's Brief
            name = choice["function"]["name"]
            args = {"purpose": "sale", "budget_aed": 2000000, "min_bedrooms": 2, "specialists": ["visa"]} \
                if name == "Brief" else {"purpose": "sale"}
            msg["tool_calls"] = [{"id": "call_1", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}]
        else:                                                # specialists / writer: answer in text
            ids = ID.findall(json.dumps(body["messages"][1:]))       # skip the system prompt
            msg["content"] = f"Recommended: [{ids[0]}]." if ids else "Done."
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_factories_pick_the_openai_compatible_server():
    s = Settings(llm_provider="openai_compatible", llm_base_url="http://localhost:11434/v1", llm_model="qwen2.5:7b-instruct",
                 azure_openai_endpoint="")
    assert s.use_llm and not s.use_azure_openai and s.llm_label == "open-source:qwen2.5:7b-instruct"
    assert str(llm_client(s).base_url).startswith("http://localhost:11434/v1")
    m = chat_model(s)
    assert type(m).__name__ == "ChatOpenAI" and m.model_name == "qwen2.5:7b-instruct"


def test_langgraph_team_runs_on_an_open_source_model_server(client):
    from app.main import state
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = state["rec"].s.model_copy(update={"llm_provider": "openai_compatible", "azure_openai_endpoint": "",
                                              "llm_base_url": f"http://127.0.0.1:{srv.server_address[1]}/v1",
                                              "llm_model": "qwen2.5:7b-instruct"})
        r = GraphAgent(state["rec"], s).chat(ChatRequest(message="2 bed apartment under 2M with golden visa"))
    finally:
        srv.shutdown()
    assert r.mode == "open-source:qwen2.5:7b-instruct" and r.engine == "langgraph"
    assert "visa" in r.agents and r.grounded and ID.search(r.answer)
    assert all(req["model"] == "qwen2.5:7b-instruct" for req in FakeOllama.requests)
    assert not any("reasoning_effort" in req for req in FakeOllama.requests)   # not sent to open models


def test_azure_reasoning_model_gets_fast_and_normal_effort():
    s = Settings(azure_openai_endpoint="https://x.openai.azure.com/", azure_openai_api_key="k",
                 azure_openai_chat_deployment="chat", azure_openai_chat_model="gpt-5-mini")
    assert chat_model(s).reasoning_effort == "low"
    assert chat_model(s, effort=s.llm_fast_reasoning_effort).reasoning_effort == "minimal"
    assert chat_model(s).temperature is None                      # reasoning models reject temperature
