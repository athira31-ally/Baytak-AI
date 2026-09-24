"""Input guardrail for the home-search agent.

1. Azure AI Content Safety **Prompt Shields** (when CONTENT_SAFETY_ENDPOINT is set): detects
   jailbreak / prompt-injection attempts. The Foundry (AI Services) resource already exposes it,
   so no extra Azure resource is needed.
2. A local heuristic check that always runs (and is the only check offline): obvious injection
   phrases and abusive length. It is deliberately conservative - it should never block a normal
   property question.

The guard fails OPEN on service errors (logs and continues) so an outage can't take the app down;
the grounding check and tool-only numbers still protect the answer."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import httpx

from app.config import Settings

log = logging.getLogger(__name__)
MAX_CHARS = 4000
INJECTION = re.compile(
    r"ignore (all |the |your )?(previous|prior|above) (instructions|rules|prompt)|"
    r"disregard (the |your )?(system|previous) (prompt|instructions)|"
    r"(reveal|print|show) (me )?(your|the) (system prompt|instructions|hidden prompt)|"
    r"you are now (dan|developer mode)|jailbreak",
    re.I)


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    checks: list[str] = field(default_factory=list)


def _prompt_shields(settings: Settings, text: str) -> bool | None:
    """True = attack detected, False = clean, None = service not configured / unavailable."""
    if not settings.content_safety_endpoint:
        return None
    url = settings.content_safety_endpoint.rstrip("/") + "/contentsafety/text:shieldPrompt?api-version=2024-09-01"
    headers = {"Content-Type": "application/json"}
    if settings.content_safety_key:
        headers["Ocp-Apim-Subscription-Key"] = settings.content_safety_key
    else:
        from azure.identity import DefaultAzureCredential
        headers["Authorization"] = "Bearer " + DefaultAzureCredential().get_token(
            "https://cognitiveservices.azure.com/.default").token
    try:
        r = httpx.post(url, headers=headers, json={"userPrompt": text[:10000], "documents": []}, timeout=10)
        r.raise_for_status()
        return bool(r.json().get("userPromptAnalysis", {}).get("attackDetected"))
    except Exception as e:                                   # fail open, but say so
        log.warning("Prompt Shields unavailable, continuing with local checks: %s", e)
        return None


def check_input(settings: Settings, text: str) -> GuardResult:
    checks = ["local-heuristics"]
    if len(text) > MAX_CHARS:
        return GuardResult(False, f"Message too long ({len(text)} characters, max {MAX_CHARS}).", checks)
    if INJECTION.search(text):
        return GuardResult(False, "Looks like an attempt to override the assistant's instructions.", checks)
    shield = _prompt_shields(settings, text)
    if shield is not None:
        checks.append("azure-prompt-shields")
        if shield:
            return GuardResult(False, "Azure AI Content Safety flagged a prompt-injection / jailbreak attempt.", checks)
    return GuardResult(True, "", checks)


REFUSAL = ("I can only help with finding homes in Dubai - searching listings, affordability, the Golden Visa, "
           "commutes and communities. Please rephrase your question.")
