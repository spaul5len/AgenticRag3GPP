"""Query routing for source-aware SA3 RAG retrieval."""

from __future__ import annotations

import json
import re
from typing import Any

from rag import llm


VALID_INTENTS = {"ask", "gap", "draft", "timeline"}


def route_query(question: str) -> dict[str, Any]:
    """Route a user question to spec and/or meeting retrieval."""

    fallback = _heuristic_route(question, reason_prefix="heuristic fallback")
    if not question.strip():
        return fallback

    prompt = _router_prompt(question)
    try:
        raw = llm.call_local_llm(prompt, system_prompt=_router_system_prompt())
        parsed = _parse_route_json(raw)
        return _normalize_route(parsed, question, fallback)
    except Exception as exc:
        fallback["reason"] = f"{fallback['reason']}; router fallback after LLM/JSON error: {exc}"
        return fallback


def _router_system_prompt() -> str:
    return (
        "You route questions for a local 3GPP SA3 RAG system. "
        "Return only valid JSON. Do not include markdown."
    )


def _router_prompt(question: str) -> str:
    return f"""
Classify this question for retrieval over official 3GPP specs and SA3 meeting/TDoc documents.

Routing rules:
- official requirements, definitions, clauses -> specs
- recent, meeting, TDoc, proposal, discussed -> meetings
- gap, draft, contribution, status -> both
- timeline/history/evolution -> meetings

Return exactly this JSON object shape:
{{
  "search_specs": true,
  "search_meetings": true,
  "intent": "ask",
  "reason": "short reason",
  "suggested_queries": ["query 1", "query 2"]
}}

Allowed intent values: ask, gap, draft, timeline.
Question: {question}
""".strip()


def _parse_route_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))

    if not isinstance(value, dict):
        raise ValueError("Router response was not a JSON object.")
    return value


def _normalize_route(
    route: dict[str, Any], question: str, fallback: dict[str, Any]
) -> dict[str, Any]:
    search_specs = route.get("search_specs")
    search_meetings = route.get("search_meetings")
    intent = str(route.get("intent") or fallback["intent"]).strip().lower()
    reason = str(route.get("reason") or fallback["reason"]).strip()
    suggested_queries = route.get("suggested_queries")

    normalized = {
        "search_specs": (
            search_specs if isinstance(search_specs, bool) else fallback["search_specs"]
        ),
        "search_meetings": (
            search_meetings
            if isinstance(search_meetings, bool)
            else fallback["search_meetings"]
        ),
        "intent": intent if intent in VALID_INTENTS else fallback["intent"],
        "reason": reason,
        "suggested_queries": _normalize_suggested_queries(suggested_queries, question),
    }

    if not normalized["search_specs"] and not normalized["search_meetings"]:
        normalized["search_specs"] = fallback["search_specs"]
        normalized["search_meetings"] = fallback["search_meetings"]
    return normalized


def _heuristic_route(question: str, reason_prefix: str = "heuristic route") -> dict[str, Any]:
    tokens = _tokens(question)
    search_specs = bool(
        tokens
        & {
            "shall",
            "must",
            "requirement",
            "requirements",
            "definition",
            "definitions",
            "clause",
            "clauses",
            "specified",
            "spec",
            "standard",
            "normative",
        }
    )
    search_meetings = bool(
        tokens
        & {
            "recent",
            "meeting",
            "meetings",
            "tdoc",
            "tdocs",
            "proposal",
            "proposals",
            "proposed",
            "discussed",
            "discussion",
            "minutes",
        }
    )

    intent = "ask"
    if tokens & {"timeline", "history", "evolution"}:
        intent = "timeline"
        search_meetings = True
    elif tokens & {"draft", "contribution"}:
        intent = "draft"
        search_specs = True
        search_meetings = True
    elif tokens & {"gap", "gaps", "status"}:
        intent = "gap"
        search_specs = True
        search_meetings = True

    if not search_specs and not search_meetings:
        search_specs = True
        search_meetings = True

    return {
        "search_specs": search_specs,
        "search_meetings": search_meetings,
        "intent": intent,
        "reason": f"{reason_prefix} based on routing keywords",
        "suggested_queries": _default_suggested_queries(question, intent),
    }


def _default_suggested_queries(question: str, intent: str) -> list[str]:
    cleaned = question.strip()
    if not cleaned:
        return []
    if intent == "timeline":
        return [cleaned, f"history evolution {cleaned}"]
    if intent == "draft":
        return [cleaned, f"official requirements and meeting proposals {cleaned}"]
    if intent == "gap":
        return [cleaned, f"official requirements meeting discussion gaps {cleaned}"]
    return [cleaned]


def _normalize_suggested_queries(value: Any, question: str) -> list[str]:
    if not isinstance(value, list):
        return _default_suggested_queries(question, "ask")
    queries = [str(item).strip() for item in value if str(item).strip()]
    return queries[:5] or _default_suggested_queries(question, "ask")


def _tokens(question: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_.-]+", question)}
