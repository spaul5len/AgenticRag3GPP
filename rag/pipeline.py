"""End-to-end question answering pipeline for local SA3 RAG."""

from __future__ import annotations

from typing import Any

from rag import llm
from rag.retriever import (
    deduplicate_results,
    format_evidence,
    hybrid_search,
    sort_by_source_quality,
)
from rag.router import route_query


def answer_question(question: str) -> str:
    """Route, retrieve evidence, and answer with explicit evidence references."""

    if not question.strip():
        return "Evidence is insufficient to answer because the question is empty."

    route = route_query(question)
    results: list[dict[str, Any]] = []
    for query in route.get("suggested_queries") or [question]:
        results.extend(
            hybrid_search(
                query,
                search_specs=bool(route.get("search_specs", True)),
                search_meetings=bool(route.get("search_meetings", True)),
            )
        )

    results = sort_by_source_quality(deduplicate_results(results))
    if not results:
        return (
            "Evidence is insufficient to answer this question. "
            "No matching official specification or meeting evidence was retrieved."
        )

    evidence = format_evidence(results)
    prompt = _answer_prompt(question, route, evidence)
    system_prompt = _answer_system_prompt()
    answer = llm.call_local_llm(prompt, system_prompt=system_prompt).strip()
    if not answer:
        return (
            "Evidence is insufficient to answer this question. "
            "The local model returned an empty answer."
        )
    return answer


def _answer_system_prompt() -> str:
    return """
You are a source-aware 3GPP SA3 research assistant.
Use only the supplied evidence.
Every factual claim must cite evidence as [Evidence X].
Separate official specification facts from meeting discussions.
Never treat proposed, unknown, noted, or withdrawn meeting documents as approved standard text.
Approved or agreed meeting documents remain meeting_doc evidence, not official specifications.
Do not invent clause numbers, TDoc IDs, company names, statuses, meeting IDs, dates, or requirements.
If the evidence is insufficient, say so plainly.
""".strip()


def _answer_prompt(question: str, route: dict[str, Any], evidence: str) -> str:
    return f"""
Question:
{question}

Route:
intent: {route.get("intent")}
search_specs: {route.get("search_specs")}
search_meetings: {route.get("search_meetings")}
reason: {route.get("reason")}

Evidence:
{evidence}

Answer requirements:
- Include [Evidence X] references for claims.
- Use separate sections for Official specification facts and Meeting discussions when both source types appear.
- Keep proposals and unknown-status meeting documents clearly labeled as meeting discussions, not approved standards.
- State when evidence is insufficient for part of the question.
""".strip()
