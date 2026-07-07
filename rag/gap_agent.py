"""Standards gap-analysis agent for SA3 research topics."""

from __future__ import annotations

from typing import Any

from rag import llm
from rag.retriever import (
    deduplicate_results,
    format_evidence,
    hybrid_search,
    sort_by_source_quality,
)


def analyze_gap(topic: str) -> dict[str, Any]:
    """Analyze a possible standards gap using official and meeting evidence."""

    cleaned_topic = topic.strip()
    if not cleaned_topic:
        return {
            "topic": topic,
            "gap_analysis": "Evidence is insufficient because the topic is empty.",
            "evidence": "",
            "raw_results": [],
        }

    raw_results = hybrid_search(
        cleaned_topic,
        search_specs=True,
        search_meetings=True,
        k_vector=10,
        k_keyword=10,
    )
    raw_results = sort_by_source_quality(deduplicate_results(raw_results))
    evidence = format_evidence(raw_results)

    if not raw_results:
        return {
            "topic": cleaned_topic,
            "gap_analysis": (
                "Evidence is insufficient to analyze this gap. No matching official "
                "specification or meeting evidence was retrieved."
            ),
            "evidence": evidence,
            "raw_results": raw_results,
        }

    gap_analysis = llm.call_local_llm(
        _gap_prompt(cleaned_topic, evidence),
        system_prompt=_gap_system_prompt(),
    ).strip()
    if not gap_analysis:
        gap_analysis = "Evidence is insufficient because the local model returned an empty analysis."

    return {
        "topic": cleaned_topic,
        "gap_analysis": gap_analysis,
        "evidence": evidence,
        "raw_results": raw_results,
    }


def _gap_system_prompt() -> str:
    return """
You are a strict 3GPP SA3 standards gap-analysis assistant.
Use only the supplied evidence.
Every factual claim must cite evidence as [Evidence X].
Separate official specification facts from meeting discussions.
Never treat proposed, unknown, noted, withdrawn, or draft meeting material as approved standard text.
Approved or agreed meeting documents remain meeting_doc evidence, not official specifications.
Do not invent requirements, clause numbers, TDoc IDs, company names, meeting IDs, dates, or statuses.
Mark model inferences clearly with "Model inference:".
If evidence is weak or missing, say so explicitly.
""".strip()


def _gap_prompt(topic: str, evidence: str) -> str:
    return f"""
Topic:
{topic}

Evidence:
{evidence}

Write a standards gap analysis with exactly these sections:
1. Officially covered areas
2. Meeting-discussed areas
3. Potential gaps
4. Strong contribution angles
5. Weak or missing evidence
6. Recommended next documents to inspect
7. Evidence map

Rules:
- Use only the evidence above.
- Do not invent requirements.
- Do not treat proposals as approved.
- Mark model inferences clearly.
- Cite evidence as [Evidence X] in each section where claims are made.
""".strip()
