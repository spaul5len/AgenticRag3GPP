"""Draft verification and revision agent."""

from __future__ import annotations

from typing import Any

from rag import llm
from rag.drafting_agent import draft_sa3_contribution


def verify_draft(draft: str, evidence: str) -> str:
    """Review a draft against supplied evidence and SA3 safety rules."""

    review = llm.call_local_llm(
        _verify_prompt(draft, evidence),
        system_prompt=_verify_system_prompt(),
    ).strip()
    if not review:
        return "Verification failed because the local model returned an empty review."
    return review


def revise_draft(draft: str, review: str, evidence: str) -> str:
    """Revise a draft according to a verification review and evidence."""

    revised = llm.call_local_llm(
        _revise_prompt(draft, review, evidence),
        system_prompt=_revise_system_prompt(),
    ).strip()
    if not revised:
        return (
            "Revision failed because the local model returned an empty draft. "
            "Use the review to revise manually."
        )
    return revised


def draft_verify_revise(
    topic: str, draft_type: str = "discussion contribution"
) -> dict[str, Any]:
    """Draft a contribution, verify it, and produce a cautious revision."""

    draft_package = draft_sa3_contribution(topic, draft_type=draft_type)
    evidence = str(draft_package.get("gap_package", {}).get("evidence", ""))
    draft = str(draft_package.get("draft", ""))
    review = verify_draft(draft, evidence)
    revised_draft = revise_draft(draft, review, evidence)

    return {
        "topic": draft_package.get("topic", topic.strip()),
        "draft_type": draft_package.get("draft_type", draft_type),
        "draft": draft,
        "review": review,
        "revised_draft": revised_draft,
        "gap_package": draft_package.get("gap_package", {}),
    }


def _verify_system_prompt() -> str:
    return """
You are a strict 3GPP SA3 contribution verifier.
Use only the supplied evidence.
Do not infer support from absent evidence.
Do not approve unsupported clause numbers, TDoc IDs, company names, company positions, requirements, statuses, or dates.
Meeting documents, including approved/agreed ones, remain meeting_doc evidence and are not official specifications.
Proposed, unknown, noted, withdrawn, or draft meeting material must not be treated as approved standard text.
""".strip()


def _verify_prompt(draft: str, evidence: str) -> str:
    return f"""
Draft to verify:
{draft}

Evidence:
{evidence}

Review the draft against the evidence. Include exactly these checks:
1. Unsupported claims
2. Invented clause numbers
3. Invented TDoc IDs or company positions
4. Treating meeting proposals as approved
5. Overuse of "shall"
6. Missing distinction between facts, gaps, proposals, and inference
7. Weak technical motivation
8. Missing evidence

For each check, state PASS/FAIL/UNCLEAR, explain briefly, and cite evidence as [Evidence X] when applicable.
Do not invent new facts during verification.
""".strip()


def _revise_system_prompt() -> str:
    return """
You revise SA3-style contribution drafts using only the supplied review and evidence.
Remove unsupported claims.
Make language cautious.
Preserve useful technical content that is supported by evidence.
Do not add new unsupported facts, clause numbers, TDoc IDs, companies, company positions, requirements, statuses, or dates.
Do not treat meeting documents as approved standard text.
Use "shall" only for proposed normative text or verified official text explicitly supported by evidence.
""".strip()


def _revise_prompt(draft: str, review: str, evidence: str) -> str:
    return f"""
Original draft:
{draft}

Verification review:
{review}

Evidence:
{evidence}

Revise the draft.

Revision requirements:
- Remove unsupported claims.
- Make language cautious.
- Preserve useful technical content that is supported by evidence.
- Do not add new unsupported facts.
- Keep facts, gaps, proposals, and model inference clearly separated.
- Keep meeting proposals labeled as proposals/discussions, not approved standard text.
- Keep or add [Evidence X] citations for supported claims.
""".strip()
