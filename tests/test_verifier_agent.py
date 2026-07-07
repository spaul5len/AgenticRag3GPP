from rag import verifier_agent


def test_verify_draft_checks_required_failure_modes(monkeypatch):
    captured = {}

    def fake_call_local_llm(prompt, system_prompt=None):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return "\n".join(
            [
                "1. Unsupported claims: FAIL",
                "2. Invented clause numbers: PASS",
                "3. Invented TDoc IDs or company positions: PASS",
                "4. Treating meeting proposals as approved: FAIL",
                '5. Overuse of "shall": UNCLEAR',
                "6. Missing distinction between facts, gaps, proposals, and inference: FAIL",
                "7. Weak technical motivation: UNCLEAR",
                "8. Missing evidence: FAIL",
            ]
        )

    monkeypatch.setattr(verifier_agent.llm, "call_local_llm", fake_call_local_llm)

    review = verifier_agent.verify_draft("draft", "Evidence 1")

    assert "Unsupported claims" in review
    assert "1. Unsupported claims" in captured["prompt"]
    assert "2. Invented clause numbers" in captured["prompt"]
    assert "3. Invented TDoc IDs or company positions" in captured["prompt"]
    assert "4. Treating meeting proposals as approved" in captured["prompt"]
    assert '5. Overuse of "shall"' in captured["prompt"]
    assert "6. Missing distinction between facts, gaps, proposals, and inference" in captured["prompt"]
    assert "7. Weak technical motivation" in captured["prompt"]
    assert "8. Missing evidence" in captured["prompt"]
    assert "Meeting documents" in captured["system_prompt"]
    assert "must not be treated as approved standard text" in captured["system_prompt"]


def test_revise_draft_prompt_enforces_revision_rules(monkeypatch):
    captured = {}

    def fake_call_local_llm(prompt, system_prompt=None):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return "Revised cautious draft [Evidence 1]"

    monkeypatch.setattr(verifier_agent.llm, "call_local_llm", fake_call_local_llm)

    revised = verifier_agent.revise_draft(
        "Original unsupported draft",
        "Review says remove unsupported claims",
        "Evidence 1",
    )

    assert revised == "Revised cautious draft [Evidence 1]"
    assert "Remove unsupported claims" in captured["prompt"]
    assert "Make language cautious" in captured["prompt"]
    assert "Preserve useful technical content" in captured["prompt"]
    assert "Do not add new unsupported facts" in captured["prompt"]
    assert "facts, gaps, proposals, and model inference" in captured["prompt"]
    assert "Remove unsupported claims" in captured["system_prompt"]
    assert "Do not add new unsupported facts" in captured["system_prompt"]


def test_draft_verify_revise_runs_full_flow(monkeypatch):
    draft_package = {
        "topic": "AKMA privacy",
        "draft_type": "discussion contribution",
        "draft": "Initial draft [Evidence 1]",
        "gap_package": {
            "topic": "AKMA privacy",
            "gap_analysis": "gap",
            "evidence": "Evidence 1\ndoc_type: official_spec\nstatus: official",
            "raw_results": [],
        },
    }
    calls = {}

    def fake_draft(topic, draft_type="discussion contribution"):
        calls["draft"] = (topic, draft_type)
        return draft_package

    def fake_verify(draft, evidence):
        calls["verify"] = (draft, evidence)
        return "review"

    def fake_revise(draft, review, evidence):
        calls["revise"] = (draft, review, evidence)
        return "revised draft"

    monkeypatch.setattr(verifier_agent, "draft_sa3_contribution", fake_draft)
    monkeypatch.setattr(verifier_agent, "verify_draft", fake_verify)
    monkeypatch.setattr(verifier_agent, "revise_draft", fake_revise)

    output = verifier_agent.draft_verify_revise("AKMA privacy")

    assert calls["draft"] == ("AKMA privacy", "discussion contribution")
    assert calls["verify"] == (
        "Initial draft [Evidence 1]",
        "Evidence 1\ndoc_type: official_spec\nstatus: official",
    )
    assert calls["revise"] == (
        "Initial draft [Evidence 1]",
        "review",
        "Evidence 1\ndoc_type: official_spec\nstatus: official",
    )
    assert output == {
        "topic": "AKMA privacy",
        "draft_type": "discussion contribution",
        "draft": "Initial draft [Evidence 1]",
        "review": "review",
        "revised_draft": "revised draft",
        "gap_package": draft_package["gap_package"],
    }


def test_empty_llm_outputs_return_clear_failures(monkeypatch):
    monkeypatch.setattr(
        verifier_agent.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: "   ",
    )

    assert "empty review" in verifier_agent.verify_draft("draft", "evidence")
    assert "empty draft" in verifier_agent.revise_draft("draft", "review", "evidence")
