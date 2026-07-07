import app_streamlit


def test_run_mode_ask_uses_answer_question(monkeypatch):
    monkeypatch.setattr(app_streamlit, "answer_question", lambda prompt: "answer")

    message = app_streamlit.run_mode("question", {"mode": "Ask"})

    assert message == {"role": "assistant", "content": "answer", "kind": "answer"}


def test_run_mode_gap_returns_gap_package(monkeypatch):
    package = {
        "topic": "AKMA",
        "gap_analysis": "gap",
        "evidence": "Evidence 1",
        "raw_results": [],
    }
    monkeypatch.setattr(app_streamlit, "analyze_gap", lambda prompt: package)

    message = app_streamlit.run_mode("AKMA", {"mode": "Gap Analysis"})

    assert message["kind"] == "gap"
    assert message["content"] == "gap"
    assert message["package"] == package


def test_run_mode_draft_uses_verifier_by_default(monkeypatch):
    calls = {}

    def fake_draft_verify_revise(topic):
        calls["topic"] = topic
        return {
            "topic": topic,
            "draft_type": "discussion contribution",
            "draft": "initial",
            "review": "review",
            "revised_draft": "revised",
            "gap_package": {"evidence": "Evidence 1"},
        }

    monkeypatch.setattr(app_streamlit, "draft_verify_revise", fake_draft_verify_revise)

    message = app_streamlit.run_mode(
        "AKMA",
        {"mode": "Draft Contribution", "run_verifier": True},
    )

    assert calls["topic"] == "AKMA"
    assert message["kind"] == "draft"
    assert message["content"] == "revised"
    assert message["package"]["review"] == "review"


def test_run_mode_draft_can_skip_verifier(monkeypatch):
    calls = {}

    def fake_draft(topic):
        calls["topic"] = topic
        return {
            "topic": topic,
            "draft_type": "discussion contribution",
            "draft": "initial",
            "gap_package": {"evidence": "Evidence 1"},
        }

    monkeypatch.setattr(app_streamlit, "draft_sa3_contribution", fake_draft)

    message = app_streamlit.run_mode(
        "AKMA",
        {"mode": "Draft Contribution", "run_verifier": False},
    )

    assert calls["topic"] == "AKMA"
    assert message["content"] == "initial"
    assert message["package"]["review"] == ""
    assert message["package"]["revised_draft"] == "initial"


def test_run_mode_timeline_uses_timeline_agent(monkeypatch):
    monkeypatch.setattr(app_streamlit, "build_topic_timeline", lambda prompt: "timeline")

    message = app_streamlit.run_mode("AKMA", {"mode": "Timeline"})

    assert message == {"role": "assistant", "content": "timeline", "kind": "timeline"}


def test_run_mode_returns_safe_error_message(monkeypatch):
    def fail(prompt):
        raise RuntimeError("failed reading data/specs/private.pdf")

    monkeypatch.setattr(app_streamlit, "answer_question", fail)

    message = app_streamlit.run_mode("question", {"mode": "Ask"})

    assert message["kind"] == "error"
    assert "data/specs/private.pdf" not in message["content"]
    assert "[redacted-data-path]" in message["content"]


def test_input_placeholders_cover_modes():
    assert "Ask" in app_streamlit._input_placeholder("Ask")
    assert "gap" in app_streamlit._input_placeholder("Gap Analysis").lower()
    assert "draft" in app_streamlit._input_placeholder("Draft Contribution").lower()
    assert "timeline" in app_streamlit._input_placeholder("Timeline").lower()
