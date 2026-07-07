from rag import gap_agent


def make_result(text, metadata):
    return {
        "text": text,
        "metadata": metadata,
        "source": metadata.get("file_path", ""),
        "page": metadata.get("page"),
    }


def test_analyze_gap_searches_specs_and_meetings_and_returns_expected_dict(monkeypatch):
    raw_results = [
        make_result(
            "official coverage",
            {
                "doc_type": "official_spec",
                "status": "official",
                "title": "TS 33.501",
                "related_spec": "TS 33.501",
                "file_path": "/spec.txt",
                "page": 10,
            },
        ),
        make_result(
            "proposal discussion",
            {
                "doc_type": "meeting_doc",
                "status": "proposed",
                "title": "TDoc proposal",
                "source_company": "Example Corp",
                "meeting_id": "SA3_123",
                "tdoc_id": "S3-230001",
                "meeting_date": "2026-01-15",
                "related_spec": "TS 33.501",
                "file_path": "/tdoc.txt",
                "page": 2,
            },
        ),
    ]
    calls = {}

    def fake_hybrid_search(topic, search_specs, search_meetings, k_vector, k_keyword):
        calls["search"] = (topic, search_specs, search_meetings, k_vector, k_keyword)
        return raw_results

    def fake_call_local_llm(prompt, system_prompt=None):
        calls["prompt"] = prompt
        calls["system_prompt"] = system_prompt
        return "\n".join(
            [
                "1. Officially covered areas: Covered [Evidence 1]",
                "2. Meeting-discussed areas: Proposed discussion [Evidence 2]",
                "3. Potential gaps: Model inference: possible gap [Evidence 1]",
                "4. Strong contribution angles: Angle [Evidence 2]",
                "5. Weak or missing evidence: Missing detail [Evidence 1]",
                "6. Recommended next documents to inspect: Inspect more TDocs",
                "7. Evidence map: Evidence 1 official, Evidence 2 meeting",
            ]
        )

    monkeypatch.setattr(gap_agent, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(gap_agent.llm, "call_local_llm", fake_call_local_llm)

    output = gap_agent.analyze_gap("AKMA privacy")

    assert calls["search"] == ("AKMA privacy", True, True, 10, 10)
    assert output["topic"] == "AKMA privacy"
    assert "Officially covered areas" in output["gap_analysis"]
    assert "Meeting-discussed areas" in output["gap_analysis"]
    assert "Potential gaps" in output["gap_analysis"]
    assert "Strong contribution angles" in output["gap_analysis"]
    assert "Weak or missing evidence" in output["gap_analysis"]
    assert "Recommended next documents to inspect" in output["gap_analysis"]
    assert "Evidence map" in output["gap_analysis"]
    assert "doc_type: official_spec" in output["evidence"]
    assert "doc_type: meeting_doc" in output["evidence"]
    assert "status: proposed" in output["evidence"]
    assert output["raw_results"]


def test_gap_prompt_enforces_standards_safety_rules(monkeypatch):
    monkeypatch.setattr(
        gap_agent,
        "hybrid_search",
        lambda *args, **kwargs: [
            make_result(
                "proposal",
                {
                    "doc_type": "meeting_doc",
                    "status": "proposed",
                    "title": "Proposal",
                    "file_path": "/tdoc.txt",
                },
            )
        ],
    )
    captured = {}
    monkeypatch.setattr(
        gap_agent.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: captured.setdefault(
            "values", (prompt, system_prompt)
        )
        and "analysis [Evidence 1]",
    )

    gap_agent.analyze_gap("topic")

    prompt, system_prompt = captured["values"]
    assert "Do not invent requirements" in prompt
    assert "Do not treat proposals as approved" in prompt
    assert "Mark model inferences clearly" in prompt
    assert "Do not invent requirements" in system_prompt
    assert "Never treat proposed" in system_prompt
    assert "Mark model inferences clearly" in system_prompt


def test_analyze_gap_reports_empty_topic_without_search(monkeypatch):
    called = {"search": False}
    monkeypatch.setattr(
        gap_agent,
        "hybrid_search",
        lambda *args, **kwargs: called.__setitem__("search", True),
    )

    output = gap_agent.analyze_gap("   ")

    assert called["search"] is False
    assert output["topic"] == "   "
    assert "Evidence is insufficient" in output["gap_analysis"]
    assert output["evidence"] == ""
    assert output["raw_results"] == []


def test_analyze_gap_reports_no_retrieved_evidence_without_llm(monkeypatch):
    called = {"llm": False}
    monkeypatch.setattr(gap_agent, "hybrid_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        gap_agent.llm,
        "call_local_llm",
        lambda *args, **kwargs: called.__setitem__("llm", True),
    )

    output = gap_agent.analyze_gap("unknown gap")

    assert called["llm"] is False
    assert output["topic"] == "unknown gap"
    assert "Evidence is insufficient" in output["gap_analysis"]
    assert output["raw_results"] == []
