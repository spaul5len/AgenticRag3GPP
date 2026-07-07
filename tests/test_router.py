from rag import router


def test_route_query_parses_local_llm_json(monkeypatch):
    monkeypatch.setattr(
        router.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: """
        {
          "search_specs": true,
          "search_meetings": false,
          "intent": "ask",
          "reason": "official clause question",
          "suggested_queries": ["TS 33.501 authentication clause"]
        }
        """,
    )

    route = router.route_query("What does TS 33.501 require?")

    assert route == {
        "search_specs": True,
        "search_meetings": False,
        "intent": "ask",
        "reason": "official clause question",
        "suggested_queries": ["TS 33.501 authentication clause"],
    }


def test_route_query_uses_heuristic_fallback_when_json_fails(monkeypatch):
    monkeypatch.setattr(
        router.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: "not json",
    )

    route = router.route_query("What recent TDoc proposals discussed this gap?")

    assert route["search_specs"] is True
    assert route["search_meetings"] is True
    assert route["intent"] == "gap"
    assert "fallback" in route["reason"]
    assert route["suggested_queries"]


def test_heuristic_routes_timeline_to_meetings(monkeypatch):
    monkeypatch.setattr(
        router.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: "not json",
    )

    route = router.route_query("Show the timeline and evolution of SA3 discussion")

    assert route["search_specs"] is False
    assert route["search_meetings"] is True
    assert route["intent"] == "timeline"


def test_heuristic_routes_official_requirement_to_specs(monkeypatch):
    monkeypatch.setattr(
        router.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: "not json",
    )

    route = router.route_query("What shall the UE do according to the clause?")

    assert route["search_specs"] is True
    assert route["search_meetings"] is False
    assert route["intent"] == "ask"
