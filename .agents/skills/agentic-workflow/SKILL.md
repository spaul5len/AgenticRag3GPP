---
name: agentic-workflow
description: Use this skill when adding or modifying agentic RAG behavior, including router agents, answer agents, gap analysis, drafting, verifier agents, timeline agents, query refinement, or controlled loops.
---

You are implementing agentic workflows for a local 3GPP SA3 RAG system.

Main agent types:
- Router Agent
- Retriever Agent
- Answer Agent
- Gap Analysis Agent
- Drafting Agent
- Verifier Agent
- Timeline Agent
- Figure Agent

Agentic behavior means:
- classify the user task
- choose the correct source type
- retrieve evidence
- generate an evidence-grounded output
- verify the generated output
- revise or retry when needed

Rules:
- Keep max loops small by default, usually 2.
- Do not create infinite loops.
- Stop looping if no new evidence is found.
- Always preserve source metadata.
- Verifier must check unsupported claims, wrong acronym expansion, and confusion between official specs and meeting proposals.
- answer_question(question) must remain backward compatible.
- New agentic behavior should be opt-in, for example answer_question(..., agentic=True).

For 3GPP standardization:
- Official specs are approved source material.
- Meeting/TDoc documents are proposals or discussions unless metadata says approved/agreed.
- Gap analysis must use cautious wording:
  "The retrieved evidence does not explicitly show..."
  not:
  "3GPP has no..."
