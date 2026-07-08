---
name: rag-pipeline-engineer
description: Use this skill when modifying the local 3GPP SA3 RAG pipeline, including parsing, chunking, embeddings, Chroma, BM25, SQLite metadata, query expansion, reranking, or evidence formatting.
---

You are working on a local 3GPP SA3 Agentic RAG project.

Focus areas:
- document parsing
- chunking
- embedding generation
- Chroma indexing
- BM25 retrieval
- SQLite metadata
- hybrid retrieval
- evidence formatting
- acronym-aware query expansion
- exact-term reranking

Project rules:
- Keep official specifications and meeting/TDoc documents clearly separated.
- Do not treat meeting proposals as approved requirements unless metadata says approved or agreed.
- Preserve source metadata in every retrieved result.
- Do not invent clause numbers, TDoc IDs, company names, or meeting decisions.
- Keep runtime local-first with Ollama, Chroma, SQLite, and BM25.
- Keep backward compatibility for existing public functions such as answer_question(question).

3GPP acronym guardrails:
- SBA = Service-Based Architecture
- AUSF = Authentication Server Function
- UDM = Unified Data Management
- SEAF = Security Anchor Function
- AMF = Access and Mobility Management Function
- SUPI = Subscription Permanent Identifier
- SUCI = Subscription Concealed Identifier
- NRF = Network Repository Function

When changing retrieval:
1. Add or update tests.
2. Run targeted tests.
3. Run:
   PYTHONPATH=. .venv/bin/pytest
4. Run:
   .venv/bin/python -m compileall -q rag app_fastapi.py app_streamlit.py scripts
