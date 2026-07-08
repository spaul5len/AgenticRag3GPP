---
name: performance-profiler
description: Use this skill when asked to improve latency, benchmark models, reduce Ollama timeouts, profile RAG steps, or optimize local performance.
---

Before optimizing, measure first.

Profile these stages:
- parsing time
- chunking time
- embedding time
- Chroma vector search
- BM25 search
- result merging/reranking
- evidence formatting
- prompt size
- LLM generation time
- total wall time

Use the existing profiler when available:

PYTHONPATH=. .venv/bin/python scripts/profile_rag_query.py

Report:
- model name
- evidence chars
- prompt chars
- retrieved result count
- used chunk count
- hybrid_search time
- llm_generation time
- total wall time

Optimization rules:
- Do not guess the bottleneck.
- Compare before and after timings.
- Prefer small safe changes:
  - reduce max_chunks
  - reduce max_chars_per_chunk
  - reduce num_predict
  - improve retrieval quality
  - add query expansion
- Keep answer quality in mind; the fastest model may be too weak for 3GPP reasoning.

Recommended local debug settings:
- max_chunks = 2 or 3
- max_chars_per_chunk = 700 or 800
- num_predict = 160 to 256
- qwen2.5:0.5b only for UI smoke tests
- qwen2.5:1.5b or llama3.2:3b for better local answers
