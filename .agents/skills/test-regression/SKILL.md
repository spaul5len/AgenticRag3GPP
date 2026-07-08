---
name: test-regression
description: Use this skill before finalizing code changes, before commits, or when fixing failures after modifying the local 3GPP SA3 RAG project.
---

For every code change:

1. Run targeted tests first when possible.
2. Run full tests:
   PYTHONPATH=. .venv/bin/pytest

3. Run compile check:
   .venv/bin/python -m compileall -q rag app_fastapi.py app_streamlit.py scripts

4. Do not delete the user's real local data unless explicitly asked.
   Never delete casually:
   - data/
   - chroma_db/
   - metadata.sqlite
   - .venv/

5. If tests fail, fix the root cause.
6. Do not weaken tests unless the test is clearly wrong or outdated.
7. Keep backward compatibility unless the user explicitly asks for a breaking change.

Before commit, check:
git status --short

Do not stage:
- data/
- chroma_db/
- metadata.sqlite
- .venv/
- __pycache__/
- .pytest_cache/
- .env
