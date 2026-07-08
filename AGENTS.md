# Local 3GPP SA3 Agentic RAG - Codex Instructions

## Project purpose

This repository implements a local-first Agentic RAG system for 3GPP SA3 and telecom-security research.

The system ingests:
- official 3GPP specifications
- SA3 meeting documents
- TDocs
- metadata

It supports:
- evidence-grounded Q&A
- gap analysis
- SA3-style drafting
- verification
- timeline exploration
- future figure retrieval

## Runtime stack

- Ollama for local chat LLM
- Ollama embedding model
- Chroma vector database
- SQLite metadata
- BM25 keyword search
- Streamlit GUI
- optional FastAPI backend

The runtime RAG system must not require the OpenAI API.

## Source-trust rules

Official specifications and meeting documents must remain separated.

Official specs are approved sources.
Meeting/TDoc documents are proposals or discussions unless metadata explicitly says approved or agreed.

Never invent:
- clause numbers
- TDoc IDs
- company names
- meeting decisions
- normative requirements

Generated output must separate:
- official fact
- meeting discussion
- proposal
- inferred gap
- model inference

## 3GPP acronym rules

Use these meanings unless evidence clearly says otherwise:

- SBA = Service-Based Architecture
- NF = Network Function
- NRF = Network Repository Function
- AUSF = Authentication Server Function
- UDM = Unified Data Management
- SEAF = Security Anchor Function
- AMF = Access and Mobility Management Function
- SUPI = Subscription Permanent Identifier
- SUCI = Subscription Concealed Identifier

Do not invent acronym expansions.

## Commands

Use these commands from the repository root:

Install:
pip install -r requirements.txt

Index specs:
.venv/bin/python scripts/index_specs.py

Index meetings:
.venv/bin/python scripts/index_meetings.py

Run Streamlit:
HOME=/tmp .venv/bin/streamlit run app_streamlit.py

Run FastAPI:
.venv/bin/uvicorn app_fastapi:app --reload

Run tests:
PYTHONPATH=. .venv/bin/pytest

Compile check:
.venv/bin/python -m compileall -q rag app_fastapi.py app_streamlit.py scripts

Profile RAG:
PYTHONPATH=. .venv/bin/python scripts/profile_rag_query.py

## Git hygiene

Do not commit:
- data/
- chroma_db/
- metadata.sqlite
- .venv/
- __pycache__/
- .pytest_cache/
- .env

Before committing, run:
git status --short
