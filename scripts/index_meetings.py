#!/usr/bin/env python
"""CLI entrypoint for 3GPP SA3 meeting document ingestion."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingest_meetings import main


if __name__ == "__main__":
    raise SystemExit(main())
