---
name: figure-retrieval
description: Use this skill when implementing retrieval or display of exact figures, diagrams, images, captions, or tables from 3GPP specs and meeting documents.
---

Goal:
Retrieve and display exact original figures from 3GPP documents, not generated approximations.

Rules:
- Extract embedded images from DOCX first.
- Save extracted images under data/figures/.
- Do not commit extracted 3GPP figures.
- Store figure metadata:
  - figure_id
  - document_id
  - figure_number
  - caption
  - image_path
  - source_file
  - page/order
  - clause
  - surrounding_text
  - doc_type
  - status

Indexing:
- Add a separate Chroma collection for figures, for example FIGURE_COLLECTION.
- Embed searchable figure text:
  document title, figure number, caption, clause, and surrounding text.
- Store image_path and source metadata in Chroma metadata.

Retrieval:
- Add search_figures(query, k=3).
- Return image_path, figure_number, caption, source_file, clause, page/order.

Streamlit:
- Display retrieved figures with st.image().
- Show caption and metadata.
- Warn if image_path does not exist.

Repository hygiene:
- Ensure data/figures/ is ignored by git.
