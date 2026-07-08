---
name: streamlit-gui
description: Use this skill when modifying the Streamlit GUI, including sidebar controls, source filters, evidence display, figure display, agentic loop controls, or error handling.
---

You are improving the Streamlit GUI for a local 3GPP SA3 Agentic RAG app.

Rules:
- Do not break existing chat behavior.
- Sidebar controls must actually affect the pipeline.
- If the UI has a max_chunks slider, pass it into answer_question().
- If the UI has source filters, pass them into retrieval.
- Show useful errors instead of raw stack traces.
- Keep the interface simple and research-oriented.

Useful sidebar controls:
- Official specs
- Meeting docs
- Max chunks
- Show evidence
- Use agentic loop
- Retrieve figures
- Model/backend information

Evidence display:
- Show evidence IDs.
- Show doc_type, status, title, source_file, page/order, related_spec.
- Make it clear whether evidence came from official specs or meeting documents.

Figure display:
- If figure evidence exists, display actual extracted images using st.image().
- Show figure number, caption, source file, clause, and page/order.
- Warn if the image file is missing.

WSL note:
- Streamlit may print a harmless gio browser-opening warning.
- Users can open http://localhost:8501 manually.
