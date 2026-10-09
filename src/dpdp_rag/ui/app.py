"""Streamlit entry point: `uv run streamlit run src/dpdp_rag/ui/app.py`."""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="DPDP RAG", page_icon=":material/gavel:", layout="wide")
here = Path(__file__).parent
st.navigation(
    [
        st.Page(here / "chat_page.py", title="Chat", icon=":material/chat:", default=True),
        st.Page(here / "dashboard_page.py", title="Metrics", icon=":material/monitoring:"),
    ]
).run()
