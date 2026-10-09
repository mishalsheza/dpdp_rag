"""Chat page: ask questions, see answers with expandable citations and badges."""

from __future__ import annotations

from datetime import date

import streamlit as st

from dpdp_rag.config import load_system_config
from dpdp_rag.ui.render import (
    REFUSAL_LABELS,
    api_url,
    ask,
    badge_markdown,
    citation_badges,
    footer,
    source_line,
    unverified_note,
    version,
)

config = load_system_config()
ui_cfg = config["ui"]
base_url = api_url(ui_cfg)

with st.sidebar:
    as_of = st.date_input(
        "As-of date",
        value=date.today(),
        format="YYYY-MM-DD",
        help="Answers say whether each provision is in force on this date.",
    )
    info = version(base_url)
    if info:
        st.caption(
            f"API `{base_url}` · model `{info['model']}` · config "
            f"`{info['config_hash'][:8]}` · git `{info['git_sha'][:8]}`"
        )
    else:
        st.caption(f"API `{base_url}` is not reachable yet.")
    if st.button("Clear conversation"):
        st.session_state.messages = []

st.title("DPDP Act & Rules assistant")
st.caption(
    "Answers come only from the Digital Personal Data Protection Act, 2023, the DPDP "
    "Rules, 2025 and related notifications. **Not legal advice.**"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_answer(resp: dict, asked_on: date) -> None:
    if resp.get("refused"):
        reason = resp.get("refusal_reason") or ""
        st.warning(
            REFUSAL_LABELS.get(reason, "The assistant declined to answer."), icon=":material/info:"
        )
    st.markdown(resp["answer"])
    if note := unverified_note(resp):
        st.warning(note, icon=":material/warning:")
    citations = resp.get("citations", [])
    if citations:
        st.markdown(f"**Sources ({len(citations)})**")
    for cit in citations:
        badges = citation_badges(cit, asked_on)
        with st.expander(f"{cit['pinpoint']}" + (f" — {cit['title']}" if cit.get("title") else "")):
            if badges:
                st.markdown(badge_markdown(badges))
            st.caption(source_line(cit))
            if cit.get("text"):
                st.markdown(f"> {cit['text']}")
    st.caption(footer(resp))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            st.caption(f"as of {msg['as_of']}")
        else:
            show_answer(msg["content"], date.fromisoformat(msg["as_of"]))

if question := st.chat_input("Ask about the DPDP Act or Rules"):
    st.session_state.messages.append(
        {"role": "user", "content": question, "as_of": as_of.isoformat()}
    )
    with st.chat_message("user"):
        st.markdown(question)
        st.caption(f"as of {as_of.isoformat()}")
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the Act and Rules ..."):
                resp = ask(base_url, question, as_of, float(ui_cfg["request_timeout_s"]))
        except RuntimeError as exc:
            st.error(str(exc), icon=":material/error:")
        else:
            show_answer(resp, as_of)
            st.session_state.messages.append(
                {"role": "assistant", "content": resp, "as_of": as_of.isoformat()}
            )
