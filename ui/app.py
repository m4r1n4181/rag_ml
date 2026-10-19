import html
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from src.config import settings  # noqa: E402
from src.generation.rag_chain import ask_stream  # noqa: E402
from src.retrieval.vector_store import get_embedder, load_vector_store  # noqa: E402

EXAMPLE_QUESTIONS = [
    "Objasniti pojam regularizacije u kontekstu linearne regresije i na koje načine se ona primenjuje.",
    "Objasniti ulogu parametra brzine učenja kod gradijentnog silaska.",
    "Kako se vrši izbor modela kod stabla odluke ukoliko se koristi orezivanje?",
    "Kako radi transformer arhitektura?",
]

st.set_page_config(page_title="Mašinsko učenje 1 asistent", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .source-card {
        background-color: #1e1e2e;
        border-left: 3px solid #7c3aed;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .rank-badge {
        background-color: #7c3aed;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-left: 8px;
    }
    .stChatMessage { padding: 8px 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Učitavam model i bazu...")
def load_resources():
    # Shared by all sessions: the embedding model is ~1 GB and should load once per process.
    get_embedder()
    return load_vector_store()


collection = load_resources()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("sources", [])

questions_asked = sum(m["role"] == "user" for m in st.session_state.messages)
limit_reached = questions_asked >= settings.max_questions_per_session

col_chat, col_sources = st.columns([2, 1])

with col_chat:
    st.title("🎓 Mašinsko učenje 1 asistent")
    st.caption(
        "Postavi pitanje o gradivu predmeta. Odgovori se zasnivaju samo na predavanjima i praktikumu, "
        "a izvori se prikazuju desno."
    )

    clicked = None
    if not st.session_state.messages:
        st.markdown("**Primeri pitanja:**")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                clicked = question

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if limit_reached:
        st.info(f"Ovo je javni demo, ograničen na {settings.max_questions_per_session} pitanja po sesiji.")

    typed = st.chat_input("Postavi pitanje...", disabled=limit_reached)
    query = clicked or typed

    if query and not limit_reached:
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            answer, sources = "", []
            placeholder = st.empty()
            for event, value in ask_stream(query, collection=collection, history=history):
                if event == "query" and value != query:
                    st.caption(f"🔎 Pretraga: {value}")
                elif event == "token":
                    answer += value
                    placeholder.markdown(answer + "▌")
                elif event == "sources":
                    sources = value
            placeholder.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.sources = sources
        st.rerun()

with col_sources:
    st.subheader("📚 Izvori")

    if not st.session_state.sources:
        st.caption("Izvori će se prikazati nakon pitanja.")
    else:
        for i, source in enumerate(st.session_state.sources, 1):
            meta = source["metadata"]
            is_slide = meta.get("type") == "presentation"
            icon = "📊" if is_slide else "📖"
            location = f"Slajd {meta.get('slide', '?')}" if is_slide else f"Strana {meta.get('page', '?')}"
            name = meta.get("lecture") or meta.get("source", "?")
            name = name[:40] + "..." if len(name) > 40 else name

            st.markdown(
                f'<div class="source-card">{icon} <strong>{html.escape(name)}</strong>'
                f'<span class="rank-badge">#{i}</span><br><small>{location}</small></div>',
                unsafe_allow_html=True,
            )

            image_path = meta.get("image_path")
            if is_slide and image_path and Path(image_path).exists():
                st.image(image_path, caption=f"{meta.get('source')}, slajd {meta.get('slide')}", use_column_width=True)

            with st.expander("Prikaži tekst"):
                st.caption(source["content"][:400] + "...")

        st.divider()
        if st.button("🗑️ Očisti razgovor"):
            st.session_state.messages = []
            st.session_state.sources = []
            st.rerun()
