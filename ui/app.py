import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.generation.rag_chain import ask_stream
from src.retrieval.vector_store import load_vector_store

# ------------------------------------------------------------------ #
# Streamlit config
# st.set_page_config mora biti PRVI Streamlit poziv u fajlu —
# ako ga staviš posle bilo kog st.* poziva, dobijaš grešku.
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Faculty RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------------------------------------------ #
# CSS — minimalno stilizovanje da izgleda pristojno
# ------------------------------------------------------------------ #
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
    .similarity-badge {
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


# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "vector_store" not in st.session_state:
    with st.spinner("Učitavam knowledge base..."):
        st.session_state.vector_store = load_vector_store()


# ------------------------------------------------------------------ #
# Layout — dva stupca: chat (levo) i sources (desno)
# ------------------------------------------------------------------ #
col_chat, col_sources = st.columns([2, 1])

with col_chat:
    st.title("🎓 Mašinsko učenje 1 FTN asistent")
    st.caption("Postavi pitanje o gradivu — odgovaram na osnovu praktikuma i predavanja.")

    # Prikaži chat istoriju
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # st.markdown renderuje LaTeX ako je okružen $$ ili $
            st.markdown(msg["content"])

    # Input polje — st.chat_input se automatski fiksira na dno
    if query := st.chat_input("Postavi pitanje..."):

        # Prikaži korisnikovu poruku
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generiši i stream-uj odgovor
        with st.chat_message("assistant"):
            full_response = ""
            sources = []

            # st.write_stream ne možemo koristiti direktno jer naš
            # generator vraća tuple (tip, vrednost).
            # Koristimo placeholder koji updateujemo token po token.
            placeholder = st.empty()

            for event_type, value in ask_stream(
                query,
                collection=st.session_state.vector_store
            ):
                if event_type == "token":
                    full_response += value
                    # Pišemo u placeholder — korisnik vidi streaming
                    placeholder.markdown(full_response + "▌")

                elif event_type == "sources":
                    sources = value

            # Ukloni kursor, prikaži finalni odgovor
            placeholder.markdown(full_response)

        # Sačuvaj u istoriju
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
        st.session_state.sources = sources

        # Rerun da bi sources kolona bila ažurna
        st.rerun()


with col_sources:
    st.subheader("📚 Izvori")

    if not st.session_state.sources:
        st.caption("Izvori će se prikazati nakon pitanja.")
    else:
        for i, source in enumerate(st.session_state.sources, 1):
            meta = source["metadata"]
            similarity = source["similarity"]
            source_name = meta.get("source", "?")
            doc_type = meta.get("type", "?")

            if doc_type == "presentation":
                icon = "📊"
                location = f"Slajd {meta.get('slide', '?')}"
            else:
                icon = "📖"
                location = f"Strana {meta.get('page', '?')}"

            # Skrati naziv fajla ako je predugačak
            display_name = source_name[:35] + "..." if len(source_name) > 35 else source_name

            st.markdown(f"""
<div class="source-card">
    {icon} <strong>{display_name}</strong>
    <span class="similarity-badge">{similarity}</span><br>
    <small>{location}</small>
</div>
""", unsafe_allow_html=True)
            # Ako postoji slika slajda, prikaži je
            image_path = meta.get("image_path")
            if image_path and doc_type == "presentation":
                st.image(image_path, caption=f"{source_name} – slajd {meta.get('slide', '?')}", use_column_width=True)

            # Expander sa preview tekstom
            with st.expander("Prikaži tekst"):
                st.caption(source["content"][:400] + "...")

        # Dugme za čišćenje istorije
        st.divider()
        if st.button("🗑️ Očisti razgovor"):
            st.session_state.messages = []
            st.session_state.sources = []
            st.rerun()