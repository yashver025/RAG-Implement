from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag import (
    load_pdf,
    split_documents,
    get_embeddings,
    create_vectorstore,
    retrieve_context,
    generate_answer
)

st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 ChatDoc")

# -----------------------------
# Cache heavy resources
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_cached_embeddings():
    return get_embeddings()

# -----------------------------
# Session State
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

# -----------------------------
# Process Document (SAFE)
# -----------------------------
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing document..."):
        docs = load_pdf(uploaded_file)
        chunks = split_documents(docs)

        # embeddings load ONCE per app session
        st.session_state.embeddings = load_cached_embeddings()

        st.session_state.vectorstore = create_vectorstore(
            chunks,
            st.session_state.embeddings
        )

        st.success("Document processed successfully!")

# -----------------------------
# Query
# -----------------------------
query = st.text_input("Ask a question about the document")

if query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload a document first.")
    else:
        with st.spinner("Generating answer..."):
            context = retrieve_context(
                st.session_state.vectorstore,
                query
            )
            answer = generate_answer(context, query)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Retrieved Context"):
                st.write(context)
