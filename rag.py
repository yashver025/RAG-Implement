import tempfile
from typing import List

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage


# -----------------------------
# Load PDF
# -----------------------------
def load_pdf(uploaded_file) -> List[Document]:
    """Load uploaded PDF into LangChain Documents."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    return loader.load()


# -----------------------------
# Split Documents
# -----------------------------
def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# -----------------------------
# Embeddings (load once)
# -----------------------------
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# Vector Store
# -----------------------------
def create_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Retrieve Context
# -----------------------------
def retrieve_context(
    vectorstore: FAISS,
    query: str,
    k: int = 4,
) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------
# Generate Answer
# -----------------------------



def generate_answer(context: str, query: str) -> str:
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.7,
        max_new_tokens=300,
    )

    chat_model = ChatHuggingFace(llm=llm)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not in the context, say 'I don't know'."
            )
        ),
        HumanMessage(
            content=f"""
Context:
{context}

Question:
{query}
"""
        ),
    ]

    response = chat_model.invoke(messages)
    return response.content

