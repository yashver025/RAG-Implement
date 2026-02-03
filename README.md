# 📄 ChatDoc — RAG-based PDF Question Answering App

ChatDoc is a **Retrieval-Augmented Generation (RAG)** application built with **Streamlit, LangChain, Hugging Face, and FAISS**.
It allows users to upload a PDF document and ask natural-language questions, with answers generated **only from the document content**.

---

## 🚀 Features

* 📂 Upload and process PDF documents
* ✂️ Chunk documents using Recursive Character Text Splitting
* 🧠 Generate embeddings with Sentence Transformers
* 📦 Store and retrieve vectors using FAISS
* 🤖 Answer questions using Hugging Face LLMs (chat models)
* ⚡ Efficient caching to avoid re-processing and re-embedding
* 🧪 Streamlit-based interactive UI

---

## 🏗️ Architecture (RAG Pipeline)

```
PDF Upload
   ↓
Document Loader (PyPDFLoader)
   ↓
Text Splitter (RecursiveCharacterTextSplitter)
   ↓
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
   ↓
Vector Store (FAISS)
   ↓
Retriever
   ↓
LLM (meta-llama/Llama-3.1-8B-Instruct)
   ↓
Answer
```

---

## 🧰 Tech Stack

* **Frontend:** Streamlit
* **LLM Orchestration:** LangChain
* **Embeddings:** Sentence Transformers
* **Vector Database:** FAISS
* **LLM Provider:** Hugging Face Inference API
* **Language:** Python 3.9+

---

## 📁 Project Structure

```
RAG/
├── app.py              # Streamlit application
├── rag.py              # RAG pipeline logic
├── rag_demo.ipynb      # Notebook for experimentation
├── .env                # Environment variables (not committed)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/chatdoc-rag.git
cd chatdoc-rag
```

---

### 2️⃣ Create and activate virtual environment

```bash
python -m venv myenv
source myenv/bin/activate   # macOS/Linux
myenv\Scripts\activate      # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure environment variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```



---

### 5️⃣ Run the application

```bash
streamlit run app.py
```

Open your browser at:

```
http://localhost:8501
```

---

## 🧪 How It Works

1. Upload a PDF file
2. The document is split into chunks and embedded **once**
3. Embeddings are cached to avoid reloading
4. FAISS retrieves relevant chunks for each query
5. The LLM answers using **only the retrieved context**

---

## 🛑 Limitations

* Answers are limited to the content of the uploaded PDF
* Requires internet access for Hugging Face Inference API
* Large PDFs may take longer to process

---

## 🔮 Future Improvements

* 🔗 Source citations with page numbers
* 💬 Multi-turn conversational memory
* 💾 Persistent FAISS storage
* 🖥️ Local LLM support (Ollama)
* ☁️ Cloud deployment

---

## 👤 Author

**Yash**
IT Engineering Student
AI / ML Enthusiast
