# ☕ BeanBot RAG

**BeanBot** is an AI Knowledge Assistant designed to help engineering students learn about an automated coffee bean sorting system. It uses **Retrieval-Augmented Generation (RAG)** to answer questions based on a Thai thesis, an English research paper, Arduino code, Python YOLO tracking scripts, and training configurations.

## 🌟 Features

- **Bilingual Support**: Understands and replies in both Thai and English.
- **Multi-format Knowledge Ingestion**:
  - `docling` for handling complex Thai PDFs, tables, and academic formats.
  - LangChain splitters for Markdown, Python, C++ (Arduino), and YAML.
- **Local Embeddings**: Uses `all-MiniLM-L6-v2` locally via HuggingFace to embed chunks without external API costs.
- **Persistent Vector Store**: Uses ChromaDB to store and retrieve knowledge chunks efficiently.
- **Modern UI**: A sleek, coffee-themed Streamlit chat interface that shows conversation history and source citations.

## 🛠️ Tech Stack

- **Orchestration**: LangChain (`langchain`, `langchain-core`)
- **LLM**: Google Gemini 2.5 Flash Lite (via `langchain-google-genai`)
- **Embeddings**: `sentence-transformers`
- **Vector Database**: ChromaDB
- **Document Parsing**: `docling`, `unstructured`
- **UI**: Streamlit

## 📁 Project Structure

```text
beanbot_rag/
├── data/
│   ├── raw_docs/          # Thai thesis, English papers, diagrams (PDFs)
│   ├── python_code/       # YOLO detection code
│   ├── adruino_code/      # Hardware controller code
│   └── train_configs/     # Dataset and training YAMLs
├── 01_ingest_knowledge.py # Pipeline to parse, chunk, and embed documents
├── 02_rag_pipeline.py     # RAG retrieval logic & LLM connection
├── 03_app.py              # Streamlit Chat UI
├── rag_pipeline.py        # Shim for clean imports
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (API Keys)
```

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed.

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/yourusername/beanbot_rag.git
cd beanbot_rag
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. API Key Setup

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy the example `.env` file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and paste your `GOOGLE_API_KEY`.

### 4. Build the Knowledge Base

Run the ingestion script to parse all documents in the `data/` folder, embed them, and save them to a local ChromaDB instance:

```bash
python 01_ingest_knowledge.py
```
*(Note: The first run will download necessary models for `docling` and HuggingFace embeddings. It might take a few minutes.)*

### 5. Run the Chatbot

Start the Streamlit application:

```bash
streamlit run 03_app.py
```

Open your browser to `http://localhost:8501` to chat with BeanBot!

## 📝 License

This project is open-source and available for educational purposes.
