"""
02_rag_pipeline.py
==================
BeanBot RAG — Retrieval & Generation Pipeline

This script is Step 2 of the BeanBot RAG system.

It provides a single public function `ask_beanbot(question, chat_history)`
that the Streamlit app (03_app.py) will call.

Internals:
  1. Loads the persistent ChromaDB vector store created by 01_ingest_knowledge.py.
  2. Initializes local HuggingFace embeddings (same model used during ingestion).
  3. Builds a LangChain retrieval chain using Gemini 1.5 Flash as the LLM.
  4. Applies a strict Thai/English bilingual system prompt (no hallucination).
  5. Returns the answer text AND the source document chunks for UI display.

Usage (standalone test):
  python 02_rag_pipeline.py

Dependencies:
  - GOOGLE_API_KEY must be set in a .env file or as an environment variable.
  - Run 01_ingest_knowledge.py first to populate the ChromaDB.
"""

import os
import sys
import logging
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

# --- LangChain ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# --- Embeddings ---
from langchain_huggingface import HuggingFaceEmbeddings

# --- Vector Store ---
from langchain_chroma import Chroma

# --- LLM ---
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).parent.resolve()
CHROMA_DB_DIR = REPO_ROOT / "data" / "chroma_db"
CHROMA_COLLECTION_NAME = "beanbot_knowledge"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Gemini model
# gemini-2.5-flash-lite = newest, smallest, highest free-tier token budget
# and has a SEPARATE quota pool from gemini-2.0-* models
LLM_MODEL_NAME = "gemini-2.5-flash-lite"
LLM_TEMPERATURE = 0.1   # Low temperature = more factual, less creative

# Chunks to retrieve — reduced from 5 to 3 to cut input token count
# (fewer tokens per request = less likely to hit per-minute quota)
RETRIEVAL_TOP_K = 3

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("beanbot.pipeline")

# ============================================================
# SYSTEM PROMPT
# ============================================================

# The core Thai system prompt from the spec, with enhanced instructions added.
SYSTEM_PROMPT = """คุณคือ 'BeanBot AI Assistant' ที่เชี่ยวชาญโปรเจกต์เครื่องคัดแยกเมล็ดกาแฟอัตโนมัติ \
หน้าที่ของคุณคือตอบคำถามรุ่นน้องหรือคนอื่นๆที่สนใจโดยอิงจากเอกสารปริญญานิพนธ์และ Paper วิจัยที่ให้มาเท่านั้น

กฎการตอบ:
1. หากคำถามเกี่ยวกับโค้ด วงจร Arduino หรือโมเดล YOLO ให้ตอบแบบ Step-by-step
2. ระบุค่าพารามิเตอร์ให้ชัดเจนหากมีในเอกสาร
3. หากไม่มีข้อมูลในเอกสาร ห้ามเดาเด็ดขาด ให้ตอบว่า 'ในเล่มโครงงานไม่ได้ระบุส่วนนี้ไว้ แนะนำให้ลองทดสอบที่หน้างานจริง'
4. หากคำถามเป็นภาษาอังกฤษ ให้ตอบเป็นภาษาอังกฤษ หากเป็นภาษาไทย ให้ตอบเป็นภาษาไทย
5. อ้างอิงชื่อไฟล์หรือส่วนของเอกสารที่ใช้ตอบเสมอ เช่น (จาก: ปริญญานิพนธ์ บทที่ 3) หรือ (Source: args_v12s.yaml)
6. สำหรับโค้ด ให้แสดงใน code block พร้อม syntax ที่ถูกต้อง

เอกสารอ้างอิงที่ดึงมา (Context):
{context}
"""

# ============================================================
# RESPONSE TYPE
# ============================================================

class BotResponse(TypedDict):
    answer: str
    source_documents: list[Document]

# ============================================================
# PIPELINE INITIALIZATION (cached at module level)
# ============================================================

_vector_store: Chroma | None = None
_retriever = None
_llm: ChatGoogleGenerativeAI | None = None


def _load_env():
    """Load GOOGLE_API_KEY from .env file if present."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log.error(
            "GOOGLE_API_KEY not found!\n"
            "  1. Copy .env.example to .env\n"
            "  2. Set your key: GOOGLE_API_KEY=your_key_here\n"
            "  Get a free key at: https://aistudio.google.com/apikey"
        )
        sys.exit(1)
    return api_key


def _init_vector_store() -> Chroma:
    """Load the persisted ChromaDB vector store."""
    if not CHROMA_DB_DIR.exists():
        log.error(
            f"ChromaDB not found at: {CHROMA_DB_DIR}\n"
            "  Please run 01_ingest_knowledge.py first."
        )
        sys.exit(1)

    log.info(f"Loading ChromaDB from: {CHROMA_DB_DIR}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )
    count = vector_store._collection.count()
    log.info(f"✓ Loaded ChromaDB — {count} chunks available")
    return vector_store


def _init_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Initialize the Gemini LLM."""
    log.info(f"Initializing LLM: {LLM_MODEL_NAME}")
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=1024,  # Reduced from 2048 to lower token pressure
    )
    log.info("✓ LLM ready")
    return llm


def initialize_pipeline():
    """
    Initialize all pipeline components (embeddings, ChromaDB, LLM).
    This is called once at startup and cached in module-level globals.
    """
    global _vector_store, _retriever, _llm

    if _vector_store is not None:
        return  # Already initialized

    log.info("=" * 50)
    log.info("  BeanBot RAG Pipeline — Initializing")
    log.info("=" * 50)

    api_key = _load_env()
    _vector_store = _init_vector_store()
    _retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_TOP_K},
    )
    _llm = _init_llm(api_key)

    log.info("✅ Pipeline ready — BeanBot is online!")


# ============================================================
# CORE RAG FUNCTION
# ============================================================

def _format_docs(docs: list[Document]) -> str:
    """
    Format retrieved source chunks into a readable context block for the LLM.
    Each chunk is prefixed with its source file name for attribution.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name", "unknown")
        parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_history_messages(chat_history: list[dict]) -> list:
    """
    Convert the Streamlit-style chat history list
    [{"role": "user"|"assistant", "content": "..."}]
    into LangChain message objects for the conversation prompt.
    """
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def ask_beanbot(question: str, chat_history: list[dict] | None = None) -> BotResponse:
    """
    Main RAG function — retrieves relevant context and generates an answer.

    Args:
        question:     The user's question (Thai or English).
        chat_history: List of previous messages in the format
                      [{"role": "user"|"assistant", "content": "..."}].
                      Pass None or [] for a fresh conversation.

    Returns:
        A BotResponse dict with:
          - "answer":           The LLM's generated answer string.
          - "source_documents": List of Document objects used as context.
    """
    if chat_history is None:
        chat_history = []

    # Ensure pipeline is initialized
    initialize_pipeline()

    # --- Step 1: Retrieve relevant chunks ---
    source_docs: list[Document] = _retriever.invoke(question)
    context_text = _format_docs(source_docs)

    # --- Step 2: Build the prompt ---
    # We use a ChatPromptTemplate that includes:
    #   - System message (Thai instructions + retrieved context)
    #   - Conversation history (for follow-up questions)
    #   - The current human question
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    # --- Step 3: Build the chain ---
    chain = prompt | _llm | StrOutputParser()

    # --- Step 4: Invoke ---
    history_messages = _build_history_messages(chat_history)
    answer = chain.invoke({
        "context": context_text,
        "history": history_messages,
        "question": question,
    })

    return BotResponse(
        answer=answer,
        source_documents=source_docs,
    )


# ============================================================
# STANDALONE TEST
# ============================================================

def _run_test():
    """Quick sanity-check: ask a few sample questions and print results."""
    test_questions = [
        "What YOLO model is used in this project and how many epochs was it trained?",
    ]

    print("\n" + "=" * 60)
    print("  BeanBot RAG — Standalone Test")
    print("=" * 60)

    history = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─'*60}")
        print(f"Q{i}: {question}")
        print("─" * 60)

        response = ask_beanbot(question, chat_history=history)

        print(f"Answer:\n{response['answer']}")
        print(f"\nSources used ({len(response['source_documents'])} chunks):")
        for doc in response["source_documents"]:
            src = doc.metadata.get("file_name", "unknown")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  • {src}: \"{preview}...\"")

        # Add to history for conversational follow-up
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response["answer"]})

    print("\n" + "=" * 60)
    print("  ✅ Test complete — Run 03_app.py for the full UI")
    print("=" * 60)


if __name__ == "__main__":
    _run_test()
