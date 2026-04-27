"""
01_ingest_knowledge.py
======================
BeanBot RAG — Knowledge Ingestion Pipeline

This script is Step 1 of the BeanBot RAG system.

It performs the following tasks:
  1. Recursively scans all subfolders inside `data/` for supported documents.
  2. Routes each file to the correct loader based on its extension:
       - .pdf   → Parsed by `docling` (handles Thai text, tables, figures)
       - .md    → UnstructuredMarkdownLoader (preserves headings/structure)
       - .py    → Language-aware Python splitter (preserves function boundaries)
       - .ino   → Language-aware C++ splitter (for Arduino code)
       - .yaml  → TextLoader (raw text so LLM sees exact config key/values)
       - .yml   → TextLoader (same as .yaml)
  3. Splits all documents into chunks (size=1200, overlap=200).
  4. Embeds chunks using HuggingFace `all-MiniLM-L6-v2` (runs fully locally, no API needed).
  5. Stores all embeddings in a persistent local ChromaDB at `data/chroma_db/`.

Usage:
  python 01_ingest_knowledge.py

Dependencies:
  See requirements.txt — run `pip install -r requirements.txt` first.

Author: BeanBot Team
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List

# --- LangChain Core ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# --- LangChain Loaders ---
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
)

# --- Embeddings ---
from langchain_huggingface import HuggingFaceEmbeddings

# --- Vector Store ---
from langchain_chroma import Chroma

# ============================================================
# CONFIGURATION
# ============================================================

# Root directories
REPO_ROOT = Path(__file__).parent.resolve()
DATA_DIR = REPO_ROOT / "data"
CHROMA_DB_DIR = REPO_ROOT / "data" / "chroma_db"

# Chunking parameters
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Embedding model (runs locally on CPU — no API key needed)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB collection name
CHROMA_COLLECTION_NAME = "beanbot_knowledge"

# Folders to scan (relative to DATA_DIR)
# We scan everything inside data/ recursively
SCAN_DIRS = [
    DATA_DIR / "raw_docs",
    DATA_DIR / "adruino_code",
    DATA_DIR / "python_code",
    DATA_DIR / "train_configs",
]

# File extensions we handle (others are skipped)
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".py", ".ino", ".yaml", ".yml"}

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("beanbot.ingest")

# ============================================================
# HELPER: DOCLING PDF PARSER
# ============================================================

def load_pdf_with_docling(file_path: Path) -> List[Document]:
    """
    Parse a PDF file using `docling`, which handles:
      - Thai Unicode text (critical for the thesis PDF)
      - Complex academic layouts (multi-column, tables, captions)
      - Figures and diagrams (extracts surrounding text context)

    Returns a list of LangChain Document objects (one per page/section).
    """
    try:
        from docling.document_converter import DocumentConverter

        log.info(f"  [docling] Parsing PDF: {file_path.name} ...")
        converter = DocumentConverter()
        result = converter.convert(str(file_path))

        # Export to Markdown string — docling's Markdown preserves table structure
        markdown_text = result.document.export_to_markdown()

        if not markdown_text.strip():
            log.warning(f"  [docling] Warning: Empty content extracted from {file_path.name}")
            return []

        # Wrap in a single Document with rich metadata
        doc = Document(
            page_content=markdown_text,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "pdf",
                "loader": "docling",
            },
        )
        log.info(f"  [docling] ✓ Extracted {len(markdown_text):,} characters from {file_path.name}")
        return [doc]

    except ImportError:
        log.error("docling is not installed. Run: pip install docling")
        raise
    except Exception as e:
        log.error(f"  [docling] ERROR parsing {file_path.name}: {e}")
        return []


# ============================================================
# HELPER: FILE-TYPE ROUTER
# ============================================================

def load_document(file_path: Path) -> List[Document]:
    """
    Route a file to the appropriate loader based on its extension.

    Returns a list of LangChain Document objects (may be empty on failure).
    """
    ext = file_path.suffix.lower()
    file_str = str(file_path)

    try:
        # --- PDF: Use docling (handles Thai, tables, academic layout) ---
        if ext == ".pdf":
            return load_pdf_with_docling(file_path)

        # --- Markdown: UnstructuredMarkdownLoader preserves heading hierarchy ---
        elif ext == ".md":
            log.info(f"  [markdown] Loading: {file_path.name}")
            loader = UnstructuredMarkdownLoader(file_str, mode="single")
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "file_name": file_path.name,
                    "file_type": "markdown",
                    "loader": "UnstructuredMarkdownLoader",
                })
            log.info(f"  [markdown] ✓ Loaded {len(docs)} section(s) from {file_path.name}")
            return docs

        # --- Python: Language-aware splitter preserves function/class boundaries ---
        elif ext == ".py":
            log.info(f"  [python] Loading: {file_path.name}")
            loader = TextLoader(file_str, encoding="utf-8")
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.metadata.update({
                    "file_name": file_path.name,
                    "file_type": "python",
                    "loader": "TextLoader+LanguageSplitter",
                })
            log.info(f"  [python] ✓ Loaded {file_path.name} ({len(raw_docs[0].page_content):,} chars)")
            return raw_docs

        # --- Arduino .ino: Treat as C++ for language-aware splitting ---
        elif ext == ".ino":
            log.info(f"  [arduino/cpp] Loading: {file_path.name}")
            loader = TextLoader(file_str, encoding="utf-8")
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.metadata.update({
                    "file_name": file_path.name,
                    "file_type": "arduino_cpp",
                    "loader": "TextLoader+LanguageSplitter",
                })
            log.info(f"  [arduino/cpp] ✓ Loaded {file_path.name} ({len(raw_docs[0].page_content):,} chars)")
            return raw_docs

        # --- YAML/YML: Raw TextLoader — LLM sees exact keys and values ---
        elif ext in {".yaml", ".yml"}:
            log.info(f"  [yaml] Loading: {file_path.name}")
            loader = TextLoader(file_str, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "file_name": file_path.name,
                    "file_type": "yaml",
                    "loader": "TextLoader",
                })
            log.info(f"  [yaml] ✓ Loaded {file_path.name} ({len(docs[0].page_content):,} chars)")
            return docs

        else:
            log.warning(f"  [skip] Unsupported file type: {file_path.name}")
            return []

    except Exception as e:
        log.error(f"  ERROR loading {file_path.name}: {e}")
        return []


# ============================================================
# HELPER: LANGUAGE-AWARE SPLITTER SELECTOR
# ============================================================

def get_splitter_for_doc(doc: Document) -> RecursiveCharacterTextSplitter:
    """
    Return the appropriate text splitter for a document based on its file_type metadata.

    - Python files: Language.PYTHON (respects function/class boundaries)
    - Arduino/C++ files: Language.CPP (respects function boundaries)
    - All others: Standard RecursiveCharacterTextSplitter with markdown-aware separators
    """
    file_type = doc.metadata.get("file_type", "")

    if file_type == "python":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    elif file_type == "arduino_cpp":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    else:
        # General splitter with markdown-friendly separators
        # Tries to split on headings → paragraphs → sentences → words
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", ".", " ", ""],
            length_function=len,
        )


# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================

def collect_files() -> List[Path]:
    """
    Recursively walk all SCAN_DIRS and collect files with supported extensions.
    Returns a sorted list of unique file paths.
    """
    found: List[Path] = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            log.warning(f"Directory not found, skipping: {scan_dir}")
            continue
        for file_path in sorted(scan_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                found.append(file_path)
    return found


def ingest():
    """
    Main ingestion function.

    Steps:
      1. Collect all supported files from data/ subfolders.
      2. Load & parse each file with the appropriate loader.
      3. Split documents into chunks.
      4. Initialize HuggingFace embeddings (local, no API key).
      5. Persist all chunks to ChromaDB.
    """
    log.info("=" * 60)
    log.info("  BeanBot RAG — Knowledge Ingestion Pipeline")
    log.info("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Collect files
    # ----------------------------------------------------------
    log.info("\n📁 Step 1: Scanning data/ folders for documents...")
    files = collect_files()

    if not files:
        log.error("No supported files found in any of the scan directories.")
        log.error(f"Searched in: {[str(d) for d in SCAN_DIRS]}")
        sys.exit(1)

    log.info(f"  Found {len(files)} file(s) to process:\n")
    for f in files:
        log.info(f"    • {f.relative_to(REPO_ROOT)}")

    # ----------------------------------------------------------
    # Step 2: Load & parse each file
    # ----------------------------------------------------------
    log.info("\n📄 Step 2: Loading and parsing documents...")
    all_raw_docs: List[Document] = []
    failed_files: List[str] = []

    for file_path in files:
        log.info(f"\n  Processing: {file_path.relative_to(REPO_ROOT)}")
        docs = load_document(file_path)
        if docs:
            all_raw_docs.extend(docs)
        else:
            failed_files.append(str(file_path.name))

    log.info(f"\n  ✓ Loaded {len(all_raw_docs)} document section(s) from {len(files) - len(failed_files)} file(s).")
    if failed_files:
        log.warning(f"  ⚠ Failed to load {len(failed_files)} file(s): {failed_files}")

    # ----------------------------------------------------------
    # Step 3: Split into chunks
    # ----------------------------------------------------------
    log.info("\n✂️  Step 3: Splitting documents into chunks...")
    log.info(f"  Settings: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")

    all_chunks: List[Document] = []
    for doc in all_raw_docs:
        splitter = get_splitter_for_doc(doc)
        chunks = splitter.split_documents([doc])

        # Enrich each chunk with an index for traceability
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        all_chunks.extend(chunks)
        file_name = doc.metadata.get("file_name", "unknown")
        log.info(f"  • {file_name}: {len(chunks)} chunk(s)")

    log.info(f"\n  ✓ Total chunks created: {len(all_chunks)}")

    if not all_chunks:
        log.error("No chunks were created. Check if the documents have text content.")
        sys.exit(1)

    # ----------------------------------------------------------
    # Step 4: Initialize embeddings (local, CPU-based)
    # ----------------------------------------------------------
    log.info("\n🧠 Step 4: Initializing local HuggingFace embeddings...")
    log.info(f"  Model: {EMBEDDING_MODEL_NAME}")
    log.info("  (First run will download ~90MB model — this is cached locally)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("  ✓ Embedding model loaded successfully.")

    # ----------------------------------------------------------
    # Step 5: Persist to ChromaDB
    # ----------------------------------------------------------
    log.info(f"\n🗄️  Step 5: Storing chunks in ChromaDB...")
    log.info(f"  Path: {CHROMA_DB_DIR}")
    log.info(f"  Collection: '{CHROMA_COLLECTION_NAME}'")

    # Ensure the chroma_db directory exists
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"  Embedding and storing {len(all_chunks)} chunks — this may take a few minutes...")
    t_start = time.time()

    # Build in batches to avoid memory issues with large document sets
    BATCH_SIZE = 50
    batches = [all_chunks[i : i + BATCH_SIZE] for i in range(0, len(all_chunks), BATCH_SIZE)]

    vector_store = None
    for batch_idx, batch in enumerate(batches):
        log.info(f"  Batch {batch_idx + 1}/{len(batches)}: {len(batch)} chunks...")
        if vector_store is None:
            # Create the collection on the first batch
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=CHROMA_COLLECTION_NAME,
                persist_directory=str(CHROMA_DB_DIR),
            )
        else:
            # Add subsequent batches to the existing collection
            vector_store.add_documents(batch)

    elapsed = time.time() - t_start

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("  ✅ Ingestion Complete!")
    log.info("=" * 60)
    log.info(f"  Files processed     : {len(files) - len(failed_files)} / {len(files)}")
    log.info(f"  Total chunks stored : {len(all_chunks)}")
    log.info(f"  ChromaDB location   : {CHROMA_DB_DIR}")
    log.info(f"  Time elapsed        : {elapsed:.1f}s")
    log.info("\n  ▶ Next step: Run python 02_rag_pipeline.py")
    log.info("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    ingest()
