"""
03_app.py
=========
BeanBot RAG — Streamlit Chat Interface

This script is Step 3 of the BeanBot RAG system.

It provides a premium coffee-themed chat UI that:
  - Mimics a modern chat application (ChatGPT / LINE style)
  - Retains full conversation history in st.session_state
  - Displays the source document chunks used under each answer
  - Streams responses with a loading indicator
  - Works fully offline (only Gemini API needs internet)

Usage:
  streamlit run 03_app.py

Dependencies:
  - Run 01_ingest_knowledge.py first to populate ChromaDB
  - Set GOOGLE_API_KEY in .env file
"""

import streamlit as st
import time
from datetime import datetime

# ============================================================
# PAGE CONFIG — Must be the very first Streamlit call
# ============================================================

st.set_page_config(
    page_title="BeanBot AI Assistant",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Coffee-themed dark design
# ============================================================

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #1a0f00 0%, #2d1b00 50%, #1a0f00 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0800 0%, #1f1100 100%);
    border-right: 1px solid #3d2800;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #d4a853;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #c9a96e;
    font-size: 0.875rem;
}

/* ── Chat container ── */
.main .block-container {
    max-width: 900px;
    padding-top: 1rem;
    padding-bottom: 6rem;
}

/* ── User chat bubble ── */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #3d2800 0%, #5c3d00 100%);
    border: 1px solid #7a5200;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin-bottom: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}

/* ── Assistant chat bubble ── */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: linear-gradient(135deg, #1f1100 0%, #2d1a00 100%);
    border: 1px solid #3d2800;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin-bottom: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}

/* ── Chat text colors ── */
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span {
    color: #f0e6d3 !important;
    line-height: 1.6;
}

div[data-testid="stChatMessage"] strong {
    color: #d4a853 !important;
}

div[data-testid="stChatMessage"] code {
    background: #0d0800 !important;
    color: #ffd280 !important;
    border: 1px solid #5c3d00;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.85em;
}

div[data-testid="stChatMessage"] pre {
    background: #0d0800 !important;
    border: 1px solid #5c3d00 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

div[data-testid="stChatMessage"] pre code {
    border: none !important;
    padding: 0 !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] {
    background: #1f1100;
    border: 1px solid #5c3d00;
    border-radius: 16px;
    box-shadow: 0 -2px 20px rgba(0,0,0,0.5);
}

div[data-testid="stChatInput"] textarea {
    color: #f0e6d3 !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #7a5200 !important;
}

/* ── Source expander ── */
.streamlit-expanderHeader {
    background: #0d0800 !important;
    color: #d4a853 !important;
    border: 1px solid #3d2800 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
}

.streamlit-expanderContent {
    background: #0d0800 !important;
    border: 1px solid #3d2800 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Source chunk card ── */
.source-card {
    background: linear-gradient(135deg, #1a0f00, #2d1b00);
    border: 1px solid #5c3d00;
    border-left: 3px solid #d4a853;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #c9a96e;
    line-height: 1.5;
}

.source-card .source-filename {
    font-weight: 600;
    color: #ffd280;
    font-size: 0.78rem;
    margin-bottom: 4px;
    display: block;
}

.source-card .source-preview {
    color: #a88a5a;
    font-size: 0.78rem;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1f1100, #2d1a00);
    border: 1px solid #3d2800;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stMetricValue"] {
    color: #d4a853 !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
    color: #a88a5a !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #5c3d00, #7a5200);
    color: #ffd280;
    border: 1px solid #d4a853;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #7a5200, #9e6b00);
    border-color: #ffd280;
    box-shadow: 0 0 16px rgba(212, 168, 83, 0.3);
    transform: translateY(-1px);
}

/* ── Dividers ── */
hr {
    border-color: #3d2800 !important;
}

/* ── Welcome banner ── */
.welcome-banner {
    text-align: center;
    padding: 40px 20px;
    margin-bottom: 20px;
}

.welcome-banner h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffd280, #d4a853, #a87c2a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}

.welcome-banner p {
    color: #a88a5a;
    font-size: 1rem;
    margin: 0;
}

.suggested-chip {
    display: inline-block;
    background: linear-gradient(135deg, #3d2800, #5c3d00);
    border: 1px solid #7a5200;
    border-radius: 20px;
    padding: 6px 14px;
    margin: 4px;
    font-size: 0.82rem;
    color: #ffd280;
    cursor: pointer;
    transition: all 0.2s;
}

.suggested-chip:hover {
    border-color: #d4a853;
    box-shadow: 0 0 10px rgba(212,168,83,0.2);
}

/* ── Spinner override ── */
div[data-testid="stSpinner"] {
    color: #d4a853 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #1a0f00;
}
::-webkit-scrollbar-thumb {
    background: #5c3d00;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #7a5200;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize all session state variables on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []          # Chat history for display
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []       # Simplified history for RAG pipeline
    if "source_map" not in st.session_state:
        st.session_state.source_map = {}        # Maps message index → source docs
    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 0
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False


init_session_state()


# ============================================================
# LOAD PIPELINE (cached — runs only once per session)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load the RAG pipeline once and cache it for the Streamlit session.
    Returns True on success, error message string on failure.
    """
    try:
        from rag_pipeline import initialize_pipeline
        initialize_pipeline()
        return True
    except SystemExit as e:
        return f"SystemExit: {e}"
    except Exception as e:
        return str(e)


@st.cache_resource(show_spinner=False)
def get_ask_fn():
    """Return the ask_beanbot function from the pipeline module."""
    from rag_pipeline import ask_beanbot
    return ask_beanbot


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ☕ BeanBot")
    st.markdown("*AI Knowledge Assistant*")
    st.divider()

    st.markdown("### 📖 Knowledge Base")
    st.markdown("""
- 🇹🇭 ปริญญานิพนธ์ (Thai thesis)
- 🇬🇧 English research paper
- 🔧 Arduino controller code
- 🐍 Python detection system
- ⚙️ YOLO training configs
- 📊 Dataset configurations
    """)

    st.divider()

    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", st.session_state.total_questions)
    with col2:
        st.metric("Messages", len(st.session_state.messages))

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag_history = []
        st.session_state.source_map = {}
        st.session_state.total_questions = 0
        st.rerun()

    st.divider()

    st.markdown("### 💡 Example Questions")
    example_questions = [
        "YOLO model ที่ใช้คือรุ่นอะไร?",
        "Arduino ทำงานอย่างไร?",
        "ค่า confidence threshold คือเท่าไหร่?",
        "How does the conveyor system work?",
        "มีกี่ class ในการ classify?",
        "What dataset was used for training?",
    ]
    for q in example_questions:
        st.markdown(f'<div class="suggested-chip">💬 {q}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<p style='color:#5c3d00; font-size:0.75rem; text-align:center;'>"
        "BeanBot RAG v1.0<br>Powered by Gemini + ChromaDB</p>",
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN CHAT AREA
# ============================================================

# ── Header ──
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-banner">
        <h1>☕ BeanBot AI</h1>
        <p>ผู้ช่วย AI สำหรับโปรเจกต์เครื่องคัดแยกเมล็ดกาแฟอัตโนมัติ</p>
        <p style="font-size:0.85rem; margin-top:8px;">
            Ask me anything about the thesis, Arduino code, YOLO model, or training configs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggested question buttons
    st.markdown("**💡 Try asking:**")
    btn_cols = st.columns(3)
    quick_questions = [
        ("🤖 YOLO Model", "YOLO model ที่ใช้ในโปรเจกต์คือรุ่นอะไร และ train กี่ epoch?"),
        ("⚡ Arduino Code", "Arduino controller ทำงานยังไง? อธิบาย step by step"),
        ("📊 Dataset", "Dataset ที่ใช้ train มีกี่ class และมี class อะไรบ้าง?"),
        ("🎯 Confidence", "What confidence threshold is used for bean detection?"),
        ("🔧 Hardware", "กล้องและสายพานเชื่อมต่อกันอย่างไร?"),
        ("📈 Accuracy", "Model accuracy ของโปรเจกต์นี้เป็นเท่าไหร่?"),
    ]
    for i, (label, question) in enumerate(quick_questions):
        with btn_cols[i % 3]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state._pending_question = question
                st.rerun()

# ── Load pipeline (with status indicator) ──
with st.spinner("☕ Warming up BeanBot..."):
    pipeline_status = load_pipeline()

if pipeline_status is not True:
    st.error(f"❌ Failed to load pipeline: {pipeline_status}")
    st.info("Make sure you have:\n1. Run `python 01_ingest_knowledge.py` first\n2. Set `GOOGLE_API_KEY` in your `.env` file")
    st.stop()

ask_beanbot = get_ask_fn()

# ── Render existing chat history ──
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "☕"):
        st.markdown(msg["content"])

        # Show source documents for assistant messages
        if msg["role"] == "assistant" and i in st.session_state.source_map:
            sources = st.session_state.source_map[i]
            if sources:
                with st.expander(f"📚 {len(sources)} source chunk(s) used", expanded=False):
                    for j, doc in enumerate(sources, 1):
                        file_name = doc.metadata.get("file_name", "unknown")
                        file_type = doc.metadata.get("file_type", "")
                        chunk_idx = doc.metadata.get("chunk_index", "?")
                        total_chunks = doc.metadata.get("total_chunks", "?")
                        preview = doc.page_content[:300].replace("\n", " ").strip()

                        # Pick icon based on file type
                        icon = {
                            "pdf": "📄", "python": "🐍",
                            "arduino_cpp": "⚡", "yaml": "⚙️", "markdown": "📝"
                        }.get(file_type, "📄")

                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-filename">{icon} {file_name} '
                            f'<span style="color:#5c3d00; font-weight:400;">'
                            f'(chunk {chunk_idx}/{total_chunks})</span></span>'
                            f'<span class="source-preview">{preview}…</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

# ── Handle quick question buttons ──
pending = st.session_state.pop("_pending_question", None)

# ── Chat input ──
user_input = st.chat_input("ถามเกี่ยวกับโปรเจกต์เครื่องคัดแยกเมล็ดกาแฟ... / Ask about the coffee sorter project...")

# Use pending quick question if no direct input
question = pending or user_input

# ── Process question ──
if question:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.total_questions += 1

    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    # Generate and display assistant response
    with st.chat_message("assistant", avatar="☕"):
        with st.spinner("☕ BeanBot กำลังค้นหาคำตอบ..."):
            try:
                response = ask_beanbot(
                    question=question,
                    chat_history=st.session_state.rag_history,
                )
                answer = response["answer"]
                source_docs = response["source_documents"]

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    answer = (
                        "⚠️ **API Quota Exceeded**\n\n"
                        "Free-tier request limit reached for today.\n\n"
                        "**Options:**\n"
                        "- Wait until **7:00 AM tomorrow** (quota resets at midnight UTC)\n"
                        "- Or enable billing at [Google AI Studio](https://aistudio.google.com)"
                    )
                elif "API_KEY" in error_msg or "authentication" in error_msg.lower():
                    answer = (
                        "⚠️ **API Key Error**\n\n"
                        "Could not authenticate with Gemini API.\n"
                        "Please check your `GOOGLE_API_KEY` in `.env`."
                    )
                else:
                    answer = f"⚠️ **Error:** {error_msg}"
                source_docs = []

        st.markdown(answer)

        # Show sources in expander
        if source_docs:
            msg_index = len(st.session_state.messages)  # Index of the upcoming assistant msg
            with st.expander(f"📚 {len(source_docs)} source chunk(s) used", expanded=False):
                for j, doc in enumerate(source_docs, 1):
                    file_name = doc.metadata.get("file_name", "unknown")
                    file_type = doc.metadata.get("file_type", "")
                    chunk_idx = doc.metadata.get("chunk_index", "?")
                    total_chunks = doc.metadata.get("total_chunks", "?")
                    preview = doc.page_content[:300].replace("\n", " ").strip()

                    icon = {
                        "pdf": "📄", "python": "🐍",
                        "arduino_cpp": "⚡", "yaml": "⚙️", "markdown": "📝"
                    }.get(file_type, "📄")

                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="source-filename">{icon} {file_name} '
                        f'<span style="color:#5c3d00; font-weight:400;">'
                        f'(chunk {chunk_idx}/{total_chunks})</span></span>'
                        f'<span class="source-preview">{preview}…</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Persist to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.source_map[len(st.session_state.messages) - 1] = source_docs

    # Update RAG history (keep last 10 turns to avoid token overflow)
    st.session_state.rag_history.append({"role": "user", "content": question})
    st.session_state.rag_history.append({"role": "assistant", "content": answer})
    if len(st.session_state.rag_history) > 20:
        st.session_state.rag_history = st.session_state.rag_history[-20:]
