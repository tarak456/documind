import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
from document_processor import DocumentProcessor
from rag_engine import RAGEngine

st.set_page_config(
    page_title="DocMind — Smart Summarizer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #f7faf7 !important;
    font-family: 'DM Sans', sans-serif;
    color: #1a2e1a;
}
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 2px solid #c8e6c8 !important;
    box-shadow: 4px 0 24px rgba(34,139,34,0.06);
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem 2.5rem !important; max-width: 1400px !important; }

.sb-brand {
    background: linear-gradient(135deg, #1a7a1a 0%, #2ea82e 100%);
    padding: 28px 24px 22px;
}
.sb-brand-title { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: #ffffff; letter-spacing: -0.5px; line-height: 1.2; }
.sb-brand-sub { font-size: 0.75rem; color: rgba(255,255,255,0.75); margin-top: 4px; letter-spacing: 1.5px; text-transform: uppercase; }
.sb-section { padding: 20px 20px 16px; border-bottom: 1px solid #e8f5e8; }
.sb-section-title { font-size: 0.7rem; font-weight: 700; color: #5a9a5a; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px; }

.api-box {
    background: #f1f8f1; border: 1.5px solid #a5d6a7;
    border-radius: 12px; padding: 14px 16px; margin-bottom: 8px;
}
.api-box-title { font-size: 0.75rem; font-weight: 700; color: #2e7d32; margin-bottom: 4px; }
.api-box-sub { font-size: 0.72rem; color: #6a9a6a; }
.api-link { color: #2ea82e; font-weight: 600; text-decoration: none; }

.status-pill { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 10px; font-size: 0.82rem; font-weight: 500; }
.status-ok  { background: #e8f5e8; border: 1.5px solid #81c784; color: #2e7d32; }
.status-err { background: #fce4e4; border: 1.5px solid #e57373; color: #c62828; }
.status-warn{ background: #fff8e1; border: 1.5px solid #ffd54f; color: #e65100; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-green  { background: #43a047; box-shadow: 0 0 6px #43a047; }
.dot-red    { background: #e53935; }
.dot-yellow { background: #ffa000; }

.fmt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 4px; }
.fmt-badge { background: #f1f8f1; border: 1px solid #c8e6c8; border-radius: 8px; padding: 6px 10px; font-size: 0.75rem; font-weight: 600; color: #2e7d32; text-align: center; }

.page-title { font-family: 'Playfair Display', serif; font-size: 2.6rem; color: #1a2e1a; letter-spacing: -1px; line-height: 1.1; }
.page-title span { color: #2ea82e; }
.page-sub { font-size: 0.95rem; color: #6a8f6a; margin-top: 6px; font-weight: 400; }

.upload-outer { background: #ffffff; border: 2px dashed #81c784; border-radius: 20px; padding: 36px; text-align: center; margin-bottom: 20px; }
.upload-icon  { font-size: 3rem; margin-bottom: 10px; }
.upload-title { font-size: 1.15rem; font-weight: 600; color: #1a4a1a; }
.upload-sub   { font-size: 0.82rem; color: #7a9a7a; margin-top: 4px; }

.file-confirmed { display: flex; align-items: center; gap: 12px; background: #e8f5e8; border: 1.5px solid #81c784; border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; }
.fc-icon { font-size: 1.4rem; }
.fc-name { font-weight: 600; color: #1a4a1a; font-size: 0.9rem; }
.fc-meta { font-size: 0.78rem; color: #5a8a5a; }
.fc-badge { margin-left: auto; background: #2ea82e; color: white; border-radius: 8px; padding: 4px 12px; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; }

.stButton > button {
    background: linear-gradient(135deg, #1a7a1a, #2ea82e) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 12px 24px !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important; font-weight: 600 !important;
    transition: all 0.25s !important; box-shadow: 0 4px 16px rgba(46,168,46,0.3) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(46,168,46,0.45) !important; }

.metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
.metric-box { background: #ffffff; border: 1.5px solid #c8e6c8; border-radius: 16px; padding: 20px 16px; text-align: center; box-shadow: 0 2px 12px rgba(34,139,34,0.05); }
.metric-val { font-family: 'Playfair Display', serif; font-size: 1.9rem; color: #1a7a1a; line-height: 1; }
.metric-lbl { font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #7aaa7a; margin-top: 6px; }

.summary-card { background: #ffffff; border: 2px solid #c8e6c8; border-radius: 20px; padding: 32px; box-shadow: 0 4px 24px rgba(34,139,34,0.07); margin-bottom: 20px; position: relative; overflow: hidden; }
.summary-card::before { content: ''; position: absolute; top: 0; left: 0; width: 5px; height: 100%; background: linear-gradient(180deg, #2ea82e, #81c784); }
.summary-header { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding-left: 8px; }
.summary-title { font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #1a2e1a; }
.summary-tag { background: #e8f5e8; border: 1px solid #a5d6a7; border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-weight: 700; color: #2e7d32; letter-spacing: 0.5px; }
.summary-body { font-size: 0.97rem; line-height: 1.85; color: #2a4a2a; padding-left: 8px; }

.qa-section { background: #ffffff; border: 2px solid #c8e6c8; border-radius: 20px; padding: 28px 32px; margin-top: 24px; box-shadow: 0 4px 20px rgba(34,139,34,0.06); position: relative; overflow: hidden; }
.qa-section::before { content: ''; position: absolute; top: 0; left: 0; width: 5px; height: 100%; background: linear-gradient(180deg, #43a047, #a5d6a7); }
.qa-title { font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #1a2e1a; }
.qa-subtitle { font-size: 0.82rem; color: #6a9a6a; margin-top: 2px; }
.qa-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #5a9a5a; margin-bottom: 12px; padding-left: 8px; }

.chat-bubble-user { background: #e8f5e8; border: 1.5px solid #a5d6a7; border-radius: 16px 16px 4px 16px; padding: 12px 18px; margin: 8px 0; font-size: 0.88rem; color: #1a3a1a; font-weight: 500; max-width: 80%; margin-left: auto; text-align: right; }
.chat-bubble-ai { background: #f8fff8; border: 1.5px solid #c8e6c8; border-radius: 16px 16px 16px 4px; padding: 14px 18px; margin: 8px 0; font-size: 0.9rem; line-height: 1.75; color: #2a4a2a; max-width: 90%; }
.chat-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px; }
.chat-label-user { color: #5a9a5a; text-align: right; }
.chat-label-ai   { color: #2e7d32; }

.stDownloadButton > button { background: #f1f8f1 !important; color: #1a7a1a !important; border: 2px solid #81c784 !important; border-radius: 10px !important; font-weight: 600 !important; }
.stSelectbox label, .stSlider label { color: #3a6a3a !important; font-weight: 500 !important; }
.stSelectbox > div > div { background: #f8fff8 !important; border: 1.5px solid #c8e6c8 !important; border-radius: 10px !important; color: #1a2e1a !important; }
.stTextInput > div > div > input { background: #f8fff8 !important; border: 1.5px solid #c8e6c8 !important; border-radius: 10px !important; color: #1a2e1a !important; font-family: 'DM Sans', sans-serif !important; padding: 12px 16px !important; }
.stTextInput > div > div > input:focus { border-color: #2ea82e !important; box-shadow: 0 0 0 3px rgba(46,168,46,0.15) !important; }
hr { border: none; border-top: 2px solid #e0f0e0 !important; margin: 20px 0 !important; }
.stTextArea textarea { background: #f8fff8 !important; border: 1.5px solid #c8e6c8 !important; color: #1a2e1a !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "summary": None, "doc_text": None, "doc_metadata": {},
    "rag_engine": None, "pipeline_stage": 0,
    "qa_history": [], "active_tab": "summary",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Predefined Q&A ────────────────────────────────────────────────────────────
PREDEFINED_QA = {
    "📄 Document Overview": [
        "What is this document about?",
        "Who is the intended audience?",
        "What is the main purpose of this document?",
        "What are the key topics covered?",
    ],
    "🔍 Key Information": [
        "What are the most important findings?",
        "What are the main conclusions?",
        "What recommendations are made?",
        "What problems are identified?",
    ],
    "📊 Data & Numbers": [
        "What are the key statistics or numbers mentioned?",
        "What dates or timelines are mentioned?",
        "What financial figures are discussed?",
        "What metrics or KPIs are highlighted?",
    ],
    "💡 Insights": [
        "What are the strengths mentioned?",
        "What are the weaknesses or risks?",
        "What future plans or goals are discussed?",
        "What action items are suggested?",
    ],
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sb-brand'>
        <div class='sb-brand-title'>🌿 DocMind</div>
        <div class='sb-brand-sub'>Smart Document Summarizer</div>
    </div>
    """, unsafe_allow_html=True)

    # API Key section
    st.markdown("<div class='sb-section'>", unsafe_allow_html=True)
    st.markdown("<div class='sb-section-title'>🔑 Groq API Key</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='api-box'>
        <div class='api-box-title'>Free API Key Required</div>
        <div class='api-box-sub'>Get yours free at <a class='api-link' href='https://console.groq.com' target='_blank'>console.groq.com</a></div>
    </div>
    """, unsafe_allow_html=True)

    # Check for key in secrets first, else ask user
    groq_api_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    if not groq_api_key:
        groq_api_key = st.text_input(
            "Enter Groq API Key",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
        )

    if groq_api_key:
        st.markdown("<div class='status-pill status-ok'><div class='status-dot dot-green'></div>API Key set ✓</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-pill status-warn'><div class='status-dot dot-yellow'></div>Enter API key to start</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Summary style
    st.markdown("<div class='sb-section'>", unsafe_allow_html=True)
    st.markdown("<div class='sb-section-title'>Summary Style</div>", unsafe_allow_html=True)
    summary_type = st.selectbox("Type", ["Comprehensive", "Brief (3-5 lines)", "Bullet Points", "Executive", "Technical"], label_visibility="collapsed")
    summary_length = st.select_slider("Length", options=["Short", "Medium", "Long"], value="Medium")
    st.markdown("</div>", unsafe_allow_html=True)

    # Formats
    st.markdown("<div class='sb-section'>", unsafe_allow_html=True)
    st.markdown("<div class='sb-section-title'>Supported Formats</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='fmt-grid'>
        <div class='fmt-badge'>📕 PDF</div><div class='fmt-badge'>📘 DOCX</div>
        <div class='fmt-badge'>📄 TXT</div><div class='fmt-badge'>📊 CSV</div>
        <div class='fmt-badge'>📗 XLSX</div><div class='fmt-badge'>📙 PPTX</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Q&A session info
    if st.session_state.qa_history:
        st.markdown("<div class='sb-section'>", unsafe_allow_html=True)
        st.markdown("<div class='sb-section-title'>Q&A Session</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.82rem;color:#5a8a5a;margin-bottom:10px;'>💬 {len(st.session_state.qa_history)} question(s) asked</div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Q&A History", use_container_width=True):
            st.session_state.qa_history = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-title'>Smart Document<br><span>Summarizer</span></div>
<div class='page-sub'>Upload any document · AI extracts · RAG retrieves · Model summarizes & answers</div>
<br>
""", unsafe_allow_html=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────
stage = st.session_state.pipeline_stage

def pipe_class(i):
    if stage > i:  return "pipe-icon-done"
    if stage == i: return "pipe-icon-active"
    return "pipe-icon-idle"

def pipe_emoji(i, e):
    return "✅" if stage > i else e

steps = [
    ("📂", "Upload",    "Drop your file"),
    ("🔍", "Extract",   "Parse & read text"),
    ("✂️",  "Chunk",     "Split into segments"),
    ("📊", "TF-IDF",    "Score relevance"),
    ("🎯", "Retrieve",  "Top-K chunks"),
    ("🤖", "Summarize", "AI generates output"),
]

step_html = ""
for i, (emoji, name, desc) in enumerate(steps):
    arrow = "<div class='pipe-arrow'>→</div>" if i < len(steps) - 1 else ""
    step_html += f"""
    <div class='pipe-step'>
        <div class='pipe-icon-wrap {pipe_class(i)}'>{pipe_emoji(i, emoji)}</div>
        <div><div class='pipe-step-name'>{name}</div><div class='pipe-step-desc'>{desc}</div></div>
    </div>{arrow}"""

components.html(f"""
<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:transparent;font-family:'DM Sans',sans-serif;}}
  .pw{{background:#fff;border:2px solid #c8e6c8;border-radius:20px;padding:22px 28px;box-shadow:0 4px 20px rgba(34,139,34,0.07);}}
  .pl{{font-size:0.68rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#5a9a5a;margin-bottom:16px;}}
  .ps{{display:flex;align-items:center;gap:0;}}
  .pipe-step{{display:flex;flex-direction:column;align-items:center;gap:8px;flex:1;min-width:90px;}}
  .pipe-icon-wrap{{width:50px;height:50px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.35rem;border:2px solid;}}
  .pipe-icon-idle{{background:#f1f8f1;border-color:#c8e6c8;}}
  .pipe-icon-active{{background:#e8f5e8;border-color:#43a047;box-shadow:0 0 0 4px rgba(67,160,71,0.15);}}
  .pipe-icon-done{{background:#2ea82e;border-color:#2ea82e;}}
  .pipe-step-name{{font-size:0.72rem;font-weight:700;color:#2e5a2e;text-align:center;}}
  .pipe-step-desc{{font-size:0.62rem;color:#8aaa8a;text-align:center;line-height:1.3;}}
  .pipe-arrow{{font-size:1rem;color:#a8d5a8;margin:0 2px;padding-bottom:24px;flex-shrink:0;}}
</style></head><body>
  <div class="pw"><div class="pl">⚡ RAG Processing Pipeline</div><div class="ps">{step_html}</div></div>
</body></html>
""", height=150)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("upload", type=["pdf", "docx", "txt", "csv", "xlsx", "pptx"], label_visibility="collapsed")

if not uploaded_file:
    st.session_state.pipeline_stage = 0
    st.markdown("""
    <div class='upload-outer'>
        <div class='upload-icon'>📂</div>
        <div class='upload-title'>Drop your document here</div>
        <div class='upload-sub'>PDF · DOCX · TXT · CSV · XLSX · PPTX &nbsp;·&nbsp; Max 200 MB</div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    if st.session_state.pipeline_stage == 0:
        st.session_state.pipeline_stage = 1
        st.rerun()

    ext     = uploaded_file.name.split(".")[-1].upper()
    size_kb = round(uploaded_file.size / 1024, 1)
    icons   = {"PDF":"📕","DOCX":"📘","TXT":"📄","CSV":"📊","XLSX":"📗","PPTX":"📙"}
    icon    = icons.get(ext, "📄")

    st.markdown(f"""
    <div class='file-confirmed'>
        <div class='fc-icon'>{icon}</div>
        <div><div class='fc-name'>{uploaded_file.name}</div><div class='fc-meta'>{size_kb} KB &nbsp;·&nbsp; Ready to process</div></div>
        <div class='fc-badge'>{ext}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.8, 1.8, 4])
    with col1:
        go = st.button("🚀 Generate Summary", use_container_width=True)
    with col2:
        qa_btn = st.button("💬 Ask Questions", use_container_width=True)

    if qa_btn:
        if st.session_state.rag_engine is None:
            st.warning("⚠️ Please click **Generate Summary** first to process the document.")
            st.stop()
        st.session_state.active_tab = "qa"
        st.rerun()

    if go:
        if not groq_api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar first.")
            st.stop()

        st.session_state.active_tab = "summary"
        st.session_state.qa_history = []

        st.session_state.pipeline_stage = 2
        with st.spinner("📖 Extracting text from document..."):
            try:
                suffix = f".{uploaded_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                processor = DocumentProcessor()
                doc_text, metadata = processor.extract_text(tmp_path, uploaded_file.name)
                os.unlink(tmp_path)
                if not doc_text.strip():
                    st.error("❌ No text found. File may be image-based or empty.")
                    st.stop()
                st.session_state.doc_text     = doc_text
                st.session_state.doc_metadata = metadata
            except Exception as e:
                st.error(f"❌ Extraction failed: {e}")
                st.stop()

        st.session_state.pipeline_stage = 3
        with st.spinner("✂️ Chunking · Scoring · Retrieving top segments..."):
            try:
                rag = RAGEngine(api_key=groq_api_key)
                rag.build_index(st.session_state.doc_text)
                st.session_state.rag_engine = rag
                st.session_state.pipeline_stage = 5
            except Exception as e:
                st.error(f"❌ RAG build failed: {e}")
                st.stop()

        st.session_state.pipeline_stage = 6
        with st.spinner("🤖 AI is generating summary..."):
            try:
                summary = st.session_state.rag_engine.summarize(
                    st.session_state.doc_text,
                    summary_type=summary_type,
                    summary_length=summary_length,
                )
                st.session_state.summary = summary
            except Exception as e:
                st.error(f"❌ Summarization failed: {e}")
                st.stop()

        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.summary:
    meta = st.session_state.doc_metadata
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='metrics-row'>
        <div class='metric-box'><div class='metric-val'>{meta.get('word_count',0):,}</div><div class='metric-lbl'>Words</div></div>
        <div class='metric-box'><div class='metric-val'>{meta.get('char_count',0):,}</div><div class='metric-lbl'>Characters</div></div>
        <div class='metric-box'><div class='metric-val'>{meta.get('para_count',0):,}</div><div class='metric-lbl'>Paragraphs</div></div>
        <div class='metric-box'><div class='metric-val'>{meta.get('file_type','—')}</div><div class='metric-lbl'>File Type</div></div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, _ = st.columns([1.5, 1.5, 5])
    with t1:
        if st.button("📋 Summary", use_container_width=True):
            st.session_state.active_tab = "summary"
    with t2:
        if st.button("💬 Q&A", use_container_width=True):
            st.session_state.active_tab = "qa"

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SUMMARY TAB ──────────────────────────────────────────────────────────
    if st.session_state.active_tab == "summary":
        summary_html = st.session_state.summary.replace("\n", "<br>")
        st.markdown(f"""
        <div class='summary-card'>
            <div class='summary-header'>
                <div class='summary-title'>📋 Generated Summary</div>
                <span class='summary-tag'>{summary_type}</span>
                <span class='summary-tag'>{summary_length}</span>
            </div>
            <div class='summary-body'>{summary_html}</div>
        </div>
        """, unsafe_allow_html=True)

        col_dl, col_raw, _ = st.columns([1.2, 1.4, 4])
        with col_dl:
            st.download_button("⬇️ Download Summary", data=st.session_state.summary,
                file_name=f"summary_{uploaded_file.name.rsplit('.',1)[0]}.txt",
                mime="text/plain", use_container_width=True)
        with col_raw:
            if st.button("📄 View Extracted Text", use_container_width=True):
                with st.expander("Extracted Document Text", expanded=True):
                    st.text_area("", value=st.session_state.doc_text[:3000] +
                        ("\n\n... [truncated]" if len(st.session_state.doc_text) > 3000 else ""), height=280)

    # ── Q&A TAB ───────────────────────────────────────────────────────────────
    elif st.session_state.active_tab == "qa":
        st.markdown("""
        <div class='qa-section'>
            <div style='padding-left:8px; margin-bottom:22px;'>
                <div class='qa-title'>💬 Ask About This Document</div>
                <div class='qa-subtitle'>Pick a predefined question or type your own</div>
            </div>
        """, unsafe_allow_html=True)

        for category, questions in PREDEFINED_QA.items():
            st.markdown(f"<div class='qa-label'>{category}</div>", unsafe_allow_html=True)
            cols = st.columns(len(questions))
            for col, q in zip(cols, questions):
                with col:
                    if st.button(q, key=f"chip_{q}", use_container_width=True):
                        with st.spinner("🤖 Thinking..."):
                            try:
                                answer = st.session_state.rag_engine.answer_question(st.session_state.doc_text, q)
                                st.session_state.qa_history.append({"question": q, "answer": answer})
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='qa-section' style='margin-top:16px;'>
            <div style='padding-left:8px; margin-bottom:14px;'><div class='qa-title'>✏️ Ask Something Else</div></div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([5, 1])
        with c1:
            custom_q = st.text_input("", placeholder="e.g. What is the budget? Who are the key people?", label_visibility="collapsed", key="custom_q_input")
        with c2:
            ask_btn = st.button("🔍 Ask", use_container_width=True)

        if ask_btn and custom_q.strip():
            with st.spinner("🤖 Finding answer..."):
                try:
                    answer = st.session_state.rag_engine.answer_question(st.session_state.doc_text, custom_q.strip())
                    st.session_state.qa_history.append({"question": custom_q.strip(), "answer": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        elif ask_btn:
            st.warning("⚠️ Please type a question first.")

        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.qa_history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.72rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#5a9a5a;margin-bottom:12px;'>📜 Conversation History</div>", unsafe_allow_html=True)
            for item in reversed(st.session_state.qa_history):
                q_text = item["question"].replace("<","&lt;").replace(">","&gt;")
                a_text = item["answer"].replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
                st.markdown(f"""
                <div style='margin-bottom:16px;'>
                    <div class='chat-label chat-label-user'>You</div>
                    <div class='chat-bubble-user'>{q_text}</div>
                    <div class='chat-label chat-label-ai' style='margin-top:8px;'>🤖 DocMind AI</div>
                    <div class='chat-bubble-ai'>{a_text}</div>
                </div>
                """, unsafe_allow_html=True)

            qa_export = "\n\n".join(f"Q: {i['question']}\nA: {i['answer']}" for i in st.session_state.qa_history)
            st.download_button("⬇️ Download Q&A History", data=qa_export,
                file_name=f"qa_{uploaded_file.name.rsplit('.',1)[0]}.txt", mime="text/plain")
