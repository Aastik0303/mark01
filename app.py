"""
NexusRAG — Multi-Agent Intelligence Platform
Streamlit Frontend · Google Gemini API · LangChain · FAISS

FIXES:
  - Enter key submits chat (Shift+Enter for new line)
  - Vector DB showcase always renders as card grid (never plain text list)
"""

import streamlit as st
import os, sys, json, tempfile, base64, re
from pathlib import Path

# ── Page config ─────────────────────────

st.set_page_config(
    page_title="NeuralRAG · Multi-Agent AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ---------------- Global shell ---------------- */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #f8fcff;
    color: #0f1724;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
#MainMenu, footer, header { visibility: hidden; }
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #e6f7ff 0%, #f8fcff 50%, #ffffff 100%);
    transition: background 300ms ease;
}

/* Smooth default transitions */
* { transition: all 180ms cubic-bezier(.2,.9,.2,1); }

/* ---------------- Sidebar (pinned) ---------------- */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"] { display: none !important; }

[data-testid="stSidebar"] {
    min-width: 290px !important;
    max-width: 290px !important;
    background: linear-gradient(180deg,#eaf6ff 0%,#f6fbff 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.12) !important;
    box-shadow: 0 8px 28px rgba(59,130,246,0.06);
    padding-top: 0.6rem;
}

/* Ensure main content doesn't overlap sidebar */
.main .block-container { padding-left: 1.5rem !important; }

/* ---------------- Scrollbar ---------------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #e6f7ff; border-radius: 8px; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.22); border-radius: 8px; }

/* ---------------- Buttons ---------------- */
.stButton > button {
    background: linear-gradient(180deg, #e6f7ff 0%, #f6fbff 100%) !important;
    color: #0f1724 !important;
    border: 1px solid rgba(59,130,246,0.18) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.7rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.06);
}
.stButton > button:hover {
    border-color: rgba(59,130,246,0.32) !important;
    box-shadow: 0 12px 36px rgba(59,130,246,0.10) !important;
    transform: translateY(-2px) !important;
}

/* ---------------- Chat bubbles ---------------- */
.bubble-user {
    background: linear-gradient(180deg, #e6f7ff, #f0fbff);
    border: 1px solid rgba(59,130,246,0.14);
    border-radius: 20px 20px 8px 20px;
    padding: 0.85rem 1.2rem;
    max-width: 78%;
    color: #0f1724;
    font-size: 0.95rem;
    line-height: 1.5;
    margin: 0.45rem 0 0.45rem auto;
    box-shadow: inset 0 -1px 0 rgba(15,23,36,0.02);
}
.bubble-agent {
    background: linear-gradient(180deg, #ffffff, #f6fbff);
    border: 1px solid rgba(59,130,246,0.10);
    border-radius: 8px 20px 20px 20px;
    padding: 0.85rem 1.2rem;
    max-width: 86%;
    color: #0f1724;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0.45rem 0;
    box-shadow: 0 8px 24px rgba(59,130,246,0.04);
}
.bubble-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.35rem;
    opacity: 0.85;
}
.bubble-user .bubble-label { color: #2563eb; }
.bubble-agent .bubble-label { color: #3b82f6; }

/* ---------------- Inputs & selects ---------------- */
.stTextArea textarea,
.stTextInput input,
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid rgba(59,130,246,0.14) !important;
    border-radius: 10px !important;
    color: #0f1724 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.04);
}
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: rgba(59,130,246,0.28) !important;
    box-shadow: 0 10px 30px rgba(59,130,246,0.08) !important;
}

/* ---------------- Tabs ---------------- */
.stTabs [data-baseweb="tab-list"] {
    background: #f6fbff;
    border-radius: 12px;
    padding: 6px;
    gap: 6px;
    border: 1px solid rgba(59,130,246,0.10);
    overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #475569 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #cfe8ff, #e9f2ff) !important;
    color: #0f1724 !important;
    box-shadow: 0 10px 28px rgba(59,130,246,0.06);
}

/* ---------------- Badges & pills ---------------- */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #e6f7ff;
    border: 1px solid rgba(59,130,246,0.14);
    border-radius: 20px;
    padding: 0.25rem 0.7rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #1e3a8a;
    box-shadow: 0 6px 18px rgba(59,130,246,0.04);
}
.badge.on { border-color: rgba(16,185,129,0.12); color: #059669; background: #f0fdf4; }
.badge.info { border-color: rgba(59,130,246,0.18); color: #2563eb; background: #eef8ff; }

/* ---------------- Metrics & alerts ---------------- */
[data-testid="metric-container"] {
    background: #f6fbff;
    border: 1px solid rgba(59,130,246,0.10);
    border-radius: 12px;
    padding: 0.6rem;
    box-shadow: 0 8px 24px rgba(59,130,246,0.04);
}
.stSuccess { background: #ecfdf5 !important; border-left: 4px solid #10b981 !important; border-radius: 8px !important; }
.stError   { background: #fff5f5 !important; border-left: 4px solid #ef4444 !important; border-radius: 8px !important; }
.stInfo    { background: #eff6ff !important; border-left: 4px solid #3b82f6 !important; border-radius: 8px !important; }
.stWarning { background: #fffaf0 !important; border-left: 4px solid #f59e0b !important; border-radius: 8px !important; }

/* ---------------- Info card ---------------- */
.info-card {
    padding: 0.9rem;
    background: #ffffff;
    border-radius: 12px;
    border-left: 3px solid #3b82f6;
    margin-top: 0.5rem;
    box-shadow: 0 8px 24px rgba(59,130,246,0.04);
}
.info-card-title { color: #0f1724; font-weight: 600; }
.info-card-sub   { color: #6b7280; font-size: 0.86rem; margin-top: 0.25rem; }

/* ---------------- Source / timestamp pills ---------------- */
.src-pill {
    display: inline-block;
    background: #e6f7ff;
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #1e3a8a;
}
.ts-pill {
    display: inline-block;
    background: #dbeeff;
    border: 1px solid rgba(59,130,246,0.10);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #1e40af;
}

/* ---------------- File uploader ---------------- */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed rgba(59,130,246,0.12);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 8px 24px rgba(59,130,246,0.04);
}

/* ---------------- Pre / code blocks ---------------- */
pre, code {
    background: #f6fbff !important;
    border: 1px solid rgba(59,130,246,0.08) !important;
    border-radius: 10px !important;
    padding: 0.7rem !important;
    color: #0f1724 !important;
}

/* ---------------- Vector DB Showcase panel ---------------- */
.vdb-panel {
    background: #ffffff;
    border: 1px solid rgba(59,130,246,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-family: 'Space Mono', monospace;
    color: #0f1724;
}

/* Title / meta */
.vdb-title {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2563eb;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.vdb-sub { font-size:0.72rem; color:#6b7280; }

/* Stat row */
.vdb-stat-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.vdb-stat {
    background: #f6fbff;
    border: 1px solid rgba(59,130,246,0.08);
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    min-width: 120px;
    flex: 1 1 120px;
}
.vdb-stat-val { font-size: 1.05rem; font-weight: 700; color: #0f1724; }
.vdb-stat-label { font-size: 0.62rem; color: #6b7280; margin-top: 0.12rem; text-transform:uppercase; }

/* Chunk grid */
.chunk-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.75rem;
    max-height: 420px;
    overflow-y: auto;
    padding-right: 8px;
}
.chunk-grid::-webkit-scrollbar { width: 8px; }
.chunk-grid::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.12); border-radius: 8px; }

/* Chunk card */
.chunk-card {
    background: #ffffff;
    border: 1px solid rgba(59,130,246,0.06);
    border-radius: 12px;
    padding: 0.9rem;
    cursor: default;
    transition: box-shadow 220ms ease, border-color 220ms ease, transform 180ms ease;
    position: relative;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}
.chunk-card:hover {
    box-shadow: 0 12px 36px rgba(59,130,246,0.08);
    transform: translateY(-4px);
    border-color: rgba(59,130,246,0.14);
}
.chunk-card.highlighted {
    border-color: rgba(59,130,246,0.22);
    box-shadow: 0 16px 44px rgba(59,130,246,0.10);
    background: linear-gradient(180deg,#fbfdff,#ffffff);
}

/* Chunk meta */
.chunk-idx { font-size: 0.62rem; color: #6b7280; position: absolute; top: 0.6rem; right: 0.8rem; }
.chunk-hash { font-size: 0.72rem; color: #3b5bdb; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 70%; }
.chunk-src { font-size: 0.72rem; color: #475569; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 100%; }
.chunk-preview {
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.45;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-top: 0.15rem;
}
.chunk-chars { font-size: 0.68rem; color: #6b7280; margin-top: auto; }

/* Similarity results */
.sim-section { margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(59,130,246,0.06); }
.sim-title { font-size: 0.68rem; color: #0f6bff; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.6rem; }
.sim-query {
    font-size: 0.85rem;
    color: #0f1724;
    background: #f6fbff;
    border: 1px solid rgba(59,130,246,0.08);
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
    margin-bottom: 0.7rem;
    word-break: break-word;
}

/* Each result card */
.sim-result {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    padding: 0.6rem;
    background: #ffffff;
    border: 1px solid rgba(59,130,246,0.08);
    border-radius: 10px;
    position: relative;
    overflow: hidden;
    margin-bottom: 0.6rem;
}
.sim-result::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: var(--sim-pct);
    background: linear-gradient(90deg, rgba(59,130,246,0.06), transparent);
    pointer-events: none;
}
.sim-rank { font-size: 0.82rem; font-weight: 700; color: #3b5bdb; min-width: 36px; text-align: center; }
.sim-body { flex: 1; min-width: 0; }
.sim-score-row { display: flex; gap: 0.6rem; align-items: center; margin-bottom: 0.25rem; flex-wrap: wrap; }
.sim-pct { font-size: 0.82rem; font-weight: 700; color: #059669; }
.sim-chunk-id { font-size: 0.72rem; color: #3b5bdb; }
.sim-src-pill { font-size: 0.68rem; color: #475569; background: #eef8ff; border-radius: 6px; padding: 0.12rem 0.45rem; }

/* Preview text */
.sim-preview { font-size: 0.9rem; color: #374151; line-height: 1.45; }

/* Embedding visualizer */
.emb-section { margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(59,130,246,0.06); }
.emb-title { font-size: 0.68rem; color: #0ea5a4; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.6rem; }
.emb-bar-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.45rem; }
.emb-bar-label { font-size: 0.72rem; color: #6b7280; min-width: 80px; text-align: right; }
.emb-bar-wrap { flex: 1; height: 8px; background: #e6f7ff; border-radius: 6px; overflow: hidden; }
.emb-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg,#3b82f6,#7c6df2); transition: width 420ms ease; }
.emb-bar-val { font-size: 0.72rem; color: #0f1724; min-width: 40px; text-align: right; }

/* Routing log entries */
.routing-log { margin-top: 1rem; }
.routing-entry {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
    padding: 0.5rem 0.7rem;
    background: #f6fbff;
    border: 1px solid rgba(59,130,246,0.10);
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    box-shadow: 0 6px 18px rgba(59,130,246,0.04);
}
.routing-entry .agent { color: #2563eb; font-weight: 600; }
.routing-entry .method { color: #7c3aed; }
.routing-entry .confidence-high { color: #059669; }
.routing-entry .confidence-mid { color: #b45309; }
.routing-entry .confidence-low { color: #dc2626; }

/* ---------------- Responsive tweaks ---------------- */
@media (max-width: 768px) {
    .vdb-stat-row { flex-direction: column; gap: 0.5rem; }
    .chunk-grid { grid-template-columns: 1fr; max-height: 360px; }
    .chunk-card { min-height: 120px; }
    .stButton > button { min-height: 46px !important; }
}

/* End of CSS */
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = st.secrets.get("GEMINI_MODEL",   "gemini-2.0-flash")

_ss("agents_ready",      False)
_ss("orchestrator",      None)
_ss("active_agent",      "chat")
_ss("messages",          [])
_ss("rag_ingested",      False)
_ss("video_ingested",    False)
_ss("data_loaded",       False)
_ss("data_filename",     "")
_ss("data_shape",        "")
_ss("data_columns",      [])
_ss("_boot_error",       "")
_ss("video_url_saved",   "")
_ss("video_lang_saved",  "en")
_ss("last_vstore_data",  None)
_ss("last_routing",      None)
_ss("_enter_submit",     "")   # ← holds pending Enter-key submission


# ── Init agents ───────────────────────────────────────────────────────────────
def init_agents():
    from agents import MultiAgentOrchestrator, set_api_key
    set_api_key(GEMINI_API_KEY, GEMINI_MODEL)
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = MultiAgentOrchestrator()
    st.session_state.agents_ready = True
    st.session_state._boot_error  = ""

if not st.session_state.agents_ready:
    try:
        init_agents()
    except Exception as e:
        st.session_state._boot_error = str(e)


# ── Helpers ───────────────────────────────────────────────────────────────────
def push_msg(role, content, agent="", chart=None, code=None, lang="python",
             sources=None, research_sources=None, queries=None,
             timestamps=None, delegated=False, vstore_data=None, routing=None):
    st.session_state.messages.append({
        "role": role, "content": content, "agent": agent,
        "chart": chart, "code": code, "lang": lang,
        "sources":          sources or [],
        "research_sources": research_sources or [],
        "queries":          queries or [],
        "timestamps":       timestamps or [],
        "delegated":        delegated,
        "vstore_data":      vstore_data,
        "routing":          routing,
    })


def get_context():
    return {
        "rag_ingested":   st.session_state.rag_ingested,
        "video_ingested": st.session_state.video_ingested,
        "data_loaded":    st.session_state.data_loaded,
        "data_filename":  st.session_state.data_filename,
    }


def _conf_class(conf: float) -> str:
    if conf >= 0.8: return "conf-high"
    if conf >= 0.5: return "conf-mid"
    return "conf-low"


def render_message(msg):
    from vectordb_showcase import render_vectordb_showcase
    role    = msg["role"]
    content = msg["content"]
    agent   = msg.get("agent", "NEXUS")

    if role == "user":
        st.markdown(f"""
<div class="bubble-user">
  <div class="bubble-label">▸ You</div>
  {content}
</div>""", unsafe_allow_html=True)
    else:
        del_tag = '<span class="delegate-tag">↗ delegated</span>' if msg.get("delegated") else ""

        routing    = msg.get("routing") or {}
        conf_badge = ""
        if routing:
            conf = routing.get("confidence", 0)
            method = routing.get("method", "")
            cc = _conf_class(conf)
            conf_badge = f'<span class="conf-pill {cc}">{method} {int(conf*100)}%</span>'

        st.markdown(f"""
<div class="bubble-agent">
  <div class="bubble-label">⬡ {agent}{del_tag}{conf_badge}</div>
  {content.replace(chr(10), "<br>")}
</div>""", unsafe_allow_html=True)

        if msg.get("chart"):
            st.image(base64.b64decode(msg["chart"]), use_container_width=True)
        if msg.get("code"):
            st.code(msg["code"], language=msg.get("lang", "python"))
        if msg.get("sources"):
            pills = "".join(f'<span class="src-pill">📄 {s}</span>' for s in msg["sources"])
            st.markdown(f'<div style="margin-top:0.4rem">{pills}</div>', unsafe_allow_html=True)
        if msg.get("timestamps"):
            ts_html = "".join(
                f'<a href="{t["yt_link"]}" target="_blank" class="ts-pill">⏱ {t["timestamp"]}</a>'
                for t in msg["timestamps"] if t.get("yt_link")
            )
            if ts_html:
                st.markdown(f'<div style="margin-top:0.4rem">{ts_html}</div>', unsafe_allow_html=True)
        if msg.get("research_sources"):
            with st.expander(f"📚 {len(msg['research_sources'])} Sources"):
                for s in msg["research_sources"][:12]:
                    if s.get("url"):
                        st.markdown(f"- [{s['title']}]({s['url']})")
        if msg.get("queries"):
            with st.expander("🔍 Research Queries Used"):
                for q in msg["queries"]:
                    st.markdown(f"`{q}`")
        if msg.get("vstore_data") and msg["vstore_data"].get("total_chunks", 0) > 0:
            with st.expander("◈ Vector DB Showcase", expanded=False):
                render_vectordb_showcase(msg["vstore_data"])


# ── Enter-to-send JS injection ────────────────────────────────────────────────
def inject_enter_to_send():
    """
    Inject JS so that pressing Enter in the chat textarea triggers the Send button.
    Shift+Enter inserts a newline as normal.
    We identify the textarea by placeholder text and the button by its text content.
    """
    st.markdown("""
<script>
(function() {
  function attachEnterListener() {
    // Find all textareas in the page
    const textareas = window.parent.document.querySelectorAll('textarea');
    textareas.forEach(function(ta) {
      if (ta._nexusEnterAttached) return;
      ta._nexusEnterAttached = true;
      ta.addEventListener('keydown', function(e) {
        // Enter without Shift = send
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          // Find the Send button — it contains the ⬆ symbol or "Send" text
          const buttons = window.parent.document.querySelectorAll('button');
          for (let btn of buttons) {
            if (btn.innerText && (btn.innerText.includes('⬆') || btn.innerText.trim() === 'Send')) {
              btn.click();
              break;
            }
          }
        }
      });
    });
  }
  // Run immediately and re-run on DOM changes (Streamlit re-renders)
  attachEnterListener();
  const observer = new MutationObserver(attachEnterListener);
  observer.observe(window.parent.document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem">
  <div style="font-size:1.8rem;font-weight:800;
    background:linear-gradient(135deg,#7c6df2,#3b82f6,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;letter-spacing:-0.02em">⬡ NEURALRAG</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
    color:#475569;letter-spacing:0.15em;margin-top:0.2rem">MULTI-AGENT AI</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.agents_ready:
        st.markdown(f'<span class="badge on">● ONLINE · {GEMINI_MODEL}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge off">● OFFLINE</span>', unsafe_allow_html=True)
        if st.button("🔄 Retry", use_container_width=True):
            try:
                init_agents(); st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    st.markdown("---")
    st.markdown("**⬡ Select Agent**")

    AGENTS = [
        ("chat",     "🤖", "General Chatbot",  "Smart hub · memory · auto-delegates"),
        ("rag",      "📄", "Document Q&A",     "PDF · DOCX · TXT semantic search"),
        ("video",    "🎬", "YouTube RAG",      "YouTube URL → Q&A with timestamps"),
        ("data",     "📊", "Data Analyst",     "CSV/Excel analysis + charts"),
        ("code",     "💻", "Code Generator",   "Generate · Explain · Debug"),
        ("research", "🔬", "Web Researcher",   "Multi-step live web research"),
        ("auto",     "🧠", "Auto-Route",       "LLM picks the best agent"),
    ]

    for aid, icon, name, desc in AGENTS:
        is_active = st.session_state.active_agent == aid
        label = f"{icon} **{name}** ←" if is_active else f"{icon} {name}"
        if st.button(label, key=f"sbtn_{aid}", use_container_width=True, help=desc):
            st.session_state.active_agent = aid
            st.rerun()

    st.markdown("---")
    st.markdown("**📊 System Status**")
    c1, c2 = st.columns(2)
    c1.markdown(f'<span class="badge {"on" if st.session_state.rag_ingested else "off"}">📄 Docs</span>',   unsafe_allow_html=True)
    c2.markdown(f'<span class="badge {"on" if st.session_state.video_ingested else "off"}">🎬 Video</span>', unsafe_allow_html=True)
    c1.markdown(f'<span class="badge {"on" if st.session_state.data_loaded else "off"}">📊 Data</span>',    unsafe_allow_html=True)
    c2.markdown(f'<span class="badge info">{len(st.session_state.messages)} msgs</span>',                    unsafe_allow_html=True)

    st.markdown("")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.agents_ready and st.session_state.orchestrator:
            st.session_state.orchestrator.chatbot.clear_history()
        st.rerun()

    if (st.session_state.active_agent == "chat"
            and st.session_state.agents_ready
            and len(st.session_state.messages) > 4):
        if st.button("📝 Summarize Conversation", use_container_width=True):
            with st.spinner("Summarizing..."):
                s = st.session_state.orchestrator.chatbot.get_summary()
            st.info(s)

    st.markdown("---")
    if st.session_state.agents_ready and st.session_state.orchestrator:
        log = st.session_state.orchestrator.get_routing_log()
        if log:
            with st.expander("⟳ Routing Log", expanded=False):
                from vectordb_showcase import render_routing_log
                render_routing_log(log)

    if st.session_state.last_vstore_data:
        vd = st.session_state.last_vstore_data
        st.markdown("---")
        st.markdown("**◈ Vector DB**")
        st.markdown(
            f'<span class="badge info">🗂 {vd.get("total_chunks",0)} chunks</span>'
            f'<span class="badge info">📐 {vd.get("embedding_dim",0)}d</span>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="nexus-header">
  <div class="nexus-title">NEURALRAG</div>
  <div class="nexus-sub">Multi-Agent Intelligence · Gemini · LangChain · FAISS</div>
</div>""", unsafe_allow_html=True)

if st.session_state._boot_error:
    st.error(f"⚠️ {st.session_state._boot_error}")

if not st.session_state.agents_ready:
    st.error("⚠️ Could not connect. Please refresh the page.")
    if st.button("🔄 Retry Connection"):
        try:
            init_agents(); st.rerun()
        except Exception as e:
            st.error(str(e))
    st.stop()

orch   = st.session_state.orchestrator
active = st.session_state.active_agent

# Auto-restore video after server restart
if (st.session_state.get("video_ingested")
        and st.session_state.get("video_url_saved")
        and not orch.video_rag.is_ready()):
    try:
        orch.video_rag.ingest(
            st.session_state.video_url_saved,
            language=st.session_state.get("video_lang_saved", "en"),
        )
    except Exception:
        pass


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab_chat, tab_ingest, tab_viz, tab_vdb, tab_about = st.tabs([
    "💬 Chat", "📥 Ingest Data", "📊 Visualize", "◈ Vector DB", "ℹ️ About"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════
with tab_chat:

    AGENT_META = {
        "chat":     ("🤖", "General Chatbot",  "Conversational AI with memory · auto-delegates"),
        "rag":      ("📄", "Document Q&A",     "Semantic search over your uploaded documents"),
        "video":    ("🎬", "YouTube RAG",      "Paste a YouTube URL · Q&A with timestamps"),
        "data":     ("📊", "Data Analyst",     "Intelligent data analysis with AI charts"),
        "code":     ("💻", "Code Generator",   "Generate · Explain · Debug code"),
        "research": ("🔬", "Web Researcher",   "Multi-step live web research"),
        "auto":     ("🧠", "Auto-Route",       "LLM picks the best agent automatically"),
    }

    icon, name, desc = AGENT_META.get(active, ("⬡", "NEXUS", ""))
    st.markdown(f"""
<div class="agent-bar">
  <span class="agent-bar-icon">{icon}</span>
  <div>
    <div class="agent-bar-name">{name}</div>
    <div class="agent-bar-desc">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Message history ───────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
<div style="text-align:center;padding:3rem 1rem;color:#9ca3af;font-family:'Space Mono',monospace;font-size:0.78rem">
  ⬡<br><br>
  Start a conversation · Upload documents · Load a YouTube video<br>
  <span style="font-size:0.65rem;color:#374151">Select an agent from the sidebar or use Auto-Route</span>
</div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                render_message(msg)

    # ── YouTube URL ingest inline for video agent ─────────────────────────────
    if active == "video":
        col_url, col_lang, col_btn = st.columns([4, 1, 1])
        with col_url:
            yt_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...",
                                   key="yt_url_input", label_visibility="collapsed")
        with col_lang:
            yt_lang = st.selectbox("Lang", ["en","hi","es","fr","de","ja","ko","pt","ar","ru"],
                                   key="yt_lang_sel", label_visibility="collapsed")
        with col_btn:
            if st.button("▶ Load", key="yt_load_btn", use_container_width=True):
                if yt_url.strip():
                    with st.spinner("Loading video…"):
                        msg = orch.video_rag.ingest(yt_url.strip(), language=yt_lang)
                    if "Error" not in msg:
                        st.session_state.video_ingested   = True
                        st.session_state.video_url_saved  = yt_url.strip()
                        st.session_state.video_lang_saved = yt_lang
                        vd = orch.video_rag.get_showcase_data()
                        st.session_state.last_vstore_data = vd
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter a YouTube URL.")

    # ── Code agent language selector ──────────────────────────────────────────
    if active == "code":
        code_lang = st.selectbox(
            "Language",
            ["python","javascript","typescript","java","go","rust","sql","bash","c++","c#","php","kotlin"],
            key="code_lang_sel",
        )

    # ── Input area ────────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    placeholder_map = {
        "chat":     "Ask anything… (Enter to send · Shift+Enter for new line)",
        "rag":      "Ask about your uploaded documents… (Enter to send)",
        "video":    "Ask about the loaded video… (Enter to send)",
        "data":     "Ask about your data… (Enter to send)",
        "code":     "Describe what code you need… (Enter to send)",
        "research": "What do you want to research? (Enter to send)",
        "auto":     "Ask anything — I'll route to the best agent… (Enter to send)",
    }

    col_input, col_send = st.columns([6, 1])
    with col_input:
        user_input = st.text_area(
            "Message",
            placeholder=placeholder_map.get(active, "Type your message… (Enter to send)"),
            height=80,
            key="chat_input",
            label_visibility="collapsed",
        )
        # Hint below textarea
        st.markdown(
            '<div class="input-hint">↵ Enter to send &nbsp;·&nbsp; Shift+↵ for new line</div>',
            unsafe_allow_html=True,
        )
    with col_send:
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        send = st.button("⬆ Send", key="send_btn", use_container_width=True)

    # ── Inject Enter-to-send JS ───────────────────────────────────────────────
    inject_enter_to_send()

    # ── Handle send ───────────────────────────────────────────────────────────
    if send and user_input.strip():
        q = user_input.strip()
        push_msg("user", q)

        with st.spinner("Thinking…"):
            try:
                if active == "auto":
                    result   = orch.route(q, get_context())
                    routing  = result.get("_routing", {})
                    agent_id = routing.get("agent", "chat")
                    vd       = result.get("vstore_data")
                    if vd:
                        st.session_state.last_vstore_data = vd
                    push_msg(
                        "assistant", result.get("answer", ""),
                        agent=agent_id.upper(),
                        chart=result.get("chart"),
                        code=result.get("code"),
                        lang=result.get("lang", "python"),
                        sources=result.get("sources", []),
                        research_sources=result.get("sources", []) if agent_id == "research" else [],
                        queries=result.get("queries", []),
                        timestamps=result.get("timestamps", []),
                        vstore_data=vd,
                        routing=routing,
                        delegated=True,
                    )

                elif active == "rag":
                    if not st.session_state.rag_ingested:
                        push_msg("assistant", "⚠️ No documents loaded. Please upload files in the **Ingest Data** tab.", agent="RAG")
                    else:
                        result = orch.rag.query(q)
                        vd     = result.get("vstore_data")
                        if vd:
                            st.session_state.last_vstore_data = vd
                        push_msg(
                            "assistant", result["answer"],
                            agent="RAG",
                            sources=result.get("sources", []),
                            vstore_data=vd,
                        )

                elif active == "video":
                    if not st.session_state.video_ingested:
                        push_msg("assistant", "⚠️ No video loaded. Paste a YouTube URL above and click **▶ Load**.", agent="VIDEO RAG")
                    else:
                        result = orch.video_rag.query(q)
                        vd     = result.get("vstore_data")
                        if vd:
                            st.session_state.last_vstore_data = vd
                        push_msg(
                            "assistant", result["answer"],
                            agent="VIDEO RAG",
                            timestamps=result.get("timestamps", []),
                            vstore_data=vd,
                        )

                elif active == "data":
                    if not st.session_state.data_loaded:
                        push_msg("assistant", "⚠️ No data loaded. Upload a CSV or Excel file in the **Ingest Data** tab.", agent="DATA")
                    else:
                        result = orch.data_agent.analyze(q)
                        push_msg(
                            "assistant", result["answer"],
                            agent="DATA",
                            chart=result.get("chart"),
                        )

                elif active == "code":
                    lang   = st.session_state.get("code_lang_sel", "python")
                    result = orch.code_agent.generate(q, language=lang)
                    push_msg(
                        "assistant", result["answer"],
                        agent="CODE",
                        code=result.get("code", ""),
                        lang=lang,
                    )

                elif active == "research":
                    result = orch.research_agent.research(q)
                    push_msg(
                        "assistant", result["answer"],
                        agent="RESEARCH",
                        research_sources=result.get("sources", []),
                        queries=result.get("queries", []),
                    )

                else:  # chat
                    answer = orch.chatbot.chat(q)
                    push_msg("assistant", answer, agent="NEXUS")

            except Exception as e:
                push_msg("assistant", f"⚠️ Error: {e}", agent="SYSTEM")

        st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 2 — INGEST DATA
# ════════════════════════════════════════════════════════════
with tab_ingest:
    from vectordb_showcase import render_vectordb_showcase

    st.markdown("### 📥 Ingest Data")
    ing_tab1, ing_tab2, ing_tab3 = st.tabs(["📄 Documents", "🎬 YouTube", "📊 CSV / Excel"])

    # ── Documents ─────────────────────────────────────────────────────────────
    with ing_tab1:
        st.markdown("Upload PDF, TXT, MD, or CSV files to enable Document Q&A.")
        uploaded = st.file_uploader(
            "Drop files here", type=["pdf","txt","md","csv","docx"],
            accept_multiple_files=True, key="doc_uploader",
        )
        if st.button("🔄 Ingest Documents", key="ingest_doc_btn", use_container_width=True):
            if not uploaded:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Processing documents…"):
                    paths = []
                    for f in uploaded:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
                        tmp.write(f.read()); tmp.flush()
                        paths.append(tmp.name)
                    msg = orch.rag.ingest(paths)
                if "Error" not in msg:
                    st.session_state.rag_ingested    = True
                    vd = orch.rag.get_showcase_data()
                    st.session_state.last_vstore_data = vd
                    st.success(msg)
                    render_vectordb_showcase(vd, title="Document Vector Store")
                else:
                    st.error(msg)
        elif st.session_state.rag_ingested:
            vd = orch.rag.get_showcase_data()
            if vd.get("total_chunks", 0) > 0:
                render_vectordb_showcase(vd, title="Document Vector Store")

    # ── YouTube ───────────────────────────────────────────────────────────────
    with ing_tab2:
        st.markdown("Load a YouTube video to enable transcript Q&A.")
        yt_url2  = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", key="yt_url_ingest")
        yt_lang2 = st.selectbox("Transcript language", ["en","hi","es","fr","de","ja","ko","pt","ar","ru"], key="yt_lang_ingest")
        if st.button("▶ Load Video", key="load_yt_ingest", use_container_width=True):
            if yt_url2.strip():
                with st.spinner("Fetching transcript…"):
                    msg = orch.video_rag.ingest(yt_url2.strip(), language=yt_lang2)
                if "Error" not in msg:
                    st.session_state.video_ingested   = True
                    st.session_state.video_url_saved  = yt_url2.strip()
                    st.session_state.video_lang_saved = yt_lang2
                    vd = orch.video_rag.get_showcase_data()
                    st.session_state.last_vstore_data = vd
                    st.success(msg)
                    info = orch.video_rag.get_info()
                    if info.get("thumbnail"):
                        st.image(info["thumbnail"], width=280)
                    render_vectordb_showcase(vd, title="YouTube Vector Store")
                else:
                    st.error(msg)
            else:
                st.warning("Please enter a YouTube URL.")

        if st.session_state.video_ingested and orch.video_rag.is_ready():
            info = orch.video_rag.get_info()
            st.markdown(f"""
<div class="info-card">
  <div class="info-card-title">🎬 {info.get('title','Unknown')}</div>
  <div class="info-card-sub">
    {info.get('channel','?')} · {info.get('duration','?')} ·
    {info.get('transcript_segments',0)} segments ·
    source: {info.get('source_type','?')}
  </div>
</div>""", unsafe_allow_html=True)

            sum_style = st.selectbox("Summary style", ["detailed","brief","bullets"], key="sum_style_sel")
            if st.button("📝 Summarize Video", key="sum_video_btn", use_container_width=True):
                with st.spinner("Summarizing…"):
                    s = orch.video_rag.summarize(style=sum_style)
                st.markdown(s.get("summary", ""))

    # ── CSV / Excel ───────────────────────────────────────────────────────────
    with ing_tab3:
        st.markdown("Upload a CSV or Excel file for AI-powered data analysis.")
        data_file = st.file_uploader("Drop CSV or Excel here", type=["csv","xlsx","xls"], key="data_uploader")
        if st.button("📊 Load Data", key="load_data_btn", use_container_width=True):
            if not data_file:
                st.warning("Please upload a file.")
            else:
                with st.spinner("Loading…"):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(data_file.name).suffix)
                    tmp.write(data_file.read()); tmp.flush()
                    msg = orch.data_agent.load_data(tmp.name)
                if "Error" not in msg:
                    st.session_state.data_loaded   = True
                    st.session_state.data_filename = data_file.name
                    if orch.data_agent.df is not None:
                        df = orch.data_agent.df
                        st.session_state.data_shape   = f"{df.shape[0]:,} × {df.shape[1]}"
                        st.session_state.data_columns = list(df.columns)
                    st.success(msg)
                    if orch.data_agent.df is not None:
                        st.dataframe(orch.data_agent.df.head(10), use_container_width=True)
                else:
                    st.error(msg)
        elif st.session_state.data_loaded and orch.data_agent.df is not None:
            st.info(f"Loaded: **{st.session_state.data_filename}** · {st.session_state.data_shape}")
            st.dataframe(orch.data_agent.df.head(10), use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — VISUALIZE
# ════════════════════════════════════════════════════════════
with tab_viz:
    st.markdown("### 📊 Data Visualizer")

    if not st.session_state.data_loaded or orch.data_agent.df is None:
        st.info("Load a CSV or Excel file in the **Ingest Data** tab to visualize it here.")
    else:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import io, base64

        df = orch.data_agent.df
        numeric_cols = list(df.select_dtypes(include="number").columns)
        cat_cols     = list(df.select_dtypes(include=["object","category"]).columns)
        all_cols     = list(df.columns)

        st.markdown(f"**{st.session_state.data_filename}** · {df.shape[0]:,} rows × {df.shape[1]} cols")
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("---")
        viz_type = st.selectbox("Chart type", ["Bar","Line","Histogram","Scatter","Heatmap (correlation)","Box"], key="viz_type")

        def dark_fig(figsize=(10,4)):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("#0f1117")
            ax.set_facecolor("#1a1d2e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for sp in ax.spines.values(): sp.set_edgecolor("#333")
            return fig, ax

        def show_fig(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close(fig)

        if viz_type == "Bar":
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X (category)", cat_cols or all_cols, key="bar_x")
            y_col = c2.selectbox("Y (numeric)",  numeric_cols or all_cols, key="bar_y")
            if st.button("Generate Bar Chart", use_container_width=True):
                data = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(20)
                fig, ax = dark_fig()
                ax.bar(data.index.astype(str), data.values, color="#7c6df2", alpha=0.85)
                ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"{y_col} by {x_col}")
                plt.xticks(rotation=35, ha="right")
                show_fig(fig)

        elif viz_type == "Line":
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X", all_cols, key="line_x")
            y_col = c2.selectbox("Y (numeric)", numeric_cols or all_cols, key="line_y")
            if st.button("Generate Line Chart", use_container_width=True):
                fig, ax = dark_fig()
                ax.plot(df[x_col], df[y_col], color="#06b6d4", linewidth=2)
                ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"{y_col} over {x_col}")
                show_fig(fig)

        elif viz_type == "Histogram":
            col = st.selectbox("Column", numeric_cols or all_cols, key="hist_col")
            bins = st.slider("Bins", 5, 100, 30, key="hist_bins")
            if st.button("Generate Histogram", use_container_width=True):
                fig, ax = dark_fig()
                ax.hist(df[col].dropna(), bins=bins, color="#3b82f6", alpha=0.85, edgecolor="#1e3a5f")
                ax.set_xlabel(col); ax.set_title(f"Distribution of {col}")
                show_fig(fig)

        elif viz_type == "Scatter":
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X", numeric_cols or all_cols, key="sc_x")
            y_col = c2.selectbox("Y", numeric_cols or all_cols, key="sc_y")
            c_col = c3.selectbox("Color by (optional)", ["None"] + cat_cols, key="sc_c")
            if st.button("Generate Scatter", use_container_width=True):
                fig, ax = dark_fig()
                if c_col != "None" and c_col in df.columns:
                    groups = df[c_col].unique()[:10]
                    colors = plt.cm.tab10.colors
                    for i, g in enumerate(groups):
                        sub = df[df[c_col] == g]
                        ax.scatter(sub[x_col], sub[y_col], label=str(g), color=colors[i % 10], alpha=0.7, s=15)
                    ax.legend(fontsize=7, labelcolor="white", framealpha=0.2)
                else:
                    ax.scatter(df[x_col], df[y_col], color="#f59e0b", alpha=0.6, s=15)
                ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"{x_col} vs {y_col}")
                show_fig(fig)

        elif viz_type == "Heatmap (correlation)":
            if st.button("Generate Correlation Heatmap", use_container_width=True):
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns.")
                else:
                    corr = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols)-1)))
                    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d2e")
                    sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=True, fmt=".2f",
                                linewidths=0.5, linecolor="#1e3a5f",
                                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
                    ax.tick_params(colors="white", labelsize=8)
                    ax.set_title("Correlation Matrix", color="white")
                    show_fig(fig)

        elif viz_type == "Box":
            c1, c2 = st.columns(2)
            y_col = c1.selectbox("Numeric column", numeric_cols or all_cols, key="box_y")
            x_col = c2.selectbox("Group by (optional)", ["None"] + cat_cols, key="box_x")
            if st.button("Generate Box Plot", use_container_width=True):
                fig, ax = dark_fig()
                if x_col != "None" and x_col in df.columns:
                    groups = [df[df[x_col] == g][y_col].dropna().values for g in df[x_col].unique()[:15]]
                    labels = [str(g) for g in df[x_col].unique()[:15]]
                    bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                                   medianprops={"color":"#4ade80","linewidth":2})
                    for patch in bp["boxes"]:
                        patch.set_facecolor("#1e3a5f"); patch.set_alpha(0.8)
                    plt.xticks(rotation=35, ha="right")
                else:
                    ax.boxplot(df[y_col].dropna().values, patch_artist=True,
                               medianprops={"color":"#4ade80","linewidth":2})
                ax.set_ylabel(y_col); ax.set_title(f"Box Plot: {y_col}")
                show_fig(fig)

        st.markdown("---")
        st.markdown("**🤖 AI Chart Generator**")
        ai_chart_q = st.text_input(
            "Describe a chart you want (AI will generate it)",
            placeholder="e.g. show me a bar chart of top 10 products by sales",
            key="ai_chart_input",
        )
        if st.button("🎨 Generate with AI", key="ai_chart_btn", use_container_width=True):
            if ai_chart_q.strip():
                with st.spinner("AI is generating your chart…"):
                    result = orch.data_agent.analyze(ai_chart_q)
                st.markdown(result["answer"])
                if result.get("chart"):
                    st.image(base64.b64decode(result["chart"]), use_container_width=True)
            else:
                st.warning("Please describe the chart you want.")


# ════════════════════════════════════════════════════════════
# TAB 4 — VECTOR DB SHOWCASE
# ════════════════════════════════════════════════════════════
with tab_vdb:
    from vectordb_showcase import render_vectordb_showcase, render_routing_log

    st.markdown("### ◈ Vector DB Showcase")
    st.markdown(
        "Inspect indexed chunks, embedding metadata, and retrieval similarity scores.",
    )

    vdb_tab1, vdb_tab2, vdb_tab3 = st.tabs(["📄 Document RAG", "🎬 YouTube RAG", "⟳ Routing Log"])

    with vdb_tab1:
        if st.session_state.rag_ingested:
            vd = orch.rag.get_showcase_data()
            if vd.get("total_chunks", 0) > 0:
                render_vectordb_showcase(vd, title="Document RAG · Vector Store")
            else:
                st.info("Documents ingested but no chunks found. Try re-ingesting.")
        else:
            st.markdown("""
<div style="text-align:center;padding:2.5rem 1rem;color:#9ca3af;font-family:'Space Mono',monospace;font-size:0.75rem">
  ◈<br><br>
  Upload documents in <b>Ingest Data → Documents</b><br>
  then ask a question to see retrieval in action.
</div>""", unsafe_allow_html=True)

    with vdb_tab2:
        if st.session_state.video_ingested and orch.video_rag.is_ready():
            vd = orch.video_rag.get_showcase_data()
            if vd.get("total_chunks", 0) > 0:
                render_vectordb_showcase(vd, title="YouTube RAG · Vector Store")
            else:
                st.info("Video loaded but no chunks indexed.")
        else:
            st.markdown("""
<div style="text-align:center;padding:2.5rem 1rem;color:#9ca3af;font-family:'Space Mono',monospace;font-size:0.75rem">
  🎬<br><br>
  Load a YouTube video in <b>Ingest Data → YouTube</b><br>
  then ask a question to see timestamps and similarity scores.
</div>""", unsafe_allow_html=True)

    with vdb_tab3:
        log = orch.get_routing_log()
        if log:
            st.markdown("Every message routed through the orchestrator appears here.")
            render_routing_log(log)
            st.markdown("---")
            methods = [e["method"] for e in log]
            st.markdown("**Routing method breakdown:**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Regex (fast)", methods.count("regex"))
            c2.metric("LLM classify", methods.count("llm"))
            c3.metric("Context override", methods.count("context_override"))
        else:
            st.markdown("""
<div style="text-align:center;padding:2.5rem 1rem;color:#9ca3af;font-family:'Space Mono',monospace;font-size:0.75rem">
  ⟳<br><br>
  Send a message using <b>Auto-Route</b> mode<br>
  to see how the orchestrator routes your queries.
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### ⬡ NexusRAG — Multi-Agent Intelligence Platform

**Stack**
- **LLM**: Google Gemini (via `google-generativeai`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384d, CPU)
- **Vector Store**: FAISS IndexFlatL2 with hybrid rerank
- **Framework**: LangChain · Streamlit

---

**Agents**

| Agent | Capability |
|-------|-----------|
| 🤖 Chatbot | Conversational AI with rolling memory (20 turns) |
| 📄 Document RAG | Hybrid retrieval + contextual compression + grounded answers |
| 🎬 YouTube RAG | Transcript fetch → chunk → semantic Q&A with timestamps |
| 📊 Data Analyst | Pandas profiling + AI chart spec generation |
| 💻 Code Generator | Multi-language code gen with inline comments |
| 🔬 Web Researcher | DuckDuckGo multi-query synthesis |
| 🧠 Auto-Route | Regex fast-path → LLM classifier → context override |

---

**RAG Pipeline**
```
Upload → Load docs → Chunk (800 chars, 150 overlap)
       → Embed (MiniLM-L6-v2) → FAISS index
Query  → Dense search (k=6) → Keyword rerank → Top 5
       → Contextual compression (LLM) → Grounded answer
```

**Orchestrator routing**
```
Message → Regex fast-path (instant, no LLM cost)
        → LLM classifier (returns agent + confidence 0-1)
        → Context override (if docs/video/data loaded)
        → Dispatch → Agent → Result + routing metadata
```

---

**Secrets (`.streamlit/secrets.toml`)**
```toml
GEMINI_API_KEY = "your-key-here"
GEMINI_MODEL   = "gemini-2.0-flash"
```

---

**Keyboard Shortcuts**
- `Enter` — Send message
- `Shift + Enter` — New line in message box
""")
