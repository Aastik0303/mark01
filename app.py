"""
NexusRAG — Multi-Agent Intelligence Platform
Streamlit Frontend · Google Gemini · LangChain · FAISS

FIXES:
  - All imports at top-level (never inside tab blocks — was causing crash)
  - Nested st.tabs() replaced with st.radio() — fixes frozen tab bar
  - Tab bar CSS: flex-wrap:nowrap + overflow-x:auto — never freezes
  - SVG-compatible page_icon
  - Enter-to-send JS
  - Sidebar permanently pinned open
"""

import streamlit as st
import os, sys, json, tempfile, base64, re, io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ALL third-party imports at TOP — never inside tab/with blocks ─────────────
try:
    import pandas as pd
    PD_OK = True
except ImportError:
    PD_OK = False

try:
    import seaborn as sns
    SNS_OK = True
except ImportError:
    SNS_OK = False

# ── Safe import of our modules ────────────────────────────────────────────────
try:
    from vectordb_showcase import render_vectordb_showcase, render_routing_log
    VDB_OK = True
except Exception:
    VDB_OK = False
    def render_vectordb_showcase(data, title=""):
        st.info("vectordb_showcase.py not found in project folder.")
    def render_routing_log(log):
        st.info("vectordb_showcase.py not found in project folder.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be FIRST Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NexusRAG · Multi-Agent AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #070b14;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0f1f3d 0%, #070b14 50%, #0a0514 100%);
}

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #3b5bdb; border-radius: 2px; }

/* ── SIDEBAR — permanently pinned, no collapse button ── */
[data-testid="stSidebarCollapseButton"]   { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0f1e 0%,#070b14 100%) !important;
    border-right: 1px solid #1e2d4a !important;
    min-width: 270px !important;
    max-width: 270px !important;
    transform: none !important;
    visibility: visible !important;
}

/* ── TAB BAR — KEY FIX: scrollable, never wraps, never freezes ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0f1e !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 3px !important;
    border: 1px solid #1e2d4a !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 0.7rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#3b5bdb,#7c6df2) !important;
    color: #fff !important;
}

/* buttons */
.stButton > button {
    background: linear-gradient(135deg,#1a2744 0%,#111827 100%) !important;
    color: #c7d2fe !important; border: 1px solid #1e3a5f !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; text-align: left !important;
    transition: all 0.2s ease !important; padding: 0.6rem 1rem !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #7c6df2 !important; color: #fff !important;
    box-shadow: 0 4px 16px rgba(124,109,242,0.25) !important;
    transform: translateY(-1px) !important;
}

/* chat bubbles */
.bubble-user {
    background: linear-gradient(135deg,#1e2d4a,#1a1f2e);
    border: 1px solid #2d4a7a; border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1.1rem; max-width: 75%; color: #c7d2fe;
    font-size: 0.88rem; line-height: 1.5;
    margin: 0.4rem 0 0.4rem auto; word-break: break-word;
}
.bubble-agent {
    background: linear-gradient(135deg,#111827,#0d1526);
    border: 1px solid #1e3a5f; border-radius: 4px 16px 16px 16px;
    padding: 0.75rem 1.1rem; max-width: 85%; color: #e2e8f0;
    font-size: 0.88rem; line-height: 1.6;
    margin: 0.4rem 0; word-break: break-word;
}
.bubble-label {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.3rem; opacity: 0.7;
}
.bubble-user  .bubble-label { color: #3b82f6; }
.bubble-agent .bubble-label { color: #7c6df2; }

/* inputs */
.stTextArea textarea {
    background: #0a0f1e !important; border: 1px solid #1e3a5f !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important; font-size: 0.9rem !important; resize: none !important;
}
.stTextArea textarea:focus { border-color: #7c6df2 !important; box-shadow: 0 0 0 2px rgba(124,109,242,0.2) !important; }
.stTextInput input {
    background: #0a0f1e !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #e2e8f0 !important; font-size: 0.9rem !important;
}
.stTextInput input:focus { border-color: #7c6df2 !important; }
.stSelectbox > div > div {
    background: #0a0f1e !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}
[data-testid="stFileUploader"] {
    background: #0a0f1e; border: 1px dashed #1e3a5f; border-radius: 12px; padding: 1rem;
}

/* expander */
.streamlit-expanderHeader {
    background: #0a0f1e !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #7c6df2 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important;
}

/* alerts */
hr { border-color: #1e2d4a !important; }
.stSuccess { background: #052e16 !important; border-left: 4px solid #10b981 !important; border-radius: 6px !important; }
.stError   { background: #2d0a0a !important; border-left: 4px solid #ef4444 !important; border-radius: 6px !important; }
.stInfo    { background: #0c1a2e !important; border-left: 4px solid #3b82f6 !important; border-radius: 6px !important; }
.stWarning { background: #2d1a00 !important; border-left: 4px solid #f59e0b !important; border-radius: 6px !important; }
pre { background: #060a12 !important; border: 1px solid #1e3a5f !important; border-radius: 8px !important; }

/* badges */
.badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #0d1526; border: 1px solid #1e3a5f; border-radius: 20px;
    padding: 0.2rem 0.7rem; font-family: 'Space Mono', monospace;
    font-size: 0.65rem; color: #64748b; margin: 0.1rem;
}
.badge.on   { border-color: #10b981; color: #10b981; background: #052e16; }
.badge.off  { border-color: #374151; color: #374151; }
.badge.info { border-color: #3b82f6; color: #3b82f6; }

/* misc */
.src-pill {
    display: inline-block; background: #0d1526; border: 1px solid #1e3a5f;
    border-radius: 4px; padding: 0.1rem 0.5rem;
    font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #64748b; margin: 0.1rem;
}
.ts-pill {
    display: inline-block; background: #0d1f3a; border: 1px solid #3b5bdb;
    border-radius: 4px; padding: 0.15rem 0.6rem;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #7c6df2;
    margin: 0.15rem; text-decoration: none;
}
.agent-bar {
    display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.8rem;
    background: #0a0f1e; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 0.7rem 1rem; flex-wrap: wrap;
}
.agent-bar-icon { font-size: 1.4rem; }
.agent-bar-name { font-weight: 700; font-size: 0.95rem; color: #c7d2fe; }
.agent-bar-desc { font-size: 0.72rem; color: #475569; font-family: 'Space Mono', monospace; margin-top: 0.1rem; }
.delegate-tag {
    display: inline-block; background: #2d1a00; border: 1px solid #f59e0b;
    color: #f59e0b; border-radius: 4px; padding: 0.05rem 0.4rem;
    font-family: 'Space Mono', monospace; font-size: 0.6rem; margin-left: 0.4rem;
}
.info-card {
    padding: 0.8rem; background: #0a1628; border-radius: 8px;
    border-left: 3px solid #7c6df2; margin-top: 0.5rem;
}
.info-card-title { color: #e2e8f0; font-weight: 600; }
.info-card-sub   { color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem; }
.conf-pill {
    display: inline-flex; align-items: center; gap: 0.25rem;
    font-family: 'Space Mono', monospace; font-size: 0.58rem;
    border-radius: 4px; padding: 0.05rem 0.35rem; border: 1px solid; margin-left: 0.4rem;
}
.conf-high { border-color: #10b981; color: #10b981; background: #052e16; }
.conf-mid  { border-color: #f59e0b; color: #f59e0b; background: #2d1a00; }
.conf-low  { border-color: #ef4444; color: #ef4444; background: #2d0a0a; }
.input-hint {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    color: #374151; margin-top: 0.25rem; padding-left: 0.1rem;
}
.nexus-header { text-align: center; padding: 0.8rem 0 0.6rem; }
.nexus-title {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1;
    background: linear-gradient(135deg,#7c6df2 0%,#3b82f6 50%,#06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.nexus-sub {
    font-family: 'Space Mono', monospace; color: #475569; font-size: 0.7rem;
    letter-spacing: 0.14em; text-transform: uppercase; margin-top: 0.3rem;
}

/* radio used as sub-nav */
.stRadio > div { flex-wrap: nowrap !important; gap: 0.4rem; }
.stRadio label {
    font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important;
    color: #64748b !important; background: #0a0f1e !important;
    border: 1px solid #1e3a5f !important; border-radius: 6px !important;
    padding: 0.3rem 0.8rem !important; white-space: nowrap !important;
}

/* mobile */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] { font-size: 0.7rem !important; padding: 0.3rem 0.4rem !important; }
    .stButton > button { min-height: 46px !important; }
    .bubble-user, .bubble-agent { max-width: 98% !important; font-size: 0.82rem !important; }
    .nexus-title { font-size: 1.75rem !important; }
    .agent-bar-desc { display: none !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = st.secrets.get("GEMINI_MODEL",   "gemini-2.0-flash")

_ss("agents_ready",     False)
_ss("orchestrator",     None)
_ss("active_agent",     "chat")
_ss("messages",         [])
_ss("rag_ingested",     False)
_ss("video_ingested",   False)
_ss("data_loaded",      False)
_ss("data_filename",    "")
_ss("data_shape",       "")
_ss("data_columns",     [])
_ss("_boot_error",      "")
_ss("video_url_saved",  "")
_ss("video_lang_saved", "en")
_ss("last_vstore_data", None)


# ══════════════════════════════════════════════════════════════════════════════
# INIT AGENTS
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
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

def _conf_class(conf):
    if conf >= 0.8: return "conf-high"
    if conf >= 0.5: return "conf-mid"
    return "conf-low"

def dark_fig(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d2e")
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

def render_message(msg):
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
        del_tag    = '<span class="delegate-tag">↗ delegated</span>' if msg.get("delegated") else ""
        routing    = msg.get("routing") or {}
        conf_badge = ""
        if routing:
            conf   = routing.get("confidence", 0)
            method = routing.get("method", "")
            cc     = _conf_class(conf)
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
            with st.expander("◈ Vector DB — Retrieved Chunks", expanded=False):
                render_vectordb_showcase(msg["vstore_data"])

def inject_enter_js():
    st.markdown("""
<script>
(function(){
  function attach(){
    window.parent.document.querySelectorAll('textarea').forEach(function(ta){
      if(ta._nexusOk) return;
      ta._nexusOk = true;
      ta.addEventListener('keydown', function(e){
        if(e.key==='Enter' && !e.shiftKey){
          e.preventDefault();
          window.parent.document.querySelectorAll('button').forEach(function(b){
            if(b.innerText && (b.innerText.includes('\u2b06') || b.innerText.trim()==='Send')){
              b.click();
            }
          });
        }
      });
    });
  }
  attach();
  new MutationObserver(attach).observe(window.parent.document.body,{childList:true,subtree:true});
})();
</script>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem">
  <div style="font-size:1.8rem;font-weight:800;
    background:linear-gradient(135deg,#7c6df2,#3b82f6,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;letter-spacing:-0.02em">⬡ NEXUSRAG</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
    color:#475569;letter-spacing:0.15em;margin-top:0.2rem">MULTI-AGENT AI</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.agents_ready:
        st.markdown(f'<span class="badge on">● ONLINE · {GEMINI_MODEL}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge off">● OFFLINE</span>', unsafe_allow_html=True)
        if st.button("🔄 Retry", use_container_width=True):
            try: init_agents(); st.rerun()
            except Exception as e: st.error(str(e))

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
                render_routing_log(log)

    if st.session_state.last_vstore_data:
        vd = st.session_state.last_vstore_data
        st.markdown("---")
        st.markdown("**◈ Vector DB**")
        st.markdown(
            f'<span class="badge info">🗂 {vd.get("total_chunks",0)} chunks</span> '
            f'<span class="badge info">📐 {vd.get("embedding_dim",0)}d</span>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nexus-header">
  <div class="nexus-title">NEXUSRAG</div>
  <div class="nexus-sub">Multi-Agent Intelligence · Gemini · LangChain · FAISS</div>
</div>""", unsafe_allow_html=True)

if st.session_state._boot_error:
    st.error(f"⚠️ Boot error: {st.session_state._boot_error}")

if not st.session_state.agents_ready:
    st.error("⚠️ Could not initialise agents. Check your .streamlit/secrets.toml")
    if st.button("🔄 Retry Connection"):
        try: init_agents(); st.rerun()
        except Exception as e: st.error(str(e))
    st.stop()

orch   = st.session_state.orchestrator
active = st.session_state.active_agent

# Auto-restore video on server restart
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS — top-level ONLY, NO nested st.tabs() anywhere in this file
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_ingest, tab_viz, tab_vdb, tab_about = st.tabs([
    "💬 Chat", "📥 Ingest", "📊 Visualize", "◈ Vector DB", "ℹ️ About"
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

    # Message history
    if not st.session_state.messages:
        st.markdown("""
<div style="text-align:center;padding:3rem 1rem;opacity:0.35;
  font-family:'Space Mono',monospace;font-size:0.78rem;color:#475569">
  ⬡<br><br>Start a conversation · Upload documents · Load a YouTube video<br>
  <span style="font-size:0.65rem">Select an agent from the sidebar or use Auto-Route</span>
</div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(msg)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # YouTube URL bar (video agent only)
    if active == "video":
        c_url, c_lang, c_btn = st.columns([4, 1, 1])
        yt_url  = c_url.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...",
                                   key="yt_url_input", label_visibility="collapsed")
        yt_lang = c_lang.selectbox("Lang", ["en","hi","es","fr","de","ja","ko","pt","ar","ru"],
                                   key="yt_lang_sel", label_visibility="collapsed")
        if c_btn.button("▶ Load", key="yt_load_btn", use_container_width=True):
            if yt_url.strip():
                with st.spinner("Loading video…"):
                    r = orch.video_rag.ingest(yt_url.strip(), language=yt_lang)
                if "Error" not in r:
                    st.session_state.video_ingested   = True
                    st.session_state.video_url_saved  = yt_url.strip()
                    st.session_state.video_lang_saved = yt_lang
                    st.session_state.last_vstore_data = orch.video_rag.get_showcase_data()
                    st.success(r)
                else:
                    st.error(r)
            else:
                st.warning("Please enter a YouTube URL.")

    # Code language selector
    if active == "code":
        code_lang = st.selectbox(
            "Language",
            ["python","javascript","typescript","java","go","rust","sql","bash","c++","c#","php","kotlin"],
            key="code_lang_sel",
        )

    # Input row
    placeholder_map = {
        "chat":     "Ask anything… (Enter to send · Shift+Enter for new line)",
        "rag":      "Ask about your uploaded documents…",
        "video":    "Ask about the loaded video…",
        "data":     "Ask about your data…",
        "code":     "Describe what code you need…",
        "research": "What do you want to research?",
        "auto":     "Ask anything — I'll route to the best agent…",
    }
    col_in, col_send = st.columns([6, 1])
    with col_in:
        user_input = st.text_area(
            "Message",
            placeholder=placeholder_map.get(active, "Type your message…"),
            height=80, key="chat_input", label_visibility="collapsed",
        )
        st.markdown('<div class="input-hint">↵ Enter to send · Shift+↵ new line</div>',
                    unsafe_allow_html=True)
    with col_send:
        st.markdown("<div style='height:1.15rem'></div>", unsafe_allow_html=True)
        send = st.button("⬆ Send", key="send_btn", use_container_width=True)

    inject_enter_js()

    # Handle send
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
                    if vd: st.session_state.last_vstore_data = vd
                    push_msg("assistant", result.get("answer", ""),
                             agent=agent_id.upper(), chart=result.get("chart"),
                             code=result.get("code"), lang=result.get("lang","python"),
                             sources=result.get("sources",[]),
                             research_sources=result.get("sources",[]) if agent_id=="research" else [],
                             queries=result.get("queries",[]),
                             timestamps=result.get("timestamps",[]),
                             vstore_data=vd, routing=routing, delegated=True)

                elif active == "rag":
                    if not st.session_state.rag_ingested:
                        push_msg("assistant",
                                 "⚠️ No documents loaded. Upload files in the **Ingest** tab.",
                                 agent="RAG")
                    else:
                        result = orch.rag.query(q)
                        vd = result.get("vstore_data")
                        if vd: st.session_state.last_vstore_data = vd
                        push_msg("assistant", result["answer"], agent="RAG",
                                 sources=result.get("sources",[]), vstore_data=vd)

                elif active == "video":
                    if not st.session_state.video_ingested:
                        push_msg("assistant",
                                 "⚠️ No video loaded. Paste a YouTube URL above and click Load.",
                                 agent="VIDEO RAG")
                    else:
                        result = orch.video_rag.query(q)
                        vd = result.get("vstore_data")
                        if vd: st.session_state.last_vstore_data = vd
                        push_msg("assistant", result["answer"], agent="VIDEO RAG",
                                 timestamps=result.get("timestamps",[]), vstore_data=vd)

                elif active == "data":
                    if not st.session_state.data_loaded:
                        push_msg("assistant",
                                 "⚠️ No data loaded. Upload a CSV/Excel in the **Ingest** tab.",
                                 agent="DATA")
                    else:
                        result = orch.data_agent.analyze(q)
                        push_msg("assistant", result["answer"], agent="DATA",
                                 chart=result.get("chart"))

                elif active == "code":
                    lang   = st.session_state.get("code_lang_sel", "python")
                    result = orch.code_agent.generate(q, language=lang)
                    push_msg("assistant", result["answer"], agent="CODE",
                             code=result.get("code",""), lang=lang)

                elif active == "research":
                    result = orch.research_agent.research(q)
                    push_msg("assistant", result["answer"], agent="RESEARCH",
                             research_sources=result.get("sources",[]),
                             queries=result.get("queries",[]))

                else:  # chat
                    push_msg("assistant", orch.chatbot.chat(q), agent="NEXUS")

            except Exception as e:
                push_msg("assistant", f"⚠️ Error: {e}", agent="SYSTEM")
        st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 2 — INGEST  (uses st.radio — NO nested st.tabs)
# ════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown("### 📥 Ingest Data")

    ingest_choice = st.radio(
        "Source",
        ["📄 Documents", "🎬 YouTube", "📊 CSV / Excel"],
        horizontal=True,
        key="ingest_radio",
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Documents
    if ingest_choice == "📄 Documents":
        st.markdown("Upload **PDF, TXT, MD, or CSV** files to enable Document Q&A.")
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
                    result_msg = orch.rag.ingest(paths)
                if "Error" not in result_msg:
                    st.session_state.rag_ingested = True
                    vd = orch.rag.get_showcase_data()
                    st.session_state.last_vstore_data = vd
                    st.success(result_msg)
                    render_vectordb_showcase(vd, title="Document Vector Store")
                else:
                    st.error(result_msg)
        elif st.session_state.rag_ingested:
            vd = orch.rag.get_showcase_data()
            if vd.get("total_chunks", 0) > 0:
                render_vectordb_showcase(vd, title="Document Vector Store")

    # YouTube
    elif ingest_choice == "🎬 YouTube":
        st.markdown("Load a YouTube video to enable transcript Q&A.")
        yt_url2  = st.text_input("YouTube URL",
                                 placeholder="https://youtube.com/watch?v=...",
                                 key="yt_url_ingest")
        yt_lang2 = st.selectbox("Transcript language",
                                ["en","hi","es","fr","de","ja","ko","pt","ar","ru"],
                                key="yt_lang_ingest")
        if st.button("▶ Load Video", key="load_yt_ingest", use_container_width=True):
            if yt_url2.strip():
                with st.spinner("Fetching transcript…"):
                    r = orch.video_rag.ingest(yt_url2.strip(), language=yt_lang2)
                if "Error" not in r:
                    st.session_state.video_ingested   = True
                    st.session_state.video_url_saved  = yt_url2.strip()
                    st.session_state.video_lang_saved = yt_lang2
                    vd = orch.video_rag.get_showcase_data()
                    st.session_state.last_vstore_data = vd
                    st.success(r)
                    info = orch.video_rag.get_info()
                    if info.get("thumbnail"):
                        st.image(info["thumbnail"], width=280)
                    render_vectordb_showcase(vd, title="YouTube Vector Store")
                else:
                    st.error(r)
            else:
                st.warning("Please enter a YouTube URL.")

        if st.session_state.video_ingested and orch.video_rag.is_ready():
            info = orch.video_rag.get_info()
            st.markdown(f"""
<div class="info-card">
  <div class="info-card-title">🎬 {info.get('title','Unknown')}</div>
  <div class="info-card-sub">
    {info.get('channel','?')} · {info.get('duration','?')} ·
    {info.get('transcript_segments',0)} segments · source: {info.get('source_type','?')}
  </div>
</div>""", unsafe_allow_html=True)
            sum_style = st.selectbox("Summary style",
                                    ["detailed","brief","bullets"], key="sum_style_sel")
            if st.button("📝 Summarize Video", key="sum_video_btn", use_container_width=True):
                with st.spinner("Summarizing…"):
                    s = orch.video_rag.summarize(style=sum_style)
                st.markdown(s.get("summary", ""))

    # CSV / Excel
    elif ingest_choice == "📊 CSV / Excel":
        st.markdown("Upload a **CSV or Excel** file for AI-powered data analysis.")
        data_file = st.file_uploader("Drop CSV or Excel here",
                                     type=["csv","xlsx","xls"], key="data_uploader")
        if st.button("📊 Load Data", key="load_data_btn", use_container_width=True):
            if not data_file:
                st.warning("Please upload a file.")
            else:
                with st.spinner("Loading…"):
                    tmp = tempfile.NamedTemporaryFile(delete=False,
                                                      suffix=Path(data_file.name).suffix)
                    tmp.write(data_file.read()); tmp.flush()
                    r = orch.data_agent.load_data(tmp.name)
                if "Error" not in r:
                    st.session_state.data_loaded   = True
                    st.session_state.data_filename = data_file.name
                    if orch.data_agent.df is not None:
                        df = orch.data_agent.df
                        st.session_state.data_shape   = f"{df.shape[0]:,} × {df.shape[1]}"
                        st.session_state.data_columns = list(df.columns)
                    st.success(r)
                    if orch.data_agent.df is not None:
                        st.dataframe(orch.data_agent.df.head(10), use_container_width=True)
                else:
                    st.error(r)
        elif st.session_state.data_loaded and orch.data_agent.df is not None:
            st.info(f"Loaded: **{st.session_state.data_filename}** · {st.session_state.data_shape}")
            st.dataframe(orch.data_agent.df.head(10), use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — VISUALIZE
# ════════════════════════════════════════════════════════════
with tab_viz:
    st.markdown("### 📊 Data Visualizer")

    if not st.session_state.data_loaded or orch.data_agent.df is None:
        st.info("Load a CSV or Excel file in the **Ingest** tab to visualize it here.")
    elif not PD_OK:
        st.error("pandas not installed.")
    else:
        df           = orch.data_agent.df
        numeric_cols = list(df.select_dtypes(include="number").columns)
        cat_cols     = list(df.select_dtypes(include=["object","category"]).columns)
        all_cols     = list(df.columns)

        st.markdown(f"**{st.session_state.data_filename}** · "
                    f"{df.shape[0]:,} rows × {df.shape[1]} cols")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown("---")

        viz_type = st.selectbox(
            "Chart type",
            ["Bar","Line","Histogram","Scatter","Heatmap (correlation)","Box"],
            key="viz_type",
        )

        if viz_type == "Bar":
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X (category)", cat_cols or all_cols, key="bar_x")
            y_col = c2.selectbox("Y (numeric)",  numeric_cols or all_cols, key="bar_y")
            if st.button("Generate Bar Chart", use_container_width=True):
                data = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(20)
                fig, ax = dark_fig()
                ax.bar(data.index.astype(str), data.values, color="#7c6df2", alpha=0.85)
                ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} by {x_col}")
                plt.xticks(rotation=35, ha="right"); show_fig(fig)

        elif viz_type == "Line":
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X", all_cols, key="line_x")
            y_col = c2.selectbox("Y (numeric)", numeric_cols or all_cols, key="line_y")
            if st.button("Generate Line Chart", use_container_width=True):
                fig, ax = dark_fig()
                ax.plot(df[x_col], df[y_col], color="#06b6d4", linewidth=2)
                ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} over {x_col}"); show_fig(fig)

        elif viz_type == "Histogram":
            col  = st.selectbox("Column", numeric_cols or all_cols, key="hist_col")
            bins = st.slider("Bins", 5, 100, 30, key="hist_bins")
            if st.button("Generate Histogram", use_container_width=True):
                fig, ax = dark_fig()
                ax.hist(df[col].dropna(), bins=bins, color="#3b82f6",
                        alpha=0.85, edgecolor="#1e3a5f")
                ax.set_xlabel(col)
                ax.set_title(f"Distribution of {col}"); show_fig(fig)

        elif viz_type == "Scatter":
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X", numeric_cols or all_cols, key="sc_x")
            y_col = c2.selectbox("Y", numeric_cols or all_cols, key="sc_y")
            c_col = c3.selectbox("Color by", ["None"] + cat_cols, key="sc_c")
            if st.button("Generate Scatter", use_container_width=True):
                fig, ax = dark_fig()
                if c_col != "None" and c_col in df.columns:
                    for i, g in enumerate(df[c_col].unique()[:10]):
                        sub = df[df[c_col] == g]
                        ax.scatter(sub[x_col], sub[y_col], label=str(g),
                                   color=plt.cm.tab10.colors[i % 10], alpha=0.7, s=15)
                    ax.legend(fontsize=7, labelcolor="white", framealpha=0.2)
                else:
                    ax.scatter(df[x_col], df[y_col], color="#f59e0b", alpha=0.6, s=15)
                ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                ax.set_title(f"{x_col} vs {y_col}"); show_fig(fig)

        elif viz_type == "Heatmap (correlation)":
            if st.button("Generate Correlation Heatmap", use_container_width=True):
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns.")
                elif not SNS_OK:
                    st.error("seaborn not installed — add it to requirements.txt")
                else:
                    corr = df[numeric_cols].corr()
                    sz   = max(6, len(numeric_cols))
                    fig, ax = plt.subplots(figsize=(sz, sz - 1))
                    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d2e")
                    sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=True, fmt=".2f",
                                linewidths=0.5, linecolor="#1e3a5f",
                                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
                    ax.tick_params(colors="white", labelsize=8)
                    ax.set_title("Correlation Matrix", color="white"); show_fig(fig)

        elif viz_type == "Box":
            c1, c2 = st.columns(2)
            y_col = c1.selectbox("Numeric column", numeric_cols or all_cols, key="box_y")
            x_col = c2.selectbox("Group by", ["None"] + cat_cols, key="box_x")
            if st.button("Generate Box Plot", use_container_width=True):
                fig, ax = dark_fig()
                if x_col != "None" and x_col in df.columns:
                    uniq   = df[x_col].unique()[:15]
                    groups = [df[df[x_col] == g][y_col].dropna().values for g in uniq]
                    bp = ax.boxplot(groups, labels=[str(g) for g in uniq],
                                   patch_artist=True,
                                   medianprops={"color":"#4ade80","linewidth":2})
                    for patch in bp["boxes"]:
                        patch.set_facecolor("#1e3a5f"); patch.set_alpha(0.8)
                    plt.xticks(rotation=35, ha="right")
                else:
                    ax.boxplot(df[y_col].dropna().values, patch_artist=True,
                               medianprops={"color":"#4ade80","linewidth":2})
                ax.set_ylabel(y_col)
                ax.set_title(f"Box Plot: {y_col}"); show_fig(fig)

        st.markdown("---")
        st.markdown("**🤖 AI Chart Generator**")
        ai_q = st.text_input("Describe a chart",
                             placeholder="e.g. bar chart of top 10 products by sales",
                             key="ai_chart_input")
        if st.button("🎨 Generate with AI", key="ai_chart_btn", use_container_width=True):
            if ai_q.strip():
                with st.spinner("Generating chart…"):
                    r = orch.data_agent.analyze(ai_q)
                st.markdown(r["answer"])
                if r.get("chart"):
                    st.image(base64.b64decode(r["chart"]), use_container_width=True)
            else:
                st.warning("Please describe the chart.")


# ════════════════════════════════════════════════════════════
# TAB 4 — VECTOR DB  (uses st.radio — NO nested st.tabs)
# ════════════════════════════════════════════════════════════
with tab_vdb:
    st.markdown("### ◈ Vector DB Showcase")
    st.markdown("Inspect indexed chunks, embeddings, and similarity scores in real time.")

    vdb_choice = st.radio(
        "View",
        ["📄 Document RAG", "🎬 YouTube RAG", "⟳ Routing Log"],
        horizontal=True,
        key="vdb_radio",
        label_visibility="collapsed",
    )
    st.markdown("---")

    if vdb_choice == "📄 Document RAG":
        if st.session_state.rag_ingested:
            vd = orch.rag.get_showcase_data()
            if vd.get("total_chunks", 0) > 0:
                render_vectordb_showcase(vd, title="Document RAG · Vector Store")
            else:
                st.info("Documents ingested but no chunks found. Try re-ingesting.")
        else:
            st.markdown("""
<div style="text-align:center;padding:2.5rem;opacity:0.35;
  font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569">
  ◈<br><br>Upload documents in <b>Ingest → Documents</b><br>
  then ask a question to see retrieval in action.
</div>""", unsafe_allow_html=True)

    elif vdb_choice == "🎬 YouTube RAG":
        if st.session_state.video_ingested and orch.video_rag.is_ready():
            vd = orch.video_rag.get_showcase_data()
            if vd.get("total_chunks", 0) > 0:
                render_vectordb_showcase(vd, title="YouTube RAG · Vector Store")
            else:
                st.info("Video loaded but no chunks indexed.")
        else:
            st.markdown("""
<div style="text-align:center;padding:2.5rem;opacity:0.35;
  font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569">
  🎬<br><br>Load a YouTube video in <b>Ingest → YouTube</b><br>
  then ask a question to see timestamps and similarity scores.
</div>""", unsafe_allow_html=True)

    elif vdb_choice == "⟳ Routing Log":
        log = orch.get_routing_log()
        if log:
            st.markdown("Every routed message appears here with method and confidence.")
            render_routing_log(log)
            st.markdown("---")
            methods = [e["method"] for e in log]
            c1, c2, c3 = st.columns(3)
            c1.metric("⚡ Regex (fast)",     methods.count("regex"))
            c2.metric("🧠 LLM classify",     methods.count("llm"))
            c3.metric("🔄 Context override", methods.count("context_override"))
        else:
            st.markdown("""
<div style="text-align:center;padding:2.5rem;opacity:0.35;
  font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569">
  ⟳<br><br>Use <b>Auto-Route</b> mode and send a message<br>
  to see how the orchestrator routes your queries.
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### ⬡ NexusRAG — Multi-Agent Intelligence Platform

**Stack**
- **LLM**: Google Gemini (`google-generativeai`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384d, CPU)
- **Vector Store**: FAISS `IndexFlatL2` + hybrid BM25 rerank
- **Framework**: LangChain · Streamlit

---

**Agents**

| Agent | Capability |
|-------|-----------|
| 🤖 Chatbot | Conversational AI with rolling memory (20 turns) |
| 📄 Document RAG | Dense + keyword rerank · contextual compression · grounded answers |
| 🎬 YouTube RAG | Transcript fetch → chunk → semantic Q&A with timestamps |
| 📊 Data Analyst | Pandas profiling + AI chart spec generation |
| 💻 Code Generator | Multi-language code gen with inline comments |
| 🔬 Web Researcher | DuckDuckGo multi-query synthesis |
| 🧠 Auto-Route | Regex fast-path → LLM classifier → context override |

---

**RAG Pipeline**
```
Upload → Load → Chunk (800 chars, 150 overlap)
       → Embed (MiniLM-L6-v2) → FAISS index
Query  → Dense search (k=6) → BM25 rerank → Top 5
       → LLM contextual compression → Grounded answer
```

**Orchestrator**
```
Message → Regex fast-path (zero LLM cost)
        → LLM classifier + confidence score (0–1)
        → Context override (doc/video/data loaded?)
        → Dispatch → Agent result + routing metadata
```

---

**`.streamlit/secrets.toml`**
```toml
GEMINI_API_KEY = "your-key-here"
GEMINI_MODEL   = "gemini-2.0-flash"
```

**Keyboard shortcuts**
- `Enter` — Send message
- `Shift + Enter` — New line in message box
""")
