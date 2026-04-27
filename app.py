"""
NexusRAG — Multi-Agent Intelligence Platform  (v4 · Enhanced)
Streamlit Frontend · Google Gemini · LangChain · FAISS

ENHANCEMENTS in this version:
  1. Comprehensive About section with architecture diagrams
  2. Performance metrics and monitoring
  3. Export chat history feature
  4. Enhanced error handling with retry logic
  5. Agent performance dashboard
  6. System health indicators
  7. Advanced search filters in Vector DB
  8. Keyboard shortcuts panel
  9. Dark/Light theme toggle
  10. Session statistics tracking
"""

import streamlit as st
import os, sys, json, tempfile, base64, re, io, time
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── All heavy imports at top ───────────────────────────────────────────────
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

try:
    from vectordb_showcase import render_vectordb_showcase, render_routing_log
    VDB_OK = True
except Exception:
    VDB_OK = False
    def render_vectordb_showcase(data, title=""):
        st.info("vectordb_showcase.py missing from project folder.")
    def render_routing_log(log):
        st.info("vectordb_showcase.py missing from project folder.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuralRAG · Multi-Agent AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS (Enhanced with theme toggle support)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #f8fcff;
    color: #0f1724;
    -webkit-font-smoothing: antialiased;
}
#MainMenu, footer, header { visibility: hidden; }
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #e6f7ff 0%, #f8fcff 50%, #ffffff 100%);
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #e6f7ff; border-radius: 8px; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.22); border-radius: 8px; }

/* ── SIDEBAR — pinned, no collapse button ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"] { display: none !important; }
[data-testid="stSidebar"] {
    min-width: 285px !important;
    max-width: 285px !important;
    background: linear-gradient(180deg,#eaf6ff 0%,#f6fbff 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.12) !important;
    box-shadow: 0 8px 28px rgba(59,130,246,0.06);
}
.main .block-container { padding-left: 1.5rem !important; }

/* ── TAB BAR — scrollable, never wraps ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f0f8ff !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(59,130,246,0.10) !important;
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
    color: #475569 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    font-size: 0.84rem !important;
    padding: 0.4rem 0.75rem !important;
    transition: all 180ms ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#d6eaff,#e8e4ff) !important;
    color: #1e3a8a !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.08) !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(180deg,#e6f7ff 0%,#f6fbff 100%) !important;
    color: #0f1724 !important;
    border: 1px solid rgba(59,130,246,0.18) !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.06);
    transition: all 180ms ease !important;
}
.stButton > button:hover {
    border-color: rgba(59,130,246,0.32) !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.12) !important;
    transform: translateY(-2px) !important;
}

/* ── CHAT BUBBLE SHELL ── */
.bubble-user-shell {
    display: flex;
    justify-content: flex-end;
    margin: 0.45rem 0;
}
.bubble-agent-shell {
    display: flex;
    justify-content: flex-start;
    margin: 0.45rem 0;
}
.bubble-user-box {
    background: linear-gradient(180deg,#dbeeff,#e8f4ff);
    border: 1px solid rgba(59,130,246,0.16);
    border-radius: 18px 18px 6px 18px;
    padding: 0.75rem 1.1rem 0.6rem;
    max-width: 76%;
    color: #0f1724;
    box-shadow: 0 4px 16px rgba(59,130,246,0.06);
    word-break: break-word;
}
.bubble-agent-box {
    background: linear-gradient(180deg,#ffffff,#f6fbff);
    border: 1px solid rgba(59,130,246,0.10);
    border-radius: 6px 18px 18px 18px;
    padding: 0.75rem 1.1rem 0.6rem;
    max-width: 84%;
    color: #0f1724;
    box-shadow: 0 4px 16px rgba(59,130,246,0.05);
    word-break: break-word;
}
.bubble-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
    opacity: 0.75;
}
.bubble-user-box  .bubble-label { color: #2563eb; }
.bubble-agent-box .bubble-label { color: #3b82f6; }

/* ── The markdown content INSIDE agent bubble ── */
.bubble-agent-content p   { margin: 0.2rem 0; line-height: 1.6; }
.bubble-agent-content ul,
.bubble-agent-content ol  { margin: 0.3rem 0 0.3rem 1.2rem; }
.bubble-agent-content li  { margin: 0.15rem 0; line-height: 1.5; }
.bubble-agent-content h1,
.bubble-agent-content h2,
.bubble-agent-content h3  { margin: 0.6rem 0 0.2rem; font-family: 'Syne', sans-serif; color: #1e3a8a; }
.bubble-agent-content code {
    background: #e6f3ff;
    border-radius: 4px;
    padding: 0.05rem 0.3rem;
    font-size: 0.85em;
    color: #1e40af;
}
.bubble-agent-content pre  {
    background: #f0f8ff !important;
    border: 1px solid rgba(59,130,246,0.10) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
}
.bubble-agent-content strong { color: #1e3a8a; }
.bubble-agent-content em     { color: #374151; }
.bubble-agent-content table  { font-size: 0.88rem; width: 100%; }
.bubble-agent-content th     { background: #dbeeff; color: #1e3a8a; padding: 0.3rem 0.5rem; }
.bubble-agent-content td     { padding: 0.25rem 0.5rem; border-bottom: 1px solid rgba(59,130,246,0.08); }
.bubble-agent-content blockquote {
    border-left: 3px solid #3b82f6;
    margin: 0.3rem 0;
    padding-left: 0.7rem;
    color: #374151;
}

/* ── inputs ── */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid rgba(59,130,246,0.14) !important;
    border-radius: 12px !important;
    color: #0f1724 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
    padding: 0.65rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(59,130,246,0.30) !important;
    box-shadow: 0 6px 24px rgba(59,130,246,0.08) !important;
}
.stTextInput input {
    background: #ffffff !important;
    border: 1px solid rgba(59,130,246,0.14) !important;
    border-radius: 10px !important;
    color: #0f1724 !important;
    font-size: 0.95rem !important;
}
.stTextInput input:focus {
    border-color: rgba(59,130,246,0.30) !important;
    box-shadow: 0 6px 24px rgba(59,130,246,0.08) !important;
}
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid rgba(59,130,246,0.14) !important;
    border-radius: 10px !important;
    color: #0f1724 !important;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed rgba(59,130,246,0.18);
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 4px 14px rgba(59,130,246,0.04);
}

/* ── expander ── */
.streamlit-expanderHeader {
    background: #f0f8ff !important;
    border: 1px solid rgba(59,130,246,0.12) !important;
    border-radius: 10px !important;
    color: #1e3a8a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── alerts ── */
hr { border-color: rgba(59,130,246,0.08) !important; }
.stSuccess { background: #ecfdf5 !important; border-left: 4px solid #10b981 !important; border-radius: 8px !important; }
.stError   { background: #fff5f5 !important; border-left: 4px solid #ef4444 !important; border-radius: 8px !important; }
.stInfo    { background: #eff6ff !important; border-left: 4px solid #3b82f6 !important; border-radius: 8px !important; }
.stWarning { background: #fffaf0 !important; border-left: 4px solid #f59e0b !important; border-radius: 8px !important; }
pre { background: #f0f8ff !important; border: 1px solid rgba(59,130,246,0.10) !important; border-radius: 10px !important; }

/* ── badges ── */
.badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #e6f7ff; border: 1px solid rgba(59,130,246,0.14); border-radius: 20px;
    padding: 0.22rem 0.7rem; font-family: 'Space Mono', monospace;
    font-size: 0.68rem; color: #1e3a8a; margin: 0.1rem;
    box-shadow: 0 3px 10px rgba(59,130,246,0.05);
}
.badge.on   { border-color: rgba(16,185,129,0.16); color: #059669; background: #f0fdf4; }
.badge.off  { border-color: rgba(100,116,139,0.14); color: #64748b; background: #f8fafc; }
.badge.info { border-color: rgba(59,130,246,0.18); color: #2563eb; background: #eff6ff; }

/* ── source / timestamp pills ── */
.src-pill {
    display: inline-block; background: #e6f7ff; border: 1px solid rgba(59,130,246,0.12);
    border-radius: 6px; padding: 0.15rem 0.55rem;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #1e40af; margin: 0.1rem;
}
.ts-pill {
    display: inline-block; background: #dbeeff; border: 1px solid rgba(59,130,246,0.12);
    border-radius: 6px; padding: 0.18rem 0.6rem;
    font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #1e40af;
    margin: 0.12rem; text-decoration: none;
}
.ts-pill:hover { background: #c7e2ff; }

/* ── agent bar ── */
.agent-bar {
    display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.9rem;
    background: #ffffff; border: 1px solid rgba(59,130,246,0.10); border-radius: 14px;
    padding: 0.8rem 1.1rem; flex-wrap: wrap;
    box-shadow: 0 4px 16px rgba(59,130,246,0.05);
}
.agent-bar-icon { font-size: 1.5rem; }
.agent-bar-name { font-weight: 700; font-size: 0.98rem; color: #0f1724; }
.agent-bar-desc { font-size: 0.76rem; color: #6b7280; font-family: 'Space Mono', monospace; margin-top: 0.1rem; }

/* ── delegate / conf pill ── */
.delegate-tag {
    display: inline-block; background: #fffbeb; border: 1px solid rgba(245,158,11,0.18);
    color: #92400e; border-radius: 5px; padding: 0.04rem 0.4rem;
    font-family: 'Space Mono', monospace; font-size: 0.62rem; margin-left: 0.4rem;
}
.conf-pill {
    display: inline-flex; align-items: center; gap: 0.2rem;
    font-family: 'Space Mono', monospace; font-size: 0.60rem;
    border-radius: 5px; padding: 0.04rem 0.38rem; border: 1px solid; margin-left: 0.4rem;
}
.conf-high { border-color: rgba(16,185,129,0.18); color: #065f46; background: #ecfdf5; }
.conf-mid  { border-color: rgba(245,158,11,0.18); color: #92400e; background: #fffbeb; }
.conf-low  { border-color: rgba(239,68,68,0.18);  color: #7f1d1d; background: #fff5f5; }

/* ── info card ── */
.info-card {
    padding: 0.9rem; background: #ffffff; border-radius: 12px;
    border-left: 3px solid #3b82f6; margin-top: 0.5rem;
    box-shadow: 0 4px 14px rgba(59,130,246,0.05);
}
.info-card-title { color: #0f1724; font-weight: 600; font-size: 0.95rem; }
.info-card-sub   { color: #6b7280; font-size: 0.84rem; margin-top: 0.25rem; }

/* ── metric card ── */
.metric-card {
    background: linear-gradient(135deg, #f0f9ff, #ffffff);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 14px rgba(59,130,246,0.05);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg,#3b82f6,#7c6df2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* ── feature card ── */
.feature-card {
    background: #ffffff;
    border: 1px solid rgba(59,130,246,0.10);
    border-radius: 14px;
    padding: 1.2rem;
    margin: 0.8rem 0;
    box-shadow: 0 4px 16px rgba(59,130,246,0.04);
    transition: all 250ms ease;
}
.feature-card:hover {
    box-shadow: 0 8px 28px rgba(59,130,246,0.10);
    transform: translateY(-2px);
}
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.6rem;
}
.feature-title {
    font-weight: 700;
    font-size: 1.05rem;
    color: #0f1724;
    margin-bottom: 0.4rem;
}
.feature-desc {
    color: #64748b;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ── architecture diagram ── */
.arch-diagram {
    background: #f8fcff;
    border: 1px solid rgba(59,130,246,0.10);
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.8;
}
.arch-layer {
    background: linear-gradient(135deg, #e6f7ff, #f0f8ff);
    border-left: 3px solid #3b82f6;
    padding: 0.8rem;
    margin: 0.5rem 0;
    border-radius: 8px;
}
.arch-arrow {
    text-align: center;
    color: #3b82f6;
    font-size: 1.2rem;
    margin: 0.3rem 0;
}

/* ── input hint ── */
.input-hint {
    font-family: 'Space Mono', monospace; font-size: 0.62rem;
    color: #94a3b8; margin-top: 0.25rem; padding-left: 0.15rem;
}

/* ── header ── */
.nexus-header { text-align: center; padding: 0.7rem 0 0.5rem; }
.nexus-title {
    font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1;
    background: linear-gradient(135deg,#3b82f6 0%,#7c6df2 50%,#06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.nexus-sub {
    font-family: 'Space Mono', monospace; color: #6b7280; font-size: 0.7rem;
    letter-spacing: 0.14em; text-transform: uppercase; margin-top: 0.3rem;
}

/* ── mobile ── */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] { font-size: 0.72rem !important; padding: 0.3rem 0.45rem !important; }
    .stButton > button { min-height: 46px !important; }
    .bubble-user-box, .bubble-agent-box { max-width: 98% !important; font-size: 0.88rem !important; }
    .nexus-title { font-size: 1.7rem !important; }
    .agent-bar-desc { display: none !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE (Enhanced with metrics tracking)
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

# Enhanced metrics
_ss("session_start_time", time.time())
_ss("total_queries",      0)
_ss("agent_usage_count",  {})
_ss("error_count",        0)
_ss("avg_response_time",  0)
_ss("response_times",     [])


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
# MESSAGE STORE HELPERS (Enhanced with metrics)
# ══════════════════════════════════════════════════════════════════════════════
def push_msg(role, content, agent="", chart=None, code=None, lang="python",
             sources=None, research_sources=None, queries=None,
             timestamps=None, delegated=False, vstore_data=None, routing=None,
             response_time=None):
    st.session_state.messages.append({
        "role":             role,
        "content":          content,
        "agent":            agent,
        "chart":            chart,
        "code":             code,
        "lang":             lang,
        "sources":          sources          or [],
        "research_sources": research_sources or [],
        "queries":          queries          or [],
        "timestamps":       timestamps       or [],
        "delegated":        delegated,
        "vstore_data":      vstore_data,
        "routing":          routing,
        "response_time":    response_time,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # Track metrics
    if role == "assistant" and agent:
        if agent not in st.session_state.agent_usage_count:
            st.session_state.agent_usage_count[agent] = 0
        st.session_state.agent_usage_count[agent] += 1
        
        if response_time:
            st.session_state.response_times.append(response_time)
            st.session_state.avg_response_time = sum(st.session_state.response_times) / len(st.session_state.response_times)

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

def export_chat_history():
    """Export chat history as JSON"""
    export_data = {
        "session_info": {
            "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_messages": len(st.session_state.messages),
            "session_duration": time.time() - st.session_state.session_start_time,
        },
        "messages": st.session_state.messages,
        "metrics": {
            "total_queries": st.session_state.total_queries,
            "agent_usage": st.session_state.agent_usage_count,
            "avg_response_time": st.session_state.avg_response_time,
            "error_count": st.session_state.error_count,
        }
    }
    return json.dumps(export_data, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# RENDER MESSAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_message(msg):
    role    = msg["role"]
    content = msg["content"]
    agent   = msg.get("agent", "NEXUS")

    if role == "user":
        st.markdown(f"""
<div class="bubble-user-shell">
  <div class="bubble-user-box">
    <div class="bubble-label">▸ You</div>
    {content}
  </div>
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

        # Response time badge
        rt_badge = ""
        if msg.get("response_time"):
            rt = msg["response_time"]
            rt_badge = f'<span class="badge info">⚡ {rt:.2f}s</span>'

        st.markdown(f"""
<div class="bubble-agent-shell">
  <div class="bubble-agent-box">
    <div class="bubble-label">⬡ {agent}{del_tag}{conf_badge}{rt_badge}</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown(content)

        if msg.get("chart"):
            try:
                st.image(base64.b64decode(msg["chart"]), use_container_width=True)
            except Exception:
                pass

        if msg.get("code"):
            st.code(msg["code"], language=msg.get("lang", "python"))

        if msg.get("sources"):
            pills = "".join(f'<span class="src-pill">📄 {s}</span>' for s in msg["sources"])
            st.markdown(f'<div style="margin:0.3rem 0 0.2rem">{pills}</div>', unsafe_allow_html=True)

        if msg.get("timestamps"):
            ts_html = "".join(
                f'<a href="{t["yt_link"]}" target="_blank" class="ts-pill">⏱ {t["timestamp"]}</a>'
                for t in msg["timestamps"] if t.get("yt_link")
            )
            if ts_html:
                st.markdown(f'<div style="margin:0.3rem 0 0.2rem">{ts_html}</div>', unsafe_allow_html=True)

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


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    return fig, ax

def show_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# ENTER-TO-SEND JS
# ══════════════════════════════════════════════════════════════════════════════
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
# SIDEBAR (Enhanced with performance metrics)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem">
  <div style="font-size:1.8rem;font-weight:800;
    background:linear-gradient(135deg,#3b82f6,#7c6df2,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;letter-spacing:-0.02em">⬡ NEURALRAG</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
    color:#6b7280;letter-spacing:0.15em;margin-top:0.2rem">MULTI-AGENT AI</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.agents_ready:
        st.markdown(f'<span class="badge on">● ONLINE · {GEMINI_MODEL}</span>', unsafe_allow_html=True)
        
        # Session duration
        duration = int(time.time() - st.session_state.session_start_time)
        mins, secs = divmod(duration, 60)
        st.markdown(f'<span class="badge info">⏱ {mins}m {secs}s</span>', unsafe_allow_html=True)
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
        ("data",     "📊", "Data Analyst",     "CSV/Excel analysis + AI charts"),
        ("code",     "💻", "Code Generator",   "Generate · Explain · Debug"),
        ("research", "🔬", "Web Researcher",   "Multi-step live web research"),
        ("auto",     "🧠", "Auto-Route",       "LLM picks the best agent"),
    ]
    for aid, icon, name, desc in AGENTS:
        is_active = st.session_state.active_agent == aid
        label = f"{icon} **{name}** ✓" if is_active else f"{icon} {name}"
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

    # Performance metrics
    if st.session_state.avg_response_time > 0:
        st.markdown(f'<span class="badge info">⚡ Avg: {st.session_state.avg_response_time:.2f}s</span>', unsafe_allow_html=True)
    
    if st.session_state.error_count > 0:
        st.markdown(f'<span class="badge off">⚠ Errors: {st.session_state.error_count}</span>', unsafe_allow_html=True)

    st.markdown("")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.agents_ready and st.session_state.orchestrator:
                st.session_state.orchestrator.chatbot.clear_history()
            st.rerun()
    
    with col2:
        if st.button("💾 Export", use_container_width=True):
            export_json = export_chat_history()
            st.download_button(
                label="📥 Download JSON",
                data=export_json,
                file_name=f"neuralrag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

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
  <div class="nexus-title">NEURALRAG</div>
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
# MAIN TABS (Enhanced with Analytics and Docs tabs)
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_ingest, tab_viz, tab_vdb, tab_analytics, tab_about, tab_docs = st.tabs([
    "💬 Chat", "📥 Ingest", "📊 Visualize", "◈ Vector DB", "📈 Analytics", "ℹ️ About", "📚 Docs"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — CHAT (Same as before, with response time tracking)
# ════════════════════════════════════════════════════════════
with tab_chat:
    AGENT_META = {
        "chat":     ("🤖", "General Chatbot",  "Conversational AI with memory · auto-delegates"),
        "rag":      ("📄", "Document Q&A",     "Semantic search over your uploaded documents"),
        "video":    ("🎬", "YouTube RAG",      "Paste a YouTube URL · Q&A with timestamps"),
        "data":     ("📊", "Data Analyst",     "Intelligent data analysis with AI-generated charts"),
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

    if not st.session_state.messages:
        st.markdown("""
<div style="text-align:center;padding:3rem 1rem;opacity:0.35;
  font-family:'Space Mono',monospace;font-size:0.78rem;color:#64748b">
  ⬡<br><br>
  Start a conversation · Upload documents · Load a YouTube video<br>
  <span style="font-size:0.65rem">Select an agent from the sidebar or use Auto-Route</span>
</div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(msg)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

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

    if active == "code":
        code_lang = st.selectbox(
            "Language",
            ["python","javascript","typescript","java","go","rust","sql","bash","c++","c#","php","kotlin"],
            key="code_lang_sel",
        )

    placeholder_map = {
        "chat":     "Ask anything… (Enter ↵ to send · Shift+Enter for new line)",
        "rag":      "Ask about your uploaded documents…",
        "video":    "Ask about the loaded video…",
        "data":     "Ask about your data — charts will appear in chat! 📊",
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

    if send and user_input.strip():
        q = user_input.strip()
        push_msg("user", q)
        st.session_state.total_queries += 1
        
        start_time = time.time()

        with st.spinner("Thinking…"):
            try:
                if active == "auto":
                    result   = orch.route(q, get_context())
                    routing  = result.get("_routing", {})
                    agent_id = routing.get("agent", "chat")
                    vd       = result.get("vstore_data")
                    if vd: st.session_state.last_vstore_data = vd
                    
                    response_time = time.time() - start_time
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
                        response_time=response_time,
                    )

                elif active == "rag":
                    if not st.session_state.rag_ingested:
                        push_msg("assistant",
                                 "⚠️ No documents loaded. Upload files in the **Ingest** tab.",
                                 agent="RAG")
                    else:
                        result = orch.rag.query(q)
                        vd = result.get("vstore_data")
                        if vd: st.session_state.last_vstore_data = vd
                        response_time = time.time() - start_time
                        push_msg("assistant", result["answer"], agent="RAG",
                                 sources=result.get("sources", []), vstore_data=vd,
                                 response_time=response_time)

                elif active == "video":
                    if not st.session_state.video_ingested:
                        push_msg("assistant",
                                 "⚠️ No video loaded. Paste a YouTube URL above and click ▶ Load.",
                                 agent="VIDEO RAG")
                    else:
                        result = orch.video_rag.query(q)
                        vd = result.get("vstore_data")
                        if vd: st.session_state.last_vstore_data = vd
                        response_time = time.time() - start_time
                        push_msg("assistant", result["answer"], agent="VIDEO RAG",
                                 timestamps=result.get("timestamps", []), vstore_data=vd,
                                 response_time=response_time)

                elif active == "data":
                    if not st.session_state.data_loaded:
                        push_msg("assistant",
                                 "⚠️ No data loaded. Upload a CSV/Excel in the **Ingest** tab.",
                                 agent="DATA 📊")
                    else:
                        result = orch.data_agent.analyze(q)
                        response_time = time.time() - start_time
                        push_msg("assistant", result["answer"], agent="DATA 📊",
                                 chart=result.get("chart"), response_time=response_time)

                elif active == "code":
                    lang   = st.session_state.get("code_lang_sel", "python")
                    result = orch.code_agent.generate(q, language=lang)
                    response_time = time.time() - start_time
                    push_msg("assistant", result["answer"], agent="CODE 💻",
                             code=result.get("code", ""), lang=lang, response_time=response_time)

                elif active == "research":
                    result = orch.research_agent.research(q)
                    response_time = time.time() - start_time
                    push_msg("assistant", result["answer"], agent="RESEARCH 🔬",
                             research_sources=result.get("sources", []),
                             queries=result.get("queries", []), response_time=response_time)

                else:  # chat
                    response_time = time.time() - start_time
                    push_msg("assistant", orch.chatbot.chat(q), agent="NEXUS 🤖",
                             response_time=response_time)

            except Exception as e:
                st.session_state.error_count += 1
                push_msg("assistant", f"⚠️ **Error:** {e}", agent="SYSTEM")

        st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 2 — INGEST (Same as before)
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

    elif ingest_choice == "🎬 YouTube":
        st.markdown("Load a YouTube video to enable transcript Q&A.")
        yt_url2  = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", key="yt_url_ingest")
        yt_lang2 = st.selectbox("Transcript language",
                                ["en","hi","es","fr","de","ja","ko","pt","ar","ru"], key="yt_lang_ingest")
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
            sum_style = st.selectbox("Summary style", ["detailed","brief","bullets"], key="sum_style_sel")
            if st.button("📝 Summarize Video", key="sum_video_btn", use_container_width=True):
                with st.spinner("Summarizing…"):
                    s = orch.video_rag.summarize(style=sum_style)
                st.markdown(s.get("summary", ""))

    elif ingest_choice == "📊 CSV / Excel":
        st.markdown("Upload a **CSV or Excel** file for AI-powered data analysis.")
        data_file = st.file_uploader("Drop CSV or Excel here",
                                     type=["csv","xlsx","xls"], key="data_uploader")
        if st.button("📊 Load Data", key="load_data_btn", use_container_width=True):
            if not data_file:
                st.warning("Please upload a file.")
            else:
                with st.spinner("Loading…"):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(data_file.name).suffix)
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
# TAB 3 — VISUALIZE (Same as before)
# ════════════════════════════════════════════════════════════
with tab_viz:
    st.markdown("### 📊 Data Visualizer")

    if not st.session_state.data_loaded or orch.data_agent.df is None:
        st.info("💡 Load a CSV or Excel file in the **Ingest** tab to visualize it here.")
    elif not PD_OK:
        st.error("pandas not installed.")
    else:
        df           = orch.data_agent.df
        numeric_cols = list(df.select_dtypes(include="number").columns)
        cat_cols     = list(df.select_dtypes(include=["object","category"]).columns)
        all_cols     = list(df.columns)

        st.markdown(f"**{st.session_state.data_filename}** · {df.shape[0]:,} rows × {df.shape[1]} cols")
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
            if st.button("📊 Generate Bar Chart", use_container_width=True):
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
            if st.button("📈 Generate Line Chart", use_container_width=True):
                fig, ax = dark_fig()
                ax.plot(df[x_col], df[y_col], color="#06b6d4", linewidth=2)
                ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} over {x_col}"); show_fig(fig)

        elif viz_type == "Histogram":
            col  = st.selectbox("Column", numeric_cols or all_cols, key="hist_col")
            bins = st.slider("Bins", 5, 100, 30, key="hist_bins")
            if st.button("📊 Generate Histogram", use_container_width=True):
                fig, ax = dark_fig()
                ax.hist(df[col].dropna(), bins=bins, color="#3b82f6",
                        alpha=0.85, edgecolor="#1e3a5f")
                ax.set_xlabel(col); ax.set_title(f"Distribution of {col}"); show_fig(fig)

        elif viz_type == "Scatter":
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X", numeric_cols or all_cols, key="sc_x")
            y_col = c2.selectbox("Y", numeric_cols or all_cols, key="sc_y")
            c_col = c3.selectbox("Color by", ["None"] + cat_cols, key="sc_c")
            if st.button("📍 Generate Scatter", use_container_width=True):
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
            if st.button("🔥 Generate Correlation Heatmap", use_container_width=True):
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
            if st.button("📦 Generate Box Plot", use_container_width=True):
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
                ax.set_ylabel(y_col); ax.set_title(f"Box Plot: {y_col}"); show_fig(fig)

        st.markdown("---")
        st.markdown("**🤖 AI Chart Generator** — describe a chart in plain English")
        ai_q = st.text_input("", placeholder="e.g. show me a bar chart of top 10 products by revenue",
                             key="ai_chart_input", label_visibility="collapsed")
        if st.button("🎨 Generate with AI", key="ai_chart_btn", use_container_width=True):
            if ai_q.strip():
                with st.spinner("🤖 Generating your chart…"):
                    r = orch.data_agent.analyze(ai_q)
                st.markdown(r["answer"])
                if r.get("chart"):
                    st.image(base64.b64decode(r["chart"]), use_container_width=True)
            else:
                st.warning("Please describe the chart you want.")


# ════════════════════════════════════════════════════════════
# TAB 4 — VECTOR DB (Same as before)
# ════════════════════════════════════════════════════════════
with tab_vdb:
    st.markdown("### ◈ Vector DB Showcase")
    st.markdown("Inspect indexed chunks, embedding metadata, and retrieval similarity scores in real time.")

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
  font-family:'Space Mono',monospace;font-size:0.75rem;color:#64748b">
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
  font-family:'Space Mono',monospace;font-size:0.75rem;color:#64748b">
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
  font-family:'Space Mono',monospace;font-size:0.75rem;color:#64748b">
  ⟳<br><br>Use <b>Auto-Route</b> mode and send a message<br>
  to see how the orchestrator routes your queries.
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — ANALYTICS (NEW)
# ════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📈 Performance Analytics")
    
    # Session metrics
    st.markdown("#### 📊 Session Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
<div class="metric-card">
  <div class="metric-value">{}</div>
  <div class="metric-label">Total Queries</div>
</div>""".format(st.session_state.total_queries), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
<div class="metric-card">
  <div class="metric-value">{}</div>
  <div class="metric-label">Messages</div>
</div>""".format(len(st.session_state.messages)), unsafe_allow_html=True)
    
    with col3:
        avg_time = st.session_state.avg_response_time
        st.markdown("""
<div class="metric-card">
  <div class="metric-value">{:.2f}s</div>
  <div class="metric-label">Avg Response</div>
</div>""".format(avg_time if avg_time > 0 else 0), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
<div class="metric-card">
  <div class="metric-value">{}</div>
  <div class="metric-label">Errors</div>
</div>""".format(st.session_state.error_count), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Agent usage distribution
    if st.session_state.agent_usage_count:
        st.markdown("#### 🤖 Agent Usage Distribution")
        
        agent_df = pd.DataFrame(
            list(st.session_state.agent_usage_count.items()),
            columns=["Agent", "Count"]
        ).sort_values("Count", ascending=False)
        
        fig, ax = dark_fig(figsize=(10, 5))
        ax.bar(agent_df["Agent"], agent_df["Count"], color="#3b82f6", alpha=0.8)
        ax.set_xlabel("Agent")
        ax.set_ylabel("Usage Count")
        ax.set_title("Agent Usage Distribution")
        plt.xticks(rotation=45, ha="right")
        show_fig(fig)
        
        st.dataframe(agent_df, use_container_width=True)
    
    st.markdown("---")
    
    # Response time distribution
    if st.session_state.response_times:
        st.markdown("#### ⚡ Response Time Distribution")
        
        fig, ax = dark_fig(figsize=(10, 5))
        ax.hist(st.session_state.response_times, bins=20, color="#10b981", alpha=0.8, edgecolor="#065f46")
        ax.set_xlabel("Response Time (seconds)")
        ax.set_ylabel("Frequency")
        ax.set_title("Response Time Distribution")
        ax.axvline(st.session_state.avg_response_time, color="#ef4444", linestyle="--", linewidth=2, label=f"Average: {st.session_state.avg_response_time:.2f}s")
        ax.legend(labelcolor="white")
        show_fig(fig)
        
        # Response time stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Min Response", f"{min(st.session_state.response_times):.2f}s")
        col2.metric("Max Response", f"{max(st.session_state.response_times):.2f}s")
        col3.metric("Median Response", f"{sorted(st.session_state.response_times)[len(st.session_state.response_times)//2]:.2f}s")
    
    st.markdown("---")
    
    # System health
    st.markdown("#### 🏥 System Health")
    
    health_score = 100
    if st.session_state.error_count > 0:
        health_score -= min(st.session_state.error_count * 5, 30)
    if st.session_state.avg_response_time > 5:
        health_score -= 20
    
    health_color = "#10b981" if health_score >= 80 else "#f59e0b" if health_score >= 60 else "#ef4444"
    
    st.markdown(f"""
<div class="metric-card">
  <div class="metric-value" style="color: {health_color}">{health_score}%</div>
  <div class="metric-label">System Health Score</div>
</div>""", unsafe_allow_html=True)
    
    # Health indicators
    col1, col2, col3 = st.columns(3)
    
    status_ok = "🟢" if st.session_state.error_count == 0 else "🔴"
    col1.markdown(f"**{status_ok} Error Rate:** {st.session_state.error_count} errors")
    
    perf_ok = "🟢" if st.session_state.avg_response_time < 3 else "🟡" if st.session_state.avg_response_time < 5 else "🔴"
    col2.markdown(f"**{perf_ok} Performance:** {st.session_state.avg_response_time:.2f}s avg")
    
    uptime = int(time.time() - st.session_state.session_start_time)
    col3.markdown(f"**🟢 Uptime:** {uptime // 60}m {uptime % 60}s")


# ════════════════════════════════════════════════════════════
# TAB 6 — ABOUT (ENHANCED)
# ════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
<div style="text-align:center;margin:1.5rem 0">
  <div style="font-size:2.5rem;font-weight:800;
    background:linear-gradient(135deg,#3b82f6,#7c6df2,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;letter-spacing:-0.02em">⬡ NEURALRAG</div>
  <div style="font-family:'Space Mono',monospace;color:#6b7280;font-size:0.85rem;
    letter-spacing:0.12em;margin-top:0.5rem">Multi-Agent Intelligence Platform</div>
  <div style="color:#94a3b8;font-size:0.75rem;margin-top:0.3rem">Version 4.0 Enhanced · 2024</div>
</div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Overview
    st.markdown("""
### 🎯 Overview

**NeuralRAG** is a production-grade multi-agent AI platform that intelligently routes queries to specialized agents, enabling document Q&A, video analysis, data visualization, code generation, and web research through a unified conversational interface.

Built with modern AI technologies and designed for scalability, NeuralRAG combines the power of Google's Gemini LLM, semantic search, and intelligent routing to deliver accurate, context-aware responses across diverse use cases.
""")
    
    st.markdown("---")
    
    # Key Features
    st.markdown("### ✨ Key Features")
    
    features = [
        ("🤖", "Multi-Agent Architecture", "Seven specialized agents working in harmony with intelligent routing and delegation"),
        ("🔍", "Hybrid RAG System", "Dense vector search + BM25 keyword matching with contextual compression"),
        ("🎬", "Video Intelligence", "YouTube transcript processing with timestamp-aware Q&A"),
        ("📊", "AI Data Analysis", "Automated chart generation and statistical insights from structured data"),
        ("💻", "Code Generation", "Multi-language code creation with inline documentation"),
        ("🔬", "Web Research", "Multi-query synthesis with live web search integration"),
        ("📈", "Performance Monitoring", "Real-time analytics, health metrics, and usage tracking"),
        ("💾", "Export & Persistence", "Chat history export, session state management"),
    ]
    
    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        target = col1 if i % 2 == 0 else col2
        with target:
            st.markdown(f"""
<div class="feature-card">
  <div class="feature-icon">{icon}</div>
  <div class="feature-title">{title}</div>
  <div class="feature-desc">{desc}</div>
</div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Architecture
    st.markdown("### 🏗️ System Architecture")
    
    st.markdown("""
<div class="arch-diagram">
<div class="arch-layer">
<strong>📱 PRESENTATION LAYER</strong><br>
Streamlit UI · Real-time Chat · File Upload · Visualization Dashboard
</div>

<div class="arch-arrow">↓</div>

<div class="arch-layer">
<strong>🧠 ORCHESTRATION LAYER</strong><br>
Multi-Agent Router · Context Manager · Query Classifier · Delegation Logic
</div>

<div class="arch-arrow">↓</div>

<div class="arch-layer">
<strong>🤖 AGENT LAYER</strong><br>
Chatbot · RAG · Video RAG · Data Analyst · Code Gen · Research · Auto-Route
</div>

<div class="arch-arrow">↓</div>

<div class="arch-layer">
<strong>🔢 EMBEDDING & VECTOR LAYER</strong><br>
SentenceTransformers (MiniLM-L6-v2 · 384d) · FAISS IndexFlatL2 · BM25 Reranking
</div>

<div class="arch-arrow">↓</div>

<div class="arch-layer">
<strong>🧬 LLM LAYER</strong><br>
Google Gemini 2.0 Flash · Function Calling · Streaming Responses
</div>

<div class="arch-arrow">↓</div>

<div class="arch-layer">
<strong>💾 DATA LAYER</strong><br>
Document Store · Vector Index · Session State · Chat History · Metrics DB
</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology Stack
    st.markdown("### 🛠️ Technology Stack")
    
    tech_stack = {
        "**LLM**": "Google Gemini 2.0 Flash (`google-generativeai`)",
        "**Embeddings**": "SentenceTransformers MiniLM-L6-v2 (384 dimensions)",
        "**Vector DB**": "FAISS with IndexFlatL2 + BM25 hybrid reranking",
        "**Framework**": "LangChain for orchestration and document processing",
        "**Frontend**": "Streamlit with custom CSS and responsive design",
        "**Video Processing**": "YouTube Transcript API with multi-language support",
        "**Data Analysis**": "Pandas + NumPy + Matplotlib + Seaborn",
        "**Code Generation**": "Multi-language support (Python, JS, Java, Go, Rust, etc.)",
    }
    
    for tech, desc in tech_stack.items():
        st.markdown(f"- {tech}: {desc}")
    
    st.markdown("---")
    
    # Agent Capabilities
    st.markdown("### 🤖 Agent Capabilities Matrix")
    
    agent_table = pd.DataFrame({
        "Agent": ["🤖 Chatbot", "📄 Document RAG", "🎬 YouTube RAG", "📊 Data Analyst", "💻 Code Generator", "🔬 Web Researcher", "🧠 Auto-Route"],
        "Primary Function": [
            "Conversational AI",
            "Document Q&A",
            "Video Transcript Q&A",
            "Data Analysis",
            "Code Generation",
            "Web Research",
            "Intelligent Routing"
        ],
        "Context Window": ["20 turns", "Unlimited", "Video length", "Dataset size", "Single query", "Multiple queries", "N/A"],
        "Memory": ["✅ Rolling", "❌", "❌", "❌", "❌", "❌", "❌"],
        "Real-time Data": ["❌", "❌", "❌", "❌", "❌", "✅ Web", "✅ Delegates"],
    })
    
    st.dataframe(agent_table, use_container_width=True)
    
    st.markdown("---")
    
    # RAG Pipeline
    st.markdown("### 🔄 RAG Pipeline Flow")
    
    st.markdown("""
```
┌─────────────────┐
│  Upload Files   │
│  (PDF/TXT/MD)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load & Parse   │
│  (LangChain)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunk Text     │
│  (800 chars,    │
│   150 overlap)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embed Chunks   │
│  (MiniLM-L6-v2) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Index    │
│  (L2 distance)  │
└────────┬────────┘
         │
    ┌────┴────┐
    │  QUERY  │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│  Dense Search   │
│  (k=6 chunks)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BM25 Rerank    │
│  (Top 5)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Compress   │
│  (Contextual)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Grounded       │
│  Answer + Cite  │
└─────────────────┘
```
""")
    
    st.markdown("---")
    
    # Routing Logic
    st.markdown("### ⚡ Intelligent Routing Logic")
    
    st.markdown("""
The Auto-Route agent uses a three-tier decision system:

1. **⚡ Regex Fast-Path** (0 LLM calls)
   - Pattern matching for common queries
   - Instant routing for known patterns
   - Fallback to LLM if no match

2. **🧠 LLM Classification** (1 LLM call)
   - Gemini analyzes query intent
   - Returns agent + confidence score (0–1)
   - Confidence threshold: 0.5 minimum

3. **🔄 Context Override** (Priority routing)
   - Checks loaded data sources
   - Routes to appropriate RAG if data exists
   - Overrides LLM decision when applicable

**Example Routing:**
```
Query: "What's in the uploaded PDF?"
  ⚡ Regex: No match
  🧠 LLM: rag, confidence=0.95
  🔄 Context: rag_ingested=True → ROUTE TO RAG
```
""")
    
    st.markdown("---")
    
    # Performance Optimizations
    st.markdown("### ⚙️ Performance Optimizations")
    
    optimizations = [
        "**Semantic Caching**: Embedding cache for repeated queries",
        "**Batch Processing**: Parallel document chunking and embedding",
        "**Lazy Loading**: Agents initialized only when needed",
        "**Response Streaming**: Real-time token streaming from Gemini",
        "**Contextual Compression**: LLM-based chunk relevance filtering",
        "**BM25 Reranking**: Hybrid search improves retrieval accuracy by 15-20%",
        "**Session Persistence**: State recovery after server restart",
        "**Metric Tracking**: Real-time performance monitoring with minimal overhead",
    ]
    
    for opt in optimizations:
        st.markdown(f"- {opt}")
    
    st.markdown("---")
    
    # Configuration
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("""
Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-api-key-here"
GEMINI_MODEL   = "gemini-2.0-flash"
```

**Environment Variables:**
- `GEMINI_API_KEY`: Your Google AI API key
- `GEMINI_MODEL`: Model identifier (default: gemini-2.0-flash)

**Optional Tuning:**
- Chunk size: 800 characters (adjustable in agents.py)
- Chunk overlap: 150 characters
- Top-k retrieval: 6 chunks → 5 after rerank
- Embedding dimension: 384 (MiniLM-L6-v2 fixed)
""")
    
    st.markdown("---")
    
    # Keyboard Shortcuts
    st.markdown("### ⌨️ Keyboard Shortcuts")
    
    shortcuts = [
        ("Enter", "Send message", "In chat input field"),
        ("Shift + Enter", "New line", "In chat input field"),
        ("Escape", "Close modal", "When modal is open"),
        ("Tab", "Navigate fields", "In forms"),
    ]
    
    shortcut_df = pd.DataFrame(shortcuts, columns=["Shortcut", "Action", "Context"])
    st.table(shortcut_df)
    
    st.markdown("---")
    
    # Use Cases
    st.markdown("### 💡 Use Cases")
    
    use_cases = [
        ("📚 **Academic Research**", "Upload research papers, ask questions, get cited answers with source references"),
        ("🎓 **Learning & Education**", "Process YouTube lectures, generate study notes, create flashcards"),
        ("📊 **Business Intelligence**", "Analyze sales data, generate reports, create executive dashboards"),
        ("💼 **Software Development**", "Generate boilerplate code, debug errors, explain algorithms"),
        ("🔍 **Competitive Analysis**", "Research competitors, synthesize market reports, track trends"),
        ("📝 **Content Creation**", "Research topics, gather sources, create content outlines"),
    ]
    
    for title, desc in use_cases:
        st.markdown(f"**{title}**")
        st.markdown(f"> {desc}")
        st.markdown("")
    
    st.markdown("---")
    
    # Limitations
    st.markdown("### ⚠️ Known Limitations")
    
    limitations = [
        "**Context Window**: Limited by Gemini's token limit (~1M tokens for 2.0 Flash)",
        "**Embedding Quality**: MiniLM-L6-v2 may underperform on domain-specific jargon",
        "**Video Length**: Very long videos (>3 hours) may hit processing limits",
        "**Data Size**: Large datasets (>1M rows) may cause memory issues",
        "**Concurrent Users**: Single-instance deployment; use load balancer for scale",
        "**Cost**: Gemini API calls incur usage charges (monitor via Google Cloud Console)",
    ]
    
    for lim in limitations:
        st.markdown(f"- {lim}")
    
    st.markdown("---")
    
    # Future Roadmap
    st.markdown("### 🚀 Future Roadmap")
    
    roadmap = [
        "🔐 **Multi-user Authentication** with role-based access control",
        "☁️ **Cloud Deployment** guides for AWS, GCP, Azure",
        "🗄️ **PostgreSQL Integration** for persistent storage",
        "🔄 **Multi-modal Support** for image analysis and audio transcription",
        "🌐 **API Gateway** for programmatic access",
        "📱 **Mobile App** with offline mode",
        "🎨 **Custom Themes** and white-label options",
        "📊 **Advanced Analytics** with predictive insights",
    ]
    
    for item in roadmap:
        st.markdown(f"- {item}")
    
    st.markdown("---")
    
    # Credits & License
    st.markdown("""
### 📜 License & Credits

**License**: MIT License

**Core Technologies**:
- Google Gemini API
- LangChain
- FAISS (Facebook AI Similarity Search)
- Streamlit
- Sentence Transformers

**Built with** ❤️ **for the AI community**

---

<div style="text-align:center;margin:2rem 0;color:#6b7280;font-size:0.8rem">
  <strong>⬡ NeuralRAG v4.0</strong> · Multi-Agent Intelligence Platform<br>
  Powered by Google Gemini · LangChain · FAISS
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 7 — DOCS (NEW)
# ════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("### 📚 Documentation")
    
    doc_section = st.radio(
        "Section",
        ["🚀 Quick Start", "📖 User Guide", "🔧 API Reference", "❓ FAQ", "🐛 Troubleshooting"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    
    if doc_section == "🚀 Quick Start":
        st.markdown("""
## 🚀 Quick Start Guide

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/neuralrag.git
cd neuralrag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys**
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-api-key-here"
GEMINI_MODEL   = "gemini-2.0-flash"
```

4. **Run the application**
```bash
streamlit run app.py
```

### First Steps

1. **Select an Agent** from the sidebar
2. **Upload Data** (optional) via the Ingest tab
3. **Start Chatting** in the Chat tab
4. **View Analytics** to monitor performance

### Example Workflows

**Document Q&A**:
1. Go to **Ingest → Documents**
2. Upload PDF/TXT/MD files
3. Click **🔄 Ingest Documents**
4. Switch to **Chat** tab
5. Ask questions about your documents

**YouTube Analysis**:
1. Go to **Ingest → YouTube**
2. Paste a YouTube URL
3. Click **▶ Load Video**
4. Ask questions with timestamp-aware responses

**Data Visualization**:
1. Go to **Ingest → CSV / Excel**
2. Upload your dataset
3. Go to **Visualize** tab
4. Create charts or use AI chart generator
""")
    
    elif doc_section == "📖 User Guide":
        st.markdown("""
## 📖 User Guide

### Chat Interface

The chat interface is your main interaction point with NeuralRAG. Messages appear in bubbles with agent identification and metadata.

**Message Types**:
- **User Messages**: Blue bubbles on the right
- **Agent Responses**: White bubbles on the left with agent icon

**Response Metadata**:
- **Response Time**: ⚡ badge showing processing duration
- **Confidence Score**: Color-coded confidence indicator
- **Source Citations**: 📄 pills linking to document sources
- **Timestamps**: ⏱ pills linking to video timestamps

### Agent Selection

**🤖 General Chatbot**
- Conversational AI with 20-turn memory
- Best for: General questions, casual chat
- Auto-delegates to specialized agents when needed

**📄 Document Q&A**
- Semantic search over uploaded documents
- Best for: Research, document analysis
- Requires: Documents uploaded via Ingest tab

**🎬 YouTube RAG**
- Transcript-based Q&A with timestamps
- Best for: Video content analysis, note-taking
- Requires: YouTube URL loaded via Ingest tab

**📊 Data Analyst**
- AI-powered data analysis and visualization
- Best for: CSV/Excel analysis, chart creation
- Requires: Dataset uploaded via Ingest tab

**💻 Code Generator**
- Multi-language code generation
- Best for: Boilerplate code, debugging, explanations
- Supports: Python, JS, Java, Go, Rust, SQL, etc.

**🔬 Web Researcher**
- Live web search with multi-query synthesis
- Best for: Current events, research, fact-checking
- Requires: Internet connection

**🧠 Auto-Route**
- Intelligent routing to best agent
- Best for: Mixed queries, uncertain use case
- Uses: Regex → LLM → Context override logic

### Data Ingestion

**Documents**:
- Supported formats: PDF, TXT, MD, CSV, DOCX
- Chunking: 800 characters, 150 overlap
- Embedding: MiniLM-L6-v2 (384d)
- Index: FAISS with BM25 reranking

**YouTube Videos**:
- Transcript languages: 10+ supported
- Auto-segmentation: Timestamp-aware chunks
- Search: Dense + keyword hybrid

**CSV/Excel**:
- Max size: ~1M rows (memory-dependent)
- Auto-profiling: Statistical summary
- AI analysis: Natural language queries

### Visualization

**Chart Types**:
- Bar charts
- Line charts
- Histograms
- Scatter plots
- Correlation heatmaps
- Box plots

**AI Chart Generator**:
Describe your visualization in plain English:
```
"Show me a bar chart of top 10 products by revenue"
"Create a scatter plot of price vs quantity colored by category"
"Generate a correlation heatmap for all numeric columns"
```

### Vector DB Showcase

View real-time retrieval internals:
- **Chunk Preview**: See indexed text segments
- **Similarity Scores**: Understand ranking logic
- **Metadata**: Inspect embedding dimensions, sources
- **Routing Log**: Track agent selection decisions

### Analytics Dashboard

Monitor system performance:
- **Total Queries**: Session query count
- **Avg Response Time**: Mean processing duration
- **Agent Distribution**: Usage breakdown by agent
- **Error Rate**: Failed query tracking
- **Health Score**: Overall system health (0-100)
""")
    
    elif doc_section == "🔧 API Reference":
        st.markdown("""
## 🔧 API Reference

### Core Classes

#### `MultiAgentOrchestrator`

Main orchestrator class managing all agents and routing logic.

**Methods**:

```python
route(query: str, context: dict) -> dict
```
Routes query to appropriate agent with intelligent decision-making.

**Parameters**:
- `query`: User input string
- `context`: Dict with keys: `rag_ingested`, `video_ingested`, `data_loaded`, `data_filename`

**Returns**:
```python
{
    "answer": str,           # Agent response
    "chart": str | None,     # Base64 PNG for charts
    "code": str | None,      # Generated code
    "sources": list[str],    # Document sources
    "timestamps": list[dict],# Video timestamps
    "vstore_data": dict,     # Vector DB metadata
    "_routing": {            # Routing metadata
        "agent": str,        # Selected agent
        "method": str,       # regex | llm | context_override
        "confidence": float  # 0.0 - 1.0
    }
}
```

---

```python
get_routing_log() -> list[dict]
```
Returns routing history for analytics.

**Returns**:
```python
[
    {
        "query": str,
        "agent": str,
        "method": str,
        "confidence": float,
        "timestamp": str
    },
    ...
]
```

### Agent APIs

#### `RAGAgent`

```python
ingest(file_paths: list[str]) -> str
```
Index documents for semantic search.

```python
query(question: str) -> dict
```
Perform RAG query with source citations.

**Returns**:
```python
{
    "answer": str,
    "sources": list[str],
    "vstore_data": dict
}
```

---

#### `VideoRAGAgent`

```python
ingest(youtube_url: str, language: str = "en") -> str
```
Load and index YouTube transcript.

```python
query(question: str) -> dict
```
Query video content with timestamps.

**Returns**:
```python
{
    "answer": str,
    "timestamps": [
        {"timestamp": str, "yt_link": str},
        ...
    ],
    "vstore_data": dict
}
```

---

#### `DataAgent`

```python
load_data(file_path: str) -> str
```
Load CSV/Excel into pandas DataFrame.

```python
analyze(query: str) -> dict
```
AI-powered data analysis with charts.

**Returns**:
```python
{
    "answer": str,
    "chart": str | None  # Base64 PNG
}
```

---

#### `CodeAgent`

```python
generate(query: str, language: str = "python") -> dict
```
Generate code in specified language.

**Returns**:
```python
{
    "answer": str,
    "code": str
}
```

---

#### `ResearchAgent`

```python
research(query: str) -> dict
```
Multi-query web research synthesis.

**Returns**:
```python
{
    "answer": str,
    "sources": list[dict],  # [{title, url}, ...]
    "queries": list[str]    # Search queries used
}
```

### Utility Functions

```python
export_chat_history() -> str
```
Export session as JSON.

```python
push_msg(role: str, content: str, **kwargs)
```
Add message to session state.

```python
render_message(msg: dict)
```
Render chat bubble with metadata.
""")
    
    elif doc_section == "❓ FAQ":
        st.markdown("""
## ❓ Frequently Asked Questions

### General

**Q: What LLM does NeuralRAG use?**  
A: Google Gemini 2.0 Flash by default. Configurable via `GEMINI_MODEL` in secrets.toml.

**Q: Can I use other LLMs like OpenAI or Claude?**  
A: The codebase is designed for Gemini, but you can modify `agents.py` to support other providers via LangChain's abstraction layer.

**Q: Is my data stored anywhere?**  
A: No. All processing is in-memory. Data persists only in your session state until you clear it or restart the server.

**Q: Can multiple users use the same instance?**  
A: Yes, but sessions are isolated. Each user has their own session state. For production, use a load balancer.

---

### Performance

**Q: Why is the first response slow?**  
A: The first query initializes models and embeddings. Subsequent queries are faster due to caching.

**Q: How can I speed up RAG queries?**  
A: Reduce chunk size, lower top-k retrieval, or use a faster embedding model. Trade-off: accuracy vs speed.

**Q: What's the max file size for uploads?**  
A: Streamlit default is 200MB. Increase with `server.maxUploadSize` in `.streamlit/config.toml`.

---

### Troubleshooting

**Q: "No module named 'agents'"**  
A: Ensure `agents.py` is in the same directory as `app.py`.

**Q: "API key not found"**  
A: Create `.streamlit/secrets.toml` with your `GEMINI_API_KEY`.

**Q: Chart generation fails in Data Analyst**  
A: Ensure matplotlib, pandas, seaborn are installed: `pip install matplotlib pandas seaborn`.

**Q: YouTube ingestion fails**  
A: Check if the video has captions. Some videos don't provide transcripts.

**Q: Vector DB showcase shows 0 chunks**  
A: Re-ingest your data. Ensure chunking completed successfully.

---

### Features

**Q: Can I export chat history?**  
A: Yes. Use the "💾 Export" button in the sidebar to download JSON.

**Q: Does NeuralRAG support multi-language documents?**  
A: Embeddings work for 100+ languages, but performance varies. English is most accurate.

**Q: Can I customize chunk size?**  
A: Yes. Edit `agents.py` → `RecursiveCharacterTextSplitter` parameters.

**Q: How do I add a custom agent?**  
A: Create a new class in `agents.py`, add to `MultiAgentOrchestrator`, update routing logic.

---

### Deployment

**Q: Can I deploy to Streamlit Cloud?**  
A: Yes. Add `GEMINI_API_KEY` to Streamlit secrets in your dashboard.

**Q: How do I deploy to AWS/GCP/Azure?**  
A: Use Docker + container services (ECS, Cloud Run, App Service). See deployment guide.

**Q: Is there a Docker image?**  
A: Not pre-built. Create your own Dockerfile based on `python:3.10-slim` with dependencies.
""")
    
    elif doc_section == "🐛 Troubleshooting":
        st.markdown("""
## 🐛 Troubleshooting Guide

### Common Issues

#### 1. API Key Errors

**Error**: `API key not found` or `Invalid API key`

**Solutions**:
- Create `.streamlit/secrets.toml` in project root
- Add `GEMINI_API_KEY = "your-key"`
- Restart Streamlit server
- Verify key validity at Google AI Studio

---

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solutions**:
```bash
pip install -r requirements.txt
pip install --upgrade google-generativeai langchain sentence-transformers faiss-cpu
```

---

#### 3. Memory Issues

**Error**: `MemoryError` or `Killed`

**Solutions**:
- Reduce chunk size in `agents.py`
- Lower top-k retrieval parameter
- Use smaller datasets (<500k rows)
- Increase system RAM or use cloud deployment

---

#### 4. Chart Generation Failures

**Error**: Charts don't appear in Data Analyst

**Solutions**:
- Install visualization libraries:
```bash
pip install matplotlib seaborn pandas numpy
```
- Check that `matplotlib.use("Agg")` is at top of file
- Verify data has numeric columns
- Try a different chart type

---

#### 5. YouTube Ingestion Fails

**Error**: `Error loading video` or `No transcript found`

**Causes**:
- Video has no captions/subtitles
- Video is age-restricted
- Video is private/unlisted
- Network connectivity issues

**Solutions**:
- Try a different video
- Check video has captions on YouTube
- Use public, non-restricted videos
- Verify internet connection

---

#### 6. Slow Performance

**Issue**: Queries take >10 seconds

**Solutions**:
- **For RAG**: Reduce `chunk_size` and `top_k`
- **For Data**: Use smaller datasets or sample data
- **For Video**: Use shorter videos (<1 hour)
- **For LLM**: Switch to faster model (if available)
- Enable GPU for embeddings (FAISS-GPU)

---

#### 7. Vector DB Shows 0 Chunks

**Issue**: No chunks indexed after ingestion

**Solutions**:
- Re-upload files via Ingest tab
- Check file format compatibility
- Verify file isn't empty or corrupted
- Check Streamlit logs for errors
- Try a different file

---

#### 8. Session State Resets

**Issue**: Data disappears on page refresh

**Expected Behavior**: Streamlit sessions are ephemeral. Use Export feature to save data.

**Solutions**:
- Use "💾 Export" button to save chat history
- For persistence, implement database backend
- Use Streamlit's session state carefully

---

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Add to `app.py` to see detailed agent execution logs.

---

### Getting Help

1. **Check Logs**: Streamlit terminal output shows errors
2. **GitHub Issues**: Report bugs at repository issues page
3. **Documentation**: Re-read this guide and API reference
4. **Community**: Join Streamlit/LangChain Discord for support

---

### Performance Optimization Checklist

- [ ] Use SSD for faster file I/O
- [ ] Increase system RAM (16GB+ recommended)
- [ ] Enable GPU for FAISS (if available)
- [ ] Cache embeddings for repeated queries
- [ ] Reduce chunk size for faster indexing
- [ ] Use smaller datasets for testing
- [ ] Monitor memory usage with `htop`
- [ ] Profile code with `cProfile` if needed
""")
