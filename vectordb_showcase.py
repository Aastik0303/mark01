"""
vectordb_showcase.py — Vector DB Showcase UI Component
Drop-in Streamlit panel — call render_vectordb_showcase(data) from your main app.

FIXES:
  - Proper card grid layout that never degrades to plain list
  - All stats always in card boxes
  - Chunk grid uses CSS grid (not bare text flow)
  - Similarity bars always rendered in styled containers
"""

import streamlit as st
import math

_CSS_INJECTED = False

def _inject_css():
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    _CSS_INJECTED = True
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

/* ── Vector DB Showcase ──────────────────────────────────────────── */
.vdb-panel {
    background: #060c1a;
    border: 1px solid #1a2d4a;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.6rem;
    font-family: 'Space Mono', monospace;
    box-sizing: border-box;
    width: 100%;
}
.vdb-title {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Stats ── */
.vdb-stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.55rem;
    margin-bottom: 0.9rem;
}
@media (max-width: 600px) {
    .vdb-stat-row { grid-template-columns: repeat(2, 1fr); }
}
.vdb-stat {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    box-sizing: border-box;
}
.vdb-stat-val {
    font-size: 1.05rem;
    font-weight: 700;
    color: #7c6df2;
    line-height: 1.2;
    display: block;
}
.vdb-stat-label {
    font-size: 0.56rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.15rem;
    display: block;
}
.vdb-model-line {
    font-size: 0.58rem;
    color: #374151;
    margin-bottom: 0.8rem;
    font-family: 'Space Mono', monospace;
}

/* ── Section label ── */
.vdb-section-label {
    font-size: 0.58rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.vdb-retrieved-badge {
    color: #7c6df2;
    background: #12193a;
    border: 1px solid #2d3b6a;
    border-radius: 4px;
    padding: 0.05rem 0.4rem;
    font-size: 0.55rem;
}

/* ── Chunk grid ── */
.chunk-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 0.5rem;
    max-height: 330px;
    overflow-y: auto;
    padding-right: 4px;
    box-sizing: border-box;
}
.chunk-card {
    background: #0a0f1e;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
    box-sizing: border-box;
    position: relative;
    transition: border-color 0.18s;
}
.chunk-card:hover { border-color: #3b5bdb; }
.chunk-card.highlighted {
    border-color: #7c6df2;
    background: #0d1526;
    box-shadow: 0 0 0 1px #7c6df2, 0 4px 14px rgba(124,109,242,0.14);
}
.chunk-idx-badge {
    position: absolute;
    top: 0.35rem;
    right: 0.45rem;
    font-size: 0.54rem;
    color: #374151;
}
.chunk-rank-badge {
    position: absolute;
    top: 0.35rem;
    left: 0.45rem;
    font-size: 0.54rem;
    color: #7c6df2;
}
.chunk-hash {
    font-size: 0.6rem;
    color: #3b5bdb;
    margin-bottom: 0.2rem;
    margin-top: 0.1rem;
}
.chunk-src {
    font-size: 0.56rem;
    color: #475569;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
    margin-bottom: 0.28rem;
    display: block;
}
.chunk-preview {
    font-size: 0.63rem;
    color: #94a3b8;
    line-height: 1.42;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.chunk-chars {
    font-size: 0.54rem;
    color: #374151;
    margin-top: 0.28rem;
    display: block;
}

/* ── Similarity results ── */
.sim-section {
    margin-top: 0.9rem;
    padding-top: 0.7rem;
    border-top: 1px solid #1e2d4a;
}
.sim-title {
    font-size: 0.6rem;
    color: #f59e0b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}
.sim-query-box {
    font-size: 0.68rem;
    color: #c7d2fe;
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 0.3rem 0.65rem;
    margin-bottom: 0.6rem;
    word-break: break-word;
    display: block;
    box-sizing: border-box;
}
.sim-result-card {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    margin-bottom: 0.5rem;
    padding: 0.5rem 0.65rem;
    background: #0a0f1e;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    position: relative;
    overflow: hidden;
    box-sizing: border-box;
}
.sim-bg-fill {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    background: linear-gradient(90deg, rgba(124,109,242,0.07), transparent);
    pointer-events: none;
}
.sim-rank-num {
    font-size: 0.75rem;
    font-weight: 700;
    color: #7c6df2;
    min-width: 18px;
    text-align: center;
    padding-top: 0.08rem;
    position: relative;
    z-index: 1;
}
.sim-body {
    flex: 1;
    min-width: 0;
    position: relative;
    z-index: 1;
}
.sim-meta-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.18rem;
    flex-wrap: wrap;
}
.sim-pct-label {
    font-size: 0.68rem;
    font-weight: 700;
}
.sim-cid-label {
    font-size: 0.55rem;
    color: #3b5bdb;
}
.sim-src-tag {
    font-size: 0.53rem;
    color: #475569;
    background: #0d1526;
    border: 1px solid #1e2d4a;
    border-radius: 4px;
    padding: 0.03rem 0.28rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 130px;
    display: inline-block;
}
.sim-preview-text {
    font-size: 0.61rem;
    color: #94a3b8;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ── Score bars ── */
.score-dist-section {
    margin-top: 0.9rem;
    padding-top: 0.7rem;
    border-top: 1px solid #1e2d4a;
}
.score-dist-title {
    font-size: 0.6rem;
    color: #06b6d4;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.55rem;
}
.score-bar-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.32rem;
}
.score-bar-lbl {
    font-size: 0.58rem;
    color: #475569;
    min-width: 55px;
    text-align: right;
}
.score-bar-track {
    flex: 1;
    height: 6px;
    background: #0a1628;
    border-radius: 3px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 3px;
}
.score-bar-val {
    font-size: 0.58rem;
    min-width: 34px;
    text-align: right;
}

/* ── Chunk grid scrollbar ── */
.chunk-grid::-webkit-scrollbar { width: 3px; }
.chunk-grid::-webkit-scrollbar-track { background: #0a0f1e; }
.chunk-grid::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

/* ── Routing log ── */
.vdb-routing-panel {
    background: #060c1a;
    border: 1px solid #1a3a2a;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    box-sizing: border-box;
    width: 100%;
}
.vdb-routing-title {
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 0.7rem;
}
.route-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.38rem;
    padding: 0.38rem 0.55rem;
    background: #0a0f1e;
    border: 1px solid #1e2d4a;
    border-radius: 6px;
    flex-wrap: wrap;
    box-sizing: border-box;
}
.route-msg {
    font-size: 0.6rem;
    color: #475569;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.route-agent-tag {
    background: #0d1526;
    border: 1px solid #1e3a5f;
    border-radius: 4px;
    padding: 0.03rem 0.38rem;
    color: #c7d2fe;
    font-size: 0.58rem;
    white-space: nowrap;
}
.route-method-tag {
    font-size: 0.58rem;
    white-space: nowrap;
}
.route-conf-tag {
    font-size: 0.58rem;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)


def _short_src(src: str, maxlen: int = 26) -> str:
    if not src or src == "?":
        return "unknown"
    s = src.split("/")[-1] if "/" in src else src
    return ("…" + s[-(maxlen-1):]) if len(s) > maxlen else s


def _sim_color(pct: float) -> str:
    if pct >= 80: return "#4ade80"
    if pct >= 60: return "#f59e0b"
    if pct >= 40: return "#3b82f6"
    return "#ef4444"


def render_vectordb_showcase(data: dict, title: str = "Vector DB Showcase"):
    """
    Render the Vector DB Showcase panel with proper card grid layout.
    Always renders styled cards — never degrades to bare text list.
    """
    _inject_css()

    if not data:
        st.info("No vector store data available yet.")
        return

    total_chunks = data.get("total_chunks", 0)
    total_chars  = data.get("total_chars", 0)
    emb_dim      = data.get("embedding_dim", 384)
    emb_model    = data.get("embedding_model", "all-MiniLM-L6-v2")
    chunks       = data.get("chunks_preview", [])
    last_query   = data.get("last_query", "")
    last_results = data.get("last_results", [])
    hit_ids      = {r["chunk_id"] for r in last_results}

    # ── Panel wrapper + title ─────────────────────────────────────────────────
    st.markdown(f"""
<div class="vdb-panel">
  <div class="vdb-title"><span>◈</span> {title}</div>

  <div class="vdb-stat-row">
    <div class="vdb-stat">
      <span class="vdb-stat-val">{total_chunks:,}</span>
      <span class="vdb-stat-label">Chunks</span>
    </div>
    <div class="vdb-stat">
      <span class="vdb-stat-val">{total_chars:,}</span>
      <span class="vdb-stat-label">Characters</span>
    </div>
    <div class="vdb-stat">
      <span class="vdb-stat-val">{emb_dim}d</span>
      <span class="vdb-stat-label">Embedding dim</span>
    </div>
    <div class="vdb-stat">
      <span class="vdb-stat-val">{math.ceil(total_chars / 4):,}</span>
      <span class="vdb-stat-label">Est. tokens</span>
    </div>
  </div>

  <div class="vdb-model-line">Model: {emb_model} · FAISS IndexFlatL2</div>
""", unsafe_allow_html=True)

    # ── Chunk grid ────────────────────────────────────────────────────────────
    if chunks:
        retrieved_badge = (
            f'<span class="vdb-retrieved-badge">{len(hit_ids)} retrieved</span>'
            if hit_ids else ""
        )
        st.markdown(f"""
  <div class="vdb-section-label">All Chunks {retrieved_badge}</div>
  <div class="chunk-grid">
""", unsafe_allow_html=True)

        for ch in chunks:
            is_hit   = ch["hash"] in hit_ids
            hl_cls   = "highlighted" if is_hit else ""
            src_str  = _short_src(ch.get("source", "?"))
            rank_lbl = ""
            if is_hit:
                for r in last_results:
                    if r["chunk_id"] == ch["hash"]:
                        rank_lbl = f'<span class="chunk-rank-badge">#{r["rank"]}</span>'
                        break
            st.markdown(f"""
    <div class="chunk-card {hl_cls}">
      {rank_lbl}
      <span class="chunk-idx-badge">#{ch['index']}</span>
      <div class="chunk-hash">#{ch['hash']}</div>
      <span class="chunk-src" title="{ch.get('source','?')}">{src_str}</span>
      <div class="chunk-preview">{ch.get('preview','')}</div>
      <span class="chunk-chars">{ch.get('chars', 0)} chars</span>
    </div>
""", unsafe_allow_html=True)

        st.markdown("  </div>", unsafe_allow_html=True)

    # ── Similarity results ────────────────────────────────────────────────────
    if last_results:
        results_html = ""
        for r in last_results:
            pct   = r["similarity"]
            color = _sim_color(pct)
            src   = _short_src(r.get("source", "?"))
            fill_w = f"{pct}%"
            results_html += f"""
    <div class="sim-result-card">
      <div class="sim-bg-fill" style="width:{fill_w}"></div>
      <div class="sim-rank-num">#{r['rank']}</div>
      <div class="sim-body">
        <div class="sim-meta-row">
          <span class="sim-pct-label" style="color:{color}">{pct}%</span>
          <span class="sim-cid-label">#{r['chunk_id']}</span>
          <span class="sim-src-tag" title="{r.get('source','?')}">{src}</span>
        </div>
        <div class="sim-preview-text">{r.get('preview','')}</div>
      </div>
    </div>"""

        st.markdown(f"""
  <div class="sim-section">
    <div class="sim-title">⚡ Last Retrieved Chunks</div>
    <div class="sim-query-box">Query: {last_query}</div>
    {results_html}
  </div>
""", unsafe_allow_html=True)

        # Score distribution bars
        bars_html = ""
        for r in last_results:
            pct   = r["similarity"]
            color = _sim_color(pct)
            bars_html += f"""
    <div class="score-bar-row">
      <span class="score-bar-lbl">Rank #{r['rank']}</span>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}66,{color})"></div>
      </div>
      <span class="score-bar-val" style="color:{color}">{pct}%</span>
    </div>"""

        st.markdown(f"""
  <div class="score-dist-section">
    <div class="score-dist-title">〉 Similarity Score Distribution</div>
    {bars_html}
  </div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_routing_log(log: list):
    """Show the orchestrator's routing decisions in a styled panel."""
    if not log:
        return
    _inject_css()

    method_colors = {
        "regex":            "#06b6d4",
        "llm":              "#7c6df2",
        "context_override": "#f59e0b",
    }

    rows_html = ""
    for entry in reversed(log[-10:]):
        method = entry.get("method", "?")
        agent  = entry.get("agent", "?")
        conf   = entry.get("confidence", 0)
        msg    = entry.get("message", "")[:65]
        mc     = method_colors.get(method, "#475569")
        cc     = "#4ade80" if conf >= 0.8 else "#f59e0b" if conf >= 0.5 else "#ef4444"
        rows_html += f"""
  <div class="route-row">
    <span class="route-msg">{msg}</span>
    <span class="route-agent-tag">→ {agent}</span>
    <span class="route-method-tag" style="color:{mc}">{method}</span>
    <span class="route-conf-tag" style="color:{cc}">{int(conf*100)}%</span>
  </div>"""

    st.markdown(f"""
<div class="vdb-routing-panel">
  <div class="vdb-routing-title">⟳ Orchestrator Routing Log</div>
  {rows_html}
</div>
""", unsafe_allow_html=True)
