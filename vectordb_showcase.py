"""
vectordb_showcase.py — Vector DB Showcase UI Component
Drop-in Streamlit panel — call render_vectordb_showcase(data) from your main app.

Usage in app.py:
    from vectordb_showcase import render_vectordb_showcase
    # After RAG query:
    if result.get("vstore_data"):
        render_vectordb_showcase(result["vstore_data"])
"""

import streamlit as st
import math

# ── CSS injected once ─────────────────────────────────────────────────────────
_CSS_INJECTED = False

def _inject_css():
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    _CSS_INJECTED = True
    st.markdown("""
<style>
/* ── Vector DB Showcase ──────────────────────────────────────────── */
.vdb-panel {
    background: #060c1a;
    border: 1px solid #1a2d4a;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    font-family: 'Space Mono', monospace;
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
.vdb-stat-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.vdb-stat {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.45rem 0.9rem;
    min-width: 100px;
    flex: 1;
}
.vdb-stat-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: #7c6df2;
    line-height: 1.2;
}
.vdb-stat-label {
    font-size: 0.58rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.1rem;
}

/* ── Chunk cards ── */
.chunk-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 0.5rem;
    max-height: 320px;
    overflow-y: auto;
    padding-right: 4px;
}
.chunk-card {
    background: #0a0f1e;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    cursor: default;
    transition: border-color 0.2s;
    position: relative;
}
.chunk-card:hover { border-color: #3b5bdb; }
.chunk-card.highlighted {
    border-color: #7c6df2 !important;
    background: #0d1526 !important;
    box-shadow: 0 0 0 1px #7c6df2, 0 4px 16px rgba(124,109,242,0.15);
}
.chunk-idx {
    font-size: 0.58rem;
    color: #374151;
    position: absolute;
    top: 0.4rem;
    right: 0.5rem;
}
.chunk-hash {
    font-size: 0.6rem;
    color: #3b5bdb;
    margin-bottom: 0.2rem;
}
.chunk-src {
    font-size: 0.58rem;
    color: #475569;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 200px;
    margin-bottom: 0.3rem;
}
.chunk-preview {
    font-size: 0.65rem;
    color: #94a3b8;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.chunk-chars {
    font-size: 0.58rem;
    color: #374151;
    margin-top: 0.3rem;
}

/* ── Similarity results ── */
.sim-section {
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid #1e2d4a;
}
.sim-title {
    font-size: 0.62rem;
    color: #f59e0b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}
.sim-query {
    font-size: 0.7rem;
    color: #c7d2fe;
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    margin-bottom: 0.7rem;
    word-break: break-word;
}
.sim-result {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    margin-bottom: 0.6rem;
    padding: 0.5rem 0.7rem;
    background: #0a0f1e;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    position: relative;
    overflow: hidden;
}
/* Similarity fill bar */
.sim-result::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: var(--sim-pct);
    background: linear-gradient(90deg, rgba(124,109,242,0.08), transparent);
    pointer-events: none;
}
.sim-rank {
    font-size: 0.75rem;
    font-weight: 700;
    color: #7c6df2;
    min-width: 18px;
    text-align: center;
    padding-top: 0.1rem;
}
.sim-body { flex: 1; min-width: 0; }
.sim-score-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.2rem;
    flex-wrap: wrap;
}
.sim-pct {
    font-size: 0.7rem;
    font-weight: 700;
    color: #4ade80;
}
.sim-chunk-id {
    font-size: 0.58rem;
    color: #3b5bdb;
}
.sim-src-pill {
    font-size: 0.55rem;
    color: #475569;
    background: #0d1526;
    border: 1px solid #1e2d4a;
    border-radius: 4px;
    padding: 0.05rem 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 140px;
}
.sim-preview {
    font-size: 0.63rem;
    color: #94a3b8;
    line-height: 1.4;
}

/* ── Embedding visualizer ── */
.emb-section {
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid #1e2d4a;
}
.emb-title {
    font-size: 0.62rem;
    color: #06b6d4;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}
.emb-bar-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.emb-bar-label {
    font-size: 0.6rem;
    color: #475569;
    min-width: 60px;
    text-align: right;
}
.emb-bar-wrap {
    flex: 1;
    height: 6px;
    background: #0a1628;
    border-radius: 3px;
    overflow: hidden;
}
.emb-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #3b5bdb, #7c6df2);
    transition: width 0.5s ease;
}
.emb-bar-val {
    font-size: 0.6rem;
    color: #7c6df2;
    min-width: 32px;
    text-align: right;
}

/* ── scrollbar ── */
.chunk-grid::-webkit-scrollbar { width: 3px; }
.chunk-grid::-webkit-scrollbar-track { background: #0a0f1e; }
.chunk-grid::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _short_src(src: str) -> str:
    """Shorten a file path or URL to last 28 chars."""
    if not src or src == "?":
        return "unknown"
    return ("…" + src[-26:]) if len(src) > 28 else src


def _sim_color(pct: float) -> str:
    """Map similarity % → hex color."""
    if pct >= 80: return "#4ade80"
    if pct >= 60: return "#f59e0b"
    if pct >= 40: return "#3b82f6"
    return "#ef4444"


# ── main render ───────────────────────────────────────────────────────────────

def render_vectordb_showcase(data: dict, title: str = "Vector DB Showcase"):
    """
    Render the Vector DB Showcase panel.

    Parameters
    ----------
    data : dict  — from VectorStoreWithMeta.get_showcase_data()
    title : str  — panel title (optional)
    """
    _inject_css()

    if not data:
        st.info("No vector store data available yet.")
        return

    total_chunks  = data.get("total_chunks", 0)
    total_chars   = data.get("total_chars", 0)
    emb_dim       = data.get("embedding_dim", 384)
    emb_model     = data.get("embedding_model", "all-MiniLM-L6-v2")
    chunks        = data.get("chunks_preview", [])
    last_query    = data.get("last_query", "")
    last_results  = data.get("last_results", [])

    # highlighted chunk IDs from last search
    hit_ids = {r["chunk_id"] for r in last_results}

    st.markdown(f"""
<div class="vdb-panel">
  <div class="vdb-title">
    <span>◈</span> {title}
  </div>
""", unsafe_allow_html=True)

    # ── Stat row ──────────────────────────────────────────────────────────────
    st.markdown(f"""
  <div class="vdb-stat-row">
    <div class="vdb-stat">
      <div class="vdb-stat-val">{total_chunks:,}</div>
      <div class="vdb-stat-label">Chunks</div>
    </div>
    <div class="vdb-stat">
      <div class="vdb-stat-val">{total_chars:,}</div>
      <div class="vdb-stat-label">Characters</div>
    </div>
    <div class="vdb-stat">
      <div class="vdb-stat-val">{emb_dim}d</div>
      <div class="vdb-stat-label">Embedding dim</div>
    </div>
    <div class="vdb-stat">
      <div class="vdb-stat-val">{math.ceil(total_chars / 4):,}</div>
      <div class="vdb-stat-label">Est. tokens</div>
    </div>
  </div>
  <div style="font-size:0.6rem;color:#374151;margin-bottom:0.8rem">
    Model: {emb_model} · FAISS IndexFlatL2
  </div>
""", unsafe_allow_html=True)

    # ── Chunk grid ────────────────────────────────────────────────────────────
    if chunks:
        chunk_cards = ""
        for ch in chunks:
            is_hit  = ch["hash"] in hit_ids
            hl_cls  = "highlighted" if is_hit else ""
            src_str = _short_src(ch.get("source", "?"))
            rank_label = ""
            if is_hit:
                for r in last_results:
                    if r["chunk_id"] == ch["hash"]:
                        rank_label = f'<span style="position:absolute;top:0.4rem;left:0.5rem;font-size:0.55rem;color:#7c6df2">#{r["rank"]}</span>'
                        break
            chunk_cards += f"""
<div class="chunk-card {hl_cls}">
  {rank_label}
  <span class="chunk-idx">#{ch['index']}</span>
  <div class="chunk-hash">#{ch['hash']}</div>
  <div class="chunk-src" title="{ch.get('source', '?')}">{src_str}</div>
  <div class="chunk-preview">{ch.get('preview', '')}</div>
  <div class="chunk-chars">{ch.get('chars', 0)} chars</div>
</div>"""

        st.markdown(f"""
  <div style="font-size:0.62rem;color:#475569;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.1em">
    All Chunks {f'· <span style="color:#7c6df2">{len(hit_ids)} retrieved</span>' if hit_ids else ''}
  </div>
  <div class="chunk-grid">{chunk_cards}</div>
""", unsafe_allow_html=True)

    # ── Similarity results ────────────────────────────────────────────────────
    if last_results:
        results_html = ""
        for r in last_results:
            pct   = r["similarity"]
            color = _sim_color(pct)
            src   = _short_src(r.get("source", "?"))
            results_html += f"""
<div class="sim-result" style="--sim-pct:{pct}%">
  <div class="sim-rank">#{r['rank']}</div>
  <div class="sim-body">
    <div class="sim-score-row">
      <span class="sim-pct" style="color:{color}">{pct}%</span>
      <span class="sim-chunk-id">#{r['chunk_id']}</span>
      <span class="sim-src-pill" title="{r.get('source','?')}">{src}</span>
    </div>
    <div class="sim-preview">{r.get('preview', '')}</div>
  </div>
</div>"""

        st.markdown(f"""
  <div class="sim-section">
    <div class="sim-title">⚡ Last Retrieved Chunks</div>
    <div class="sim-query">Query: {last_query}</div>
    {results_html}
  </div>
""", unsafe_allow_html=True)

    # ── Embedding space mini-viz ───────────────────────────────────────────────
    if last_results:
        # Show similarity scores as dimension bars (sampled from actual score)
        st.markdown("""
  <div class="emb-section">
    <div class="emb-title">〉 Similarity Score Distribution</div>
""", unsafe_allow_html=True)

        bars_html = ""
        for r in last_results:
            pct   = r["similarity"]
            color = _sim_color(pct)
            label = f"Rank #{r['rank']}"
            bars_html += f"""
<div class="emb-bar-row">
  <div class="emb-bar-label">{label}</div>
  <div class="emb-bar-wrap">
    <div class="emb-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}55,{color})"></div>
  </div>
  <div class="emb-bar-val" style="color:{color}">{pct}%</div>
</div>"""

        st.markdown(bars_html + "</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Routing log panel ─────────────────────────────────────────────────────────

def render_routing_log(log: list):
    """Show the orchestrator's routing decisions."""
    if not log:
        return

    _inject_css()

    st.markdown("""
<div class="vdb-panel" style="border-color:#1a3a2a">
  <div class="vdb-title" style="color:#4ade80">⟳ Orchestrator Routing Log</div>
""", unsafe_allow_html=True)

    method_colors = {
        "regex":            "#06b6d4",
        "llm":              "#7c6df2",
        "context_override": "#f59e0b",
    }
    conf_colors = {
        lambda c: c >= 0.8: "#4ade80",
        lambda c: c >= 0.5: "#f59e0b",
    }

    rows_html = ""
    for entry in reversed(log[-8:]):
        method = entry.get("method", "?")
        agent  = entry.get("agent", "?")
        conf   = entry.get("confidence", 0)
        msg    = entry.get("message", "")[:60]
        mc     = method_colors.get(method, "#475569")
        cc     = "#4ade80" if conf >= 0.8 else "#f59e0b" if conf >= 0.5 else "#ef4444"
        rows_html += f"""
<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;
  padding:0.4rem 0.6rem;background:#0a0f1e;border:1px solid #1e2d4a;border-radius:6px;
  font-family:'Space Mono',monospace;font-size:0.62rem;flex-wrap:wrap">
  <span style="color:#475569;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{msg}</span>
  <span style="background:#0d1526;border:1px solid #1e3a5f;border-radius:4px;padding:0.05rem 0.4rem;color:#c7d2fe">→ {agent}</span>
  <span style="color:{mc}">{method}</span>
  <span style="color:{cc}">{int(conf*100)}%</span>
</div>"""

    st.markdown(rows_html + "</div>", unsafe_allow_html=True)
