### Updated vectordb_showcase.py

```python
# vectordb_showcase.py — Vector DB Showcase UI Component
# Drop-in Streamlit panel — call render_vectordb_showcase(data) from your main app.

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
/* --- Vector DB Showcase: clearer layout & spacing --- */
.vdb-panel {
  background: #ffffff;
  border: 1px solid rgba(30,45,74,0.06);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-top: 1rem;
  font-family: 'Space Mono', monospace;
  color: #0f1724;
}

/* Title / meta */
.vdb-title { font-size:0.72rem; color:#2563eb; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.6rem; display:flex; align-items:center; gap:0.5rem; }
.vdb-stat-row { display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:0.9rem; }
.vdb-stat { background:#f8fbff; border:1px solid rgba(30,45,74,0.04); border-radius:10px; padding:0.5rem 0.9rem; min-width:120px; flex:1 1 120px; }
.vdb-stat-val { font-size:1.05rem; font-weight:700; color:#0b1220; }
.vdb-stat-label { font-size:0.62rem; color:#6b7280; margin-top:0.15rem; text-transform:uppercase; letter-spacing:0.08em; }

/* Chunk grid: larger cards, consistent gutters, visible scroll */
.chunk-grid {
  display:grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 0.75rem;
  max-height: 420px;
  overflow-y: auto;
  padding-right: 8px;
}
.chunk-grid::-webkit-scrollbar { width:8px; }
.chunk-grid::-webkit-scrollbar-thumb { background: rgba(59,91,219,0.12); border-radius:8px; }

/* Chunk card: more breathing room, readable preview, consistent truncation */
.chunk-card {
  background:#ffffff;
  border:1px solid rgba(30,45,74,0.04);
  border-radius:12px;
  padding:0.9rem;
  cursor:default;
  transition: box-shadow 220ms ease, border-color 220ms ease, transform 180ms ease;
  position:relative;
  min-height:110px;
  display:flex;
  flex-direction:column;
  gap:0.45rem;
}
.chunk-card:hover { box-shadow: 0 10px 30px rgba(15,23,36,0.06); transform: translateY(-4px); border-color: rgba(59,91,219,0.12); }
.chunk-card.highlighted { border-color: rgba(124,109,242,0.22); box-shadow: 0 12px 36px rgba(124,109,242,0.08); background: linear-gradient(180deg,#fbfbff,#ffffff); }

/* Meta lines */
.chunk-idx { font-size:0.62rem; color:#6b7280; position:absolute; top:0.6rem; right:0.8rem; }
.chunk-hash { font-size:0.72rem; color:#3b5bdb; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:70%; }
.chunk-src { font-size:0.72rem; color:#475569; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:100%; }
.chunk-preview {
  font-size:0.9rem;
  color:#374151;
  line-height:1.45;
  display:-webkit-box;
  -webkit-line-clamp:4;
  -webkit-box-orient:vertical;
  overflow:hidden;
  margin-top:0.15rem;
}
.chunk-chars { font-size:0.68rem; color:#6b7280; margin-top:auto; }

/* Similarity results: clear rank, score, and preview alignment */
.sim-section { margin-top:1rem; padding-top:0.8rem; border-top:1px solid rgba(30,45,74,0.04); }
.sim-title { font-size:0.68rem; color:#b45309; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.6rem; }
.sim-query { font-size:0.85rem; color:#0b1220; background:#f8fbff; border:1px solid rgba(30,45,74,0.04); border-radius:8px; padding:0.45rem 0.6rem; margin-bottom:0.7rem; }

/* Each result is a horizontal card with a subtle left fill indicating similarity */
.sim-result {
  display:flex; gap:0.8rem; align-items:flex-start; padding:0.6rem; background:#ffffff;
  border:1px solid rgba(30,45,74,0.04); border-radius:10px; position:relative; overflow:hidden;
}
.sim-result::before {
  content:''; position:absolute; left:0; top:0; bottom:0; width: var(--sim-pct); background: linear-gradient(90deg, rgba(59,91,219,0.06), transparent);
  pointer-events:none;
}
.sim-rank { font-size:0.82rem; font-weight:700; color:#3b5bdb; min-width:36px; text-align:center; }
.sim-body { flex:1; min-width:0; }
.sim-score-row { display:flex; gap:0.6rem; align-items:center; margin-bottom:0.25rem; flex-wrap:wrap; }
.sim-pct { font-size:0.82rem; font-weight:700; color:#059669; }
.sim-chunk-id { font-size:0.72rem; color:#3b5bdb; }
.sim-src-pill { font-size:0.68rem; color:#475569; background:#f1f5f9; border-radius:6px; padding:0.12rem 0.45rem; }

/* Preview text */
.sim-preview { font-size:0.9rem; color:#374151; line-height:1.45; }

/* Embedding bars: consistent height and accessible colors */
.emb-section { margin-top:1rem; padding-top:0.8rem; border-top:1px solid rgba(30,45,74,0.04); }
.emb-title { font-size:0.68rem; color:#0ea5a4; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.6rem; }
.emb-bar-row { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.45rem; }
.emb-bar-label { font-size:0.72rem; color:#6b7280; min-width:80px; text-align:right; }
.emb-bar-wrap { flex:1; height:8px; background:#f1f5f9; border-radius:6px; overflow:hidden; }
.emb-bar-fill { height:100%; border-radius:6px; background:linear-gradient(90deg,#3b82f6,#7c6df2); transition: width 420ms ease; }
.emb-bar-val { font-size:0.72rem; color:#0b1220; min-width:40px; text-align:right; }

/* Small screens: stack stats and increase card width */
@media (max-width:768px) {
  .vdb-stat-row { flex-direction:column; gap:0.5rem; }
  .chunk-grid { grid-template-columns: 1fr; max-height: 360px; }
  .chunk-card { min-height:120px; }
}

/* Smooth transitions */
.vdb-panel, .chunk-card, .sim-result, .emb-bar-fill { transition: all 220ms cubic-bezier(.2,.9,.2,1); }
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
  <div style="font-size:0.6rem;color:#6b7280;margin-bottom:0.8rem">
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
                        rank_label = f'<span style="position:absolute;top:0.6rem;left:0.8rem;font-size:0.72rem;color:#3b5bdb;font-weight:700">#{r["rank"]}</span>'
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
  <div style="font-size:0.62rem;color:#6b7280;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.1em">
    All Chunks {f'· <span style="color:#3b5bdb">{len(hit_ids)} retrieved</span>' if hit_ids else ''}
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
  padding:0.4rem 0.6rem;background:#ffffff;border:1px solid rgba(30,45,74,0.04);border-radius:6px;
  font-family:'Space Mono',monospace;font-size:0.62rem;flex-wrap:wrap">
  <span style="color:#475569;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{msg}</span>
  <span style="background:#f1f5f9;border:1px solid rgba(30,45,74,0.04);border-radius:4px;padding:0.05rem 0.4rem;color:#2563eb">→ {agent}</span>
  <span style="color:{mc}">{method}</span>
  <span style="color:{cc}">{int(conf*100)}%</span>
</div>"""

    st.markdown(rows_html + "</div>", unsafe_allow_html=True)
```
