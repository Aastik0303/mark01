"""
agents_improved.py — NexusRAG Agent Stack (Enhanced)
Improvements:
  - Vector DB showcase: chunk viewer, similarity scores, embedding metadata
  - Improved RAG: hybrid search + reranking + contextual compression
  - Better LLM prompts: structured CoT, grounded answers
  - Smarter Orchestrator: confidence-based routing with fallback
"""

from __future__ import annotations

import io, base64, json, re, subprocess, sys, tempfile, hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import google.generativeai as genai

from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
    LOADERS_OK = True
except ImportError:
    LOADERS_OK = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    TRANSCRIPT_OK = True
except ImportError:
    TRANSCRIPT_OK = False

try:
    import yt_dlp as _yt_dlp; YTDLP_OK = True
except ImportError:
    YTDLP_OK = False

try:
    import requests as _req; REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from duckduckgo_search import DDGS; DDGS_OK = True
except ImportError:
    DDGS_OK = False


# ── GOOGLE GEMINI CLIENT ──────────────────────────────────────────────────────
import streamlit as st
import google.generativeai as genai

def set_api_key(key: str = None, model: str = None):
    global GEMINI_MODEL, _api_key_store
    
    # 1. Fallback to Streamlit secrets if no arguments are provided
    if not key and "GEMINI_API_KEY" in st.secrets:
        key = st.secrets["GEMINI_API_KEY"]
        
    if not model and "GEMINI_MODEL" in st.secrets:
        model = st.secrets["GEMINI_MODEL"]
    
    # 2. Validation
    if not key or not key.strip():
        raise ValueError("GEMINI_API_KEY is not set. Please add it to your Streamlit secrets.")
        
    if not model or not model.strip():
        # You can also set a sensible default here if you prefer
        model = "gemini-2.5-flash" 
        
    # 3. Assignment
    _api_key_store = key.strip()
    GEMINI_MODEL = model.strip()
    genai.configure(api_key=_api_key_store)
def llm_call(messages: List[Dict], temperature: float = 0.1) -> str:
    """Send messages to Gemini. Auto-retries 3x on rate limit."""
    import time
    if not _api_key_store:
        return "Error: API key not set. Check your Streamlit secrets."
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                )
            )
            system_text   = ""
            chat_messages = []

            for msg in messages:
                role    = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_text = content
                elif role == "user":
                    chat_messages.append({"role": "user",  "parts": [content]})
                elif role == "assistant":
                    chat_messages.append({"role": "model", "parts": [content]})

            if system_text and chat_messages:
                for m in chat_messages:
                    if m["role"] == "user":
                        m["parts"][0] = system_text + "\n\n" + m["parts"][0]
                        break

            if not chat_messages:
                return "Error: No messages to send."

            history  = chat_messages[:-1]
            last_msg = chat_messages[-1]["parts"][0]
            chat     = model.start_chat(history=history)
            resp     = chat.send_message(last_msg)
            return resp.text or ""

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                time.sleep((attempt + 1) * 5)
                continue
            return f"Error: {e}"
    return "Error: Rate limited. Please wait a moment and try again."


# ── EMBEDDINGS ────────────────────────────────────────────────────────────────

_embedding_model = None

def get_embeddings():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


# ── VECTOR STORE HELPERS ───────────────────────────────────────────────────────

class VectorStoreWithMeta:
    """
    Wraps FAISS and stores chunk metadata for the Vector DB Showcase UI.
    Tracks: chunks, embedding norms, similarity scores from last query,
    chunk sources, and character counts.
    """

    def __init__(self):
        self.faiss_store:     Optional[FAISS] = None
        self.chunks:          List[Document]  = []
        self.chunk_hashes:    List[str]       = []
        self.last_query:      str             = ""
        self.last_results:    List[Dict]      = []   # [{doc, score, rank}]
        self.total_chars:     int             = 0
        self.embedding_dim:   int             = 384  # MiniLM-L6-v2

    def build(self, docs: List[Document]) -> str:
        """Chunk, embed, and index documents. Returns status string."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        self.chunks = splitter.split_documents(docs)

        if not self.chunks:
            return "No chunks created."

        # Assign chunk IDs and hashes
        self.chunk_hashes = [
            hashlib.md5(c.page_content.encode()).hexdigest()[:8]
            for c in self.chunks
        ]
        self.total_chars = sum(len(c.page_content) for c in self.chunks)

        emb = get_embeddings()
        self.faiss_store = FAISS.from_documents(self.chunks, emb)
        return f"Indexed {len(self.chunks)} chunks · {self.total_chars:,} chars · {self.embedding_dim}d embeddings"

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search with similarity scores, store results for UI showcase."""
        if not self.faiss_store:
            return []
        self.last_query = query
        results_with_scores = self.faiss_store.similarity_search_with_score(query, k=k)

        # Normalize scores (FAISS L2 distance → cosine-like 0-1)
        self.last_results = []
        for rank, (doc, score) in enumerate(results_with_scores):
            # Lower FAISS L2 = more similar; convert to similarity %
            sim = float(max(0.0, 1.0 - score / 2.0))
            self.last_results.append({
                "doc":        doc,
                "score":      score,
                "similarity": round(sim * 100, 1),
                "rank":       rank + 1,
                "chunk_id":   hashlib.md5(doc.page_content.encode()).hexdigest()[:8],
            })

        return [r["doc"] for r in self.last_results]

    def get_showcase_data(self) -> Dict:
        """Return all metadata needed for the Vector DB Showcase UI panel."""
        chunks_preview = []
        for i, (chunk, ch_hash) in enumerate(zip(self.chunks[:50], self.chunk_hashes[:50])):
            chunks_preview.append({
                "index":   i,
                "hash":    ch_hash,
                "source":  chunk.metadata.get("source", "?"),
                "chars":   len(chunk.page_content),
                "preview": chunk.page_content[:120].replace("\n", " ") + "…",
            })
        return {
            "total_chunks":   len(self.chunks),
            "total_chars":    self.total_chars,
            "embedding_dim":  self.embedding_dim,
            "embedding_model":"all-MiniLM-L6-v2",
            "chunks_preview": chunks_preview,
            "last_query":     self.last_query,
            "last_results":   [
                {
                    "rank":       r["rank"],
                    "chunk_id":   r["chunk_id"],
                    "similarity": r["similarity"],
                    "source":     r["doc"].metadata.get("source", "?"),
                    "preview":    r["doc"].page_content[:100].replace("\n", " ") + "…",
                }
                for r in self.last_results
            ],
        }


def load_documents(paths: List[str]) -> List[Document]:
    docs = []
    for p in paths:
        ext = Path(p).suffix.lower()
        try:
            if ext == ".pdf" and LOADERS_OK:
                docs.extend(PyPDFLoader(p).load())
            elif ext in (".txt", ".md") and LOADERS_OK:
                docs.extend(TextLoader(p, encoding="utf-8").load())
            elif ext == ".csv" and LOADERS_OK:
                docs.extend(CSVLoader(p).load())
            else:
                text = Path(p).read_text(encoding="utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": p}))
        except Exception as e:
            docs.append(Document(
                page_content=f"Error loading {p}: {e}",
                metadata={"source": p},
            ))
    return docs


# ── CHART HELPERS ──────────────────────────────────────────────────────────────

def _dark(ax, fig):
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ── IMPROVED RAG AGENT ────────────────────────────────────────────────────────

class RAGAgent:
    """
    Improved RAG Agent with:
    - VectorStoreWithMeta for showcase UI
    - Hybrid retrieval (dense + keyword overlap rerank)
    - Contextual compression (LLM extracts only relevant snippet)
    - Structured grounded prompt
    """
    name = "RAG Agent"

    def __init__(self):
        self.vstore   = VectorStoreWithMeta()
        self._sources: List[str] = []

    def ingest(self, file_paths: List[str]) -> str:
        docs = load_documents(file_paths)
        if not docs:
            return "No documents loaded."
        status        = self.vstore.build(docs)
        self._sources = list({d.metadata.get("source", "?") for d in docs})
        return f"✅ Ingested {len(docs)} pages from {len(file_paths)} file(s). {status}"

    def _rerank(self, docs: List[Document], query: str) -> List[Document]:
        """
        Simple BM25-style keyword rerank on top of dense results.
        Boosts docs that contain query keywords.
        """
        q_tokens = set(re.findall(r"\w+", query.lower()))
        def score(doc: Document) -> float:
            text = doc.page_content.lower()
            hits = sum(1 for t in q_tokens if t in text)
            return hits / (len(q_tokens) + 1e-9)
        return sorted(docs, key=score, reverse=True)

    def _compress_context(self, docs: List[Document], question: str) -> str:
        """
        LLM-based contextual compression:
        Extract only the sentence(s) from each chunk directly relevant to the question.
        Falls back to raw text if LLM fails.
        """
        combined = "\n\n---\n\n".join(
            f"[CHUNK {i+1}]\n{d.page_content}" for i, d in enumerate(docs[:5])
        )
        compressed = llm_call([
            {"role": "system", "content": (
                "You are a context extractor. "
                "Given CHUNKS and a QUESTION, extract ONLY the sentences/phrases from the chunks "
                "that are directly relevant to answering the question. "
                "Keep the original wording. Remove everything irrelevant. "
                "Format: one paragraph per chunk, prefixed with [CHUNK N]."
            )},
            {"role": "user", "content": (
                f"QUESTION: {question}\n\nCHUNKS:\n{combined}\n\n"
                "Extract only the relevant parts:"
            )},
        ], temperature=0.0)
        return compressed if not compressed.startswith("Error") else combined

    def query(self, question: str) -> Dict:
        if not self.vstore.faiss_store:
            return {"answer": "No documents loaded yet. Please upload files first.", "sources": []}

        # Dense retrieval
        docs = self.vstore.search(question, k=6)
        if not docs:
            return {"answer": "No relevant content found.", "sources": []}

        # Hybrid rerank
        docs = self._rerank(docs, question)[:5]

        # Contextual compression
        compressed_context = self._compress_context(docs, question)
        sources = list({d.metadata.get("source", "?") for d in docs})

        answer = llm_call([
            {"role": "system", "content": (
                "You are a precise Document Q&A expert.\n\n"
                "RULES:\n"
                "1. Answer ONLY from the provided CONTEXT. Never fabricate.\n"
                "2. If the answer is not in context, say: '⚠️ Not found in uploaded documents.'\n"
                "3. Cite specific details from context (e.g., 'According to [source]...').\n"
                "4. Use markdown: **bold** key facts, bullet lists for multi-part answers.\n"
                "5. Be concise — answer the question directly before elaborating.\n"
            )},
            {"role": "user", "content": (
                f"CONTEXT (relevant excerpts):\n{compressed_context}\n\n"
                f"QUESTION: {question}\n\n"
                "ANSWER (grounded in context only):"
            )},
        ])
        return {
            "answer":      answer,
            "sources":     sources,
            "vstore_data": self.vstore.get_showcase_data(),
        }

    def get_showcase_data(self) -> Dict:
        return self.vstore.get_showcase_data()


# ── YOUTUBE RAG AGENT ─────────────────────────────────────────────────────────

class VideoRAGAgent:
    """
    YouTube RAG Agent — unchanged logic, upgraded with VectorStoreWithMeta.
    """
    name = "YouTube RAG Agent"

    def __init__(self):
        self.vstore        = VectorStoreWithMeta()
        self._chunks:      list  = []
        self._transcript:  str   = ""
        self.video_id:     str   = ""
        self.video_url:    str   = ""
        self.title:        str   = "Unknown"
        self.channel:      str   = "Unknown"
        self.duration:     str   = "Unknown"
        self.thumbnail:    str   = ""
        self.source_type:  str   = "unknown"

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        for pattern in [
            r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
            r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
            r"(?:embed/)([0-9A-Za-z_-]{11})",
        ]:
            m = re.search(pattern, url)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _secs_to_ts(seconds: float) -> str:
        s = int(seconds)
        h, r = divmod(s, 3600)
        m, sec = divmod(r, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    @staticmethod
    def _parse_seg(seg) -> tuple:
        if isinstance(seg, dict):
            return seg.get("start", 0), seg.get("duration", 0), seg.get("text", "").strip()
        return getattr(seg, "start", 0), getattr(seg, "duration", 0), getattr(seg, "text", "").strip()

    def _get_metadata(self, video_id: str):
        if YTDLP_OK:
            try:
                opts = {"quiet": True, "no_warnings": True, "skip_download": True}
                with _yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info("https://www.youtube.com/watch?v=" + video_id, download=False)
                self.title    = info.get("title",    "Unknown")
                self.channel  = info.get("uploader", "Unknown")
                dur           = info.get("duration", 0)
                self.duration = self._secs_to_ts(dur) if dur else "Unknown"
                return
            except Exception:
                pass
        if REQUESTS_OK:
            try:
                r = _req.get(
                    "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v="
                    + video_id + "&format=json", timeout=8)
                if r.status_code == 200:
                    d = r.json()
                    self.title   = d.get("title",       "Unknown")
                    self.channel = d.get("author_name", "Unknown")
            except Exception:
                pass

    def _fetch_transcript(self, video_id: str, language: str = "en") -> list:
        if not TRANSCRIPT_OK:
            return []

        def _parse_all(result) -> list:
            out = []
            for seg in result:
                s, d, txt = self._parse_seg(seg)
                if str(txt).strip():
                    out.append({"start": float(s), "duration": float(d), "text": str(txt).strip()})
            return out

        for lang_list in [[language], ["en"], ["en-US"], ["en-GB"], [language, "en"]]:
            try:
                raw = _parse_all(YouTubeTranscriptApi.get_transcript(video_id, languages=lang_list))
                if raw: return raw
            except Exception:
                continue

        try:
            api = YouTubeTranscriptApi()
            for lang_list in [[language, "en"], ["en"], []]:
                try:
                    result = api.fetch(video_id, languages=lang_list) if lang_list else api.fetch(video_id)
                    raw = _parse_all(result)
                    if raw: return raw
                except Exception:
                    continue
        except Exception:
            pass

        try:
            tlist = YouTubeTranscriptApi.list_transcripts(video_id)
            all_t = list(tlist)
            def _priority(t):
                lm = t.language_code.startswith(language) or t.language_code.startswith("en")
                return (0 if not t.is_generated else 1, 0 if lm else 1)
            all_t.sort(key=_priority)
            for t_obj in all_t:
                try:
                    raw = _parse_all(t_obj.fetch())
                    if raw: return raw
                except Exception:
                    continue
            if all_t:
                try:
                    raw = _parse_all(all_t[0].translate("en").fetch())
                    if raw: return raw
                except Exception:
                    pass
        except Exception:
            pass

        return []

    def _ai_fallback_chunks(self) -> List[Document]:
        response = llm_call([
            {"role": "user", "content": (
                f"You are analysing a YouTube video.\n"
                f"Title: {self.title}\nChannel: {self.channel}\n\n"
                "Generate 12 detailed topic segments that likely appear in this video.\n"
                "Format EACH segment on its own line as:\n"
                "[Topic N] <one or two sentence description>\n\nGenerate 12 segments now:"
            )},
        ], temperature=0.3)
        lines = [l.strip() for l in response.splitlines() if l.strip().startswith("[")]
        if not lines:
            lines = [f"[Overview] Video titled '{self.title}' by {self.channel}."]
        docs = []
        for i, line in enumerate(lines):
            docs.append(Document(
                page_content=line,
                metadata={"start_sec": i * 60, "timestamp": self._secs_to_ts(i * 60), "source": self.video_url},
            ))
        return docs

    def _chunks_from_transcript(self, raw: list, chunk_secs: int = 60) -> List[Document]:
        docs = []
        cur_text, cur_start, cur_end = [], None, 0.0
        for seg in raw:
            s, d, txt = seg["start"], seg["duration"], seg["text"]
            if cur_start is None: cur_start = s
            cur_text.append(txt)
            cur_end = s + d
            if (cur_end - cur_start) >= chunk_secs:
                ts = self._secs_to_ts(cur_start)
                docs.append(Document(
                    page_content="[" + ts + "] " + " ".join(cur_text),
                    metadata={"start_sec": cur_start, "timestamp": ts, "source": self.video_url},
                ))
                cur_text, cur_start = [], None
        if cur_text and cur_start is not None:
            ts = self._secs_to_ts(cur_start)
            docs.append(Document(
                page_content="[" + ts + "] " + " ".join(cur_text),
                metadata={"start_sec": cur_start, "timestamp": ts, "source": self.video_url},
            ))
        return docs

    def ingest(self, youtube_url: str, language: str = "en") -> str:
        self.vstore = VectorStoreWithMeta()
        self._chunks = []; self._transcript = ""
        self.title = self.channel = self.duration = "Unknown"
        self.source_type = "unknown"

        vid = self.extract_video_id(youtube_url)
        if not vid:
            return "Error: Could not extract video ID from URL."

        self.video_id  = vid
        self.video_url = "https://www.youtube.com/watch?v=" + vid
        self.thumbnail = "https://img.youtube.com/vi/" + vid + "/hqdefault.jpg"
        self._get_metadata(vid)

        raw = self._fetch_transcript(vid, language)
        if raw:
            self._chunks     = raw
            self._transcript = "\n".join(
                "[" + self._secs_to_ts(s["start"]) + "] " + s["text"] for s in raw
            )
            docs             = self._chunks_from_transcript(raw)
            self.source_type = "transcript"
            source_note      = f"{len(raw)} transcript segments"
        else:
            docs             = self._ai_fallback_chunks()
            self._transcript = "\n".join(d.page_content for d in docs)
            self.source_type = "ai_generated"
            source_note      = f"{len(docs)} AI-generated segments (transcript unavailable)"

        if not docs:
            return "Error: Could not build content chunks."
        try:
            status = self.vstore.build(docs)
        except Exception as e:
            return "Error building index: " + str(e)

        return f"Loaded: {self.title} | {source_note} | {status}"

    def is_ready(self) -> bool:
        return self.vstore.faiss_store is not None

    def query(self, question: str) -> Dict:
        if not self.is_ready():
            return {"answer": "Video not loaded. Please paste a YouTube URL and click Load.", "timestamps": []}

        docs    = self.vstore.search(question, k=5)
        context = "\n\n".join(d.page_content for d in docs)
        timestamps = [
            {
                "timestamp": d.metadata.get("timestamp", ""),
                "yt_link":   self.video_url + "&t=" + str(int(d.metadata.get("start_sec", 0))) + "s",
            }
            for d in docs
        ]
        src_note = (
            "Context is from the real video transcript."
            if self.source_type == "transcript"
            else "Note: No transcript — context is AI-generated from video title/channel."
        )
        answer = llm_call([
            {"role": "system", "content": (
                "You are an expert YouTube video assistant.\n"
                f"{src_note}\n\n"
                "RULES:\n"
                "1. Answer ONLY from the provided segments.\n"
                "2. Always cite timestamps like [MM:SS] when referencing specific moments.\n"
                "3. If not in context, say 'Not covered in the available transcript.'\n"
                "4. Use markdown bullets for multi-point answers.\n"
            )},
            {"role": "user", "content": (
                f'Video: "{self.title}" by {self.channel}\n\n'
                f"Relevant segments:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )},
        ])
        return {
            "answer":      answer,
            "timestamps":  timestamps,
            "video_url":   self.video_url,
            "source_type": self.source_type,
            "vstore_data": self.vstore.get_showcase_data(),
        }

    def summarize(self, style: str = "detailed") -> Dict:
        if not self._transcript:
            return {"summary": "No video loaded.", "title": ""}
        excerpt = self._transcript[:12000]
        instr = {
            "brief":   "Write a brief 3-5 sentence summary.",
            "bullets": "List the 10 most important points as bullet points with timestamps.",
        }.get(style, "Write a structured summary: Overview, Key Topics, Main Insights, Conclusion.")
        summary = llm_call([
            {"role": "user", "content": f'Video: "{self.title}" by {self.channel}\n\nContent:\n{excerpt}\n\n{instr}'},
        ], temperature=0.2)
        return {
            "summary":     summary,
            "title":       self.title,
            "channel":     self.channel,
            "video_url":   self.video_url,
            "thumbnail":   self.thumbnail,
            "source_type": self.source_type,
        }

    def get_info(self) -> Dict:
        return {
            "video_id":            self.video_id,
            "title":               self.title,
            "channel":             self.channel,
            "duration":            self.duration,
            "video_url":           self.video_url,
            "thumbnail":           self.thumbnail,
            "transcript_segments": len(self._chunks),
            "indexed":             self.vstore.faiss_store is not None,
            "source_type":         self.source_type,
        }

    def get_showcase_data(self) -> Dict:
        return self.vstore.get_showcase_data()

    @staticmethod
    def is_youtube_url(text: str) -> bool:
        return bool(re.search(r"(youtube\.com|youtu\.be)", text, re.IGNORECASE))


# ── DATA ANALYSIS AGENT ───────────────────────────────────────────────────────

class DataAnalysisAgent:
    name = "Data Analysis Agent"

    def __init__(self):
        self.df:        Optional[pd.DataFrame] = None
        self.file_name: str = ""

    def load_data(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".csv":
                self.df = pd.read_csv(path)
            elif ext in (".xlsx", ".xls"):
                self.df = pd.read_excel(path)
            else:
                return f"Unsupported file type: {ext}"
            self.file_name = Path(path).name
            return (
                f"✅ Loaded **{self.file_name}** — "
                f"{self.df.shape[0]:,} rows × {self.df.shape[1]} cols"
            )
        except Exception as e:
            return f"Error loading data: {e}"

    def analyze(self, question: str) -> Dict:
        if self.df is None:
            return {"answer": "No data loaded. Please upload a CSV or Excel file.", "chart": None}

        profile = {
            "shape":   list(self.df.shape),
            "columns": list(self.df.columns),
            "dtypes":  {c: str(t) for c, t in self.df.dtypes.items()},
            "sample":  self.df.head(5).to_dict(orient="records"),
            "describe": self.df.describe(include="all").fillna("").to_dict(),
            "nulls":   self.df.isnull().sum().to_dict(),
        }

        answer = llm_call([
            {"role": "system", "content": (
                "You are a senior data analyst.\n\n"
                "RULES:\n"
                "1. Answer questions using the dataset profile provided.\n"
                "2. Give specific numbers, percentages, and column names.\n"
                "3. If a chart would help, describe it in a ```chart``` block as JSON:\n"
                "   {\"type\": \"bar|line|scatter|hist\", \"x\": \"col\", \"y\": \"col\", \"title\": \"...\"}\n"
                "4. Use markdown for clarity.\n"
                "5. Flag data quality issues (nulls, types) if relevant.\n"
            )},
            {"role": "user", "content": (
                f"Dataset: {self.file_name}\n"
                f"Profile:\n{json.dumps(profile, default=str, indent=2)}\n\n"
                f"Question: {question}\n\nAnalysis:"
            )},
        ])

        # Extract and render chart if LLM suggested one
        chart_b64 = None
        chart_match = re.search(r"```chart\s*([\s\S]*?)```", answer)
        if chart_match:
            try:
                spec = json.loads(chart_match.group(1))
                chart_b64 = self._render_chart(spec)
                answer = answer[:chart_match.start()].strip()
            except Exception:
                pass

        return {"answer": answer, "chart": chart_b64}

    def _render_chart(self, spec: Dict) -> Optional[str]:
        if self.df is None:
            return None
        try:
            fig, ax = plt.subplots(figsize=(9, 4))
            _dark(ax, fig)
            ctype = spec.get("type", "bar")
            x, y  = spec.get("x"), spec.get("y")
            title = spec.get("title", "")

            if ctype == "bar" and x and y and x in self.df.columns and y in self.df.columns:
                data = self.df.groupby(x)[y].mean().sort_values(ascending=False).head(15)
                ax.bar(data.index.astype(str), data.values, color="#7c6df2", alpha=0.85)
                ax.set_xlabel(x, color="white")
                ax.set_ylabel(y, color="white")
                plt.xticks(rotation=35, ha="right")
            elif ctype == "line" and x and y and x in self.df.columns and y in self.df.columns:
                ax.plot(self.df[x], self.df[y], color="#06b6d4", linewidth=2)
                ax.set_xlabel(x, color="white"); ax.set_ylabel(y, color="white")
            elif ctype == "hist" and x and x in self.df.columns:
                ax.hist(self.df[x].dropna(), bins=30, color="#3b82f6", alpha=0.8)
                ax.set_xlabel(x, color="white")
            elif ctype == "scatter" and x and y and x in self.df.columns and y in self.df.columns:
                ax.scatter(self.df[x], self.df[y], color="#f59e0b", alpha=0.6, s=15)
                ax.set_xlabel(x, color="white"); ax.set_ylabel(y, color="white")
            else:
                return None

            ax.set_title(title, color="white", fontsize=11, pad=10)
            fig.tight_layout()
            return _b64(fig)
        except Exception:
            return None


# ── CODE AGENT ────────────────────────────────────────────────────────────────

class CodeAgent:
    name = "Code Agent"

    SYSTEM = (
        "You are an expert software engineer.\n\n"
        "RULES:\n"
        "1. Write clean, idiomatic, production-ready code.\n"
        "2. Always include a brief explanation BEFORE the code block.\n"
        "3. Include inline comments for non-obvious logic.\n"
        "4. After the code, list: edge cases handled, limitations, suggested improvements.\n"
        "5. For debugging requests, identify the root cause first, then fix.\n"
        "6. Prefer standard library when possible; note external deps explicitly.\n"
    )

    def generate(self, request: str, language: str = "python") -> Dict:
        response = llm_call([
            {"role": "system",  "content": self.SYSTEM},
            {"role": "user",    "content": f"Language: {language}\n\nRequest: {request}"},
        ], temperature=0.2)

        code_match = re.search(r"```(?:\w+)?\n([\s\S]*?)```", response)
        code = code_match.group(1).strip() if code_match else ""
        return {"answer": response, "code": code, "lang": language}

    def explain(self, code: str) -> str:
        return llm_call([
            {"role": "system", "content": (
                "You are a code explainer. Break down code step by step:\n"
                "1. Overall purpose\n2. Key logic blocks\n3. Data flow\n4. Potential issues"
            )},
            {"role": "user", "content": f"Explain this code:\n\n```\n{code}\n```"},
        ])


# ── RESEARCH AGENT ────────────────────────────────────────────────────────────

class ResearchAgent:
    name = "Research Agent"

    SYSTEM = (
        "You are an expert research analyst.\n\n"
        "RULES:\n"
        "1. Synthesize information from multiple sources.\n"
        "2. Always cite sources inline as [Source N].\n"
        "3. Separate facts from opinions/analysis.\n"
        "4. Structure: Executive Summary → Findings → Analysis → Conclusion.\n"
        "5. Flag conflicting information between sources.\n"
        "6. Note recency and credibility of sources.\n"
    )

    def research(self, query: str, num_results: int = 6) -> Dict:
        if not DDGS_OK:
            answer = llm_call([
                {"role": "system", "content": self.SYSTEM},
                {"role": "user",   "content": f"Research (from training knowledge only): {query}"},
            ], temperature=0.3)
            return {"answer": answer, "sources": [], "queries": [query]}

        sub_queries = self._generate_queries(query)
        all_results = []

        for sq in sub_queries[:3]:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(sq, max_results=num_results))
                    all_results.extend(results)
            except Exception:
                pass

        seen, unique = set(), []
        for r in all_results:
            u = r.get("href", "")
            if u not in seen:
                seen.add(u); unique.append(r)

        sources = unique[:12]
        context = "\n\n".join(
            f"[Source {i+1}] {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')}"
            for i, r in enumerate(sources)
        )

        answer = llm_call([
            {"role": "system", "content": self.SYSTEM},
            {"role": "user",   "content": (
                f"Research query: {query}\n\n"
                f"Sources:\n{context}\n\n"
                "Write a comprehensive research report:"
            )},
        ], temperature=0.2)

        return {
            "answer":  answer,
            "sources": [{"title": r.get("title", ""), "url": r.get("href", "")} for r in sources],
            "queries": sub_queries,
        }

    def _generate_queries(self, query: str) -> List[str]:
        response = llm_call([
            {"role": "user", "content": (
                f"Generate 3 specific search queries to research: '{query}'\n"
                "Return ONLY a JSON array of strings, nothing else.\n"
                "Example: [\"query 1\", \"query 2\", \"query 3\"]"
            )},
        ], temperature=0.1)
        try:
            match = re.search(r"\[[\s\S]*?\]", response)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return [query]


# ── CHATBOT AGENT ─────────────────────────────────────────────────────────────

class ChatbotAgent:
    name = "Chatbot Agent"
    MAX_HISTORY = 20

    SYSTEM = (
        "You are NEXUS, an intelligent multi-agent AI assistant.\n\n"
        "PERSONALITY: Helpful, precise, and concise. Use markdown for structure.\n\n"
        "CAPABILITIES:\n"
        "- General knowledge Q&A\n"
        "- Document analysis (RAG)\n"
        "- YouTube video Q&A\n"
        "- Data analysis & charts\n"
        "- Code generation & debugging\n"
        "- Web research\n\n"
        "RULES:\n"
        "1. Be direct — answer first, explain after.\n"
        "2. If unsure, say so rather than guessing.\n"
        "3. Suggest relevant agents when appropriate (e.g., 'Upload a CSV to the Data Agent for charts').\n"
        "4. Keep responses focused and scannable.\n"
    )

    def __init__(self):
        self.history: List[Dict] = []

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        if len(self.history) > self.MAX_HISTORY * 2:
            self.history = self.history[-self.MAX_HISTORY * 2:]

        messages = [{"role": "system", "content": self.SYSTEM}] + self.history
        response = llm_call(messages, temperature=0.5)
        self.history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        self.history = []

    def get_summary(self) -> str:
        if not self.history:
            return "No conversation to summarize."
        excerpt = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in self.history[-20:]
        )
        return llm_call([
            {"role": "user", "content": f"Summarize this conversation in 3-5 bullets:\n\n{excerpt}"},
        ], temperature=0.2)


# ── IMPROVED ORCHESTRATOR ─────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    Improved orchestrator with:
    - Confidence-scored intent classification
    - Regex fast-path for unambiguous intents
    - Context-aware routing (uses loaded state)
    - Fallback chain: regex → LLM classify → chatbot
    """

    AGENT_DESCRIPTIONS = {
        "rag":      "Document Q&A, PDF questions, uploaded file search, document summary",
        "video":    "YouTube video, transcript, watch, channel, video question",
        "data":     "CSV, Excel, data analysis, chart, statistics, plot, correlation, trend",
        "code":     "code, programming, script, function, bug, debug, explain code, algorithm",
        "research": "research, web search, latest news, find information, look up, current events",
        "chat":     "general question, conversation, opinion, help, explain concept, advice",
    }

    # Regex fast-path patterns (order matters — more specific first)
    FAST_PATH = [
        ("video",    re.compile(r"youtube\.com|youtu\.be|watch\?v=|video transcript|youtube", re.I)),
        ("data",     re.compile(r"\b(csv|excel|xlsx|dataframe|chart|plot|graph|statistic|correlation|trend|dataset)\b", re.I)),
        ("code",     re.compile(r"\b(write code|generate code|fix bug|debug|function|class|script|algorithm|implement)\b", re.I)),
        ("research", re.compile(r"\b(research|web search|latest news|look up|current event|find info|search for)\b", re.I)),
        ("rag",      re.compile(r"\b(in the document|in the pdf|uploaded file|from the file|in my document|find in doc)\b", re.I)),
    ]

    def __init__(self):
        self.rag      = RAGAgent()
        self.video_rag = VideoRAGAgent()
        self.data_agent = DataAnalysisAgent()
        self.code_agent = CodeAgent()
        self.research_agent = ResearchAgent()
        self.chatbot  = ChatbotAgent()
        self.routing_log: List[Dict] = []   # track routing decisions for debug

    def _fast_path_route(self, message: str) -> Optional[str]:
        for agent_id, pattern in self.FAST_PATH:
            if pattern.search(message):
                return agent_id
        return None

    def _llm_route(self, message: str, context: Dict) -> Tuple[str, float]:
        """
        Ask LLM to classify intent with confidence score.
        Returns (agent_id, confidence 0-1).
        """
        ctx_hints = []
        if context.get("rag_ingested"):    ctx_hints.append("Documents are loaded (RAG available)")
        if context.get("video_ingested"):  ctx_hints.append("YouTube video is loaded (Video RAG available)")
        if context.get("data_loaded"):     ctx_hints.append(f"Data file loaded: {context.get('data_filename', '')}")

        desc_block = "\n".join(f"- {k}: {v}" for k, v in self.AGENT_DESCRIPTIONS.items())
        ctx_block  = "\n".join(ctx_hints) if ctx_hints else "No special context loaded."

        response = llm_call([
            {"role": "system", "content": (
                "You are an intent classifier for a multi-agent AI system.\n"
                "Respond ONLY with valid JSON: {\"agent\": \"<id>\", \"confidence\": <0.0-1.0>}\n"
                "No other text."
            )},
            {"role": "user", "content": (
                f"Available agents:\n{desc_block}\n\n"
                f"Context:\n{ctx_block}\n\n"
                f"User message: \"{message}\"\n\n"
                "Which agent should handle this? Respond with JSON only."
            )},
        ], temperature=0.0)

        try:
            match = re.search(r"\{[\s\S]*?\}", response)
            if match:
                data = json.loads(match.group())
                agent = data.get("agent", "chat")
                conf  = float(data.get("confidence", 0.5))
                if agent in self.AGENT_DESCRIPTIONS:
                    return agent, conf
        except Exception:
            pass

        return "chat", 0.3

    def route(self, message: str, context: Optional[Dict] = None) -> Dict:
        """
        Route message to best agent. Returns result dict with routing metadata.
        """
        ctx = context or {}

        # 1. Fast-path regex
        fast = self._fast_path_route(message)
        if fast:
            agent_id   = fast
            confidence = 1.0
            method     = "regex"
        else:
            # 2. LLM classification
            agent_id, confidence = self._llm_route(message, ctx)
            method = "llm"

        # 3. Context overrides — if specific data is loaded and question is generic,
        #    prefer the loaded agent
        if confidence < 0.6:
            if ctx.get("rag_ingested") and not ctx.get("video_ingested") and not ctx.get("data_loaded"):
                agent_id = "rag"
                method   = "context_override"

        # Log routing decision
        self.routing_log.append({
            "message":    message[:80],
            "agent":      agent_id,
            "confidence": confidence,
            "method":     method,
        })

        # 4. Dispatch
        result = self._dispatch(agent_id, message, ctx)
        result["_routing"] = {"agent": agent_id, "confidence": confidence, "method": method}
        return result

    def _dispatch(self, agent_id: str, message: str, ctx: Dict) -> Dict:
        try:
            if agent_id == "rag":
                return self.rag.query(message)

            elif agent_id == "video":
                if not self.video_rag.is_ready():
                    return {"answer": "No YouTube video loaded. Please paste a URL in the YouTube RAG tab.", "agent": "video"}
                return self.video_rag.query(message)

            elif agent_id == "data":
                return self.data_agent.analyze(message)

            elif agent_id == "code":
                lang = "python"
                for l in ["javascript", "typescript", "java", "go", "rust", "sql", "bash", "c++", "c#"]:
                    if l in message.lower():
                        lang = l; break
                return self.code_agent.generate(message, language=lang)

            elif agent_id == "research":
                return self.research_agent.research(message)

            else:  # chat
                answer = self.chatbot.chat(message)
                return {"answer": answer}

        except Exception as e:
            return {"answer": f"⚠️ Agent error: {e}\n\nFalling back to general chat.", "agent": "chat"}

    def get_routing_log(self) -> List[Dict]:
        return self.routing_log[-20:]
