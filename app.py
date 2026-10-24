"""
Medical Literature Q&A — single-file demo
=========================================

Ask a medical question in plain English. The app:
    1. searches PubMed abstracts stored in a Chroma database, two ways:
       meaning-based vector search (FAISS) and keyword search (BM25),
    2. merges the two rankings with Reciprocal Rank Fusion,
    3. fetches the winning records from Chroma by ID,
    4. sends them to Google Gemini, which answers using ONLY them and cites each
       claim as [PMID:12345].

Who does what:
    Chroma  = the database. Stores every article's text, metadata (title,
              journal, year, MeSH terms) and vector on disk (chroma_db/).
              Answers "which records match this filter?" and "give me
              records with these IDs".
    FAISS   = the search engine. Loaded from Chroma's vectors at startup;
              finds the closest vectors very fast and returns their IDs.
    BM25    = keyword search over the same texts.

Run it:
    streamlit run app.py
"""

import math
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import sys

if sys.version_info < (3, 10):
    sys.exit(f"Python {sys.version.split()[0]} detected. Use Python 3.10+: See README.md.")

try:
    import numpy as np
except ImportError:
    sys.exit("Packages not found. Activate the venv and run: pip install -r requirements.txt")

# ── Settings ────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
DB_DIR = HERE / "chroma_db"     # Chroma saves the database here (created automatically)
COLLECTION = "pubmed_articles"  # Chroma's name for our table of articles

# Optional .env file in this folder (GEMINI_API_KEY, NCBI_EMAIL, GEMINI_MODEL)
try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass

# Also sync Streamlit Cloud secrets (st.secrets) to os.environ
try:
    import streamlit as st
    for _k, _v in getattr(st, "secrets", {}).items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass

# PubMedBERT fine-tuned for search (MS MARCO). Downloads ~440 MB on first use.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")
# Gemini Flash is available on Google's free tier (key from aistudio.google.com)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Tried in order if the main model is busy (503) or rate-limited (429). Also free-tier models.
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite").split(",") if m.strip()]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RRF_K = 60  # Reciprocal Rank Fusion constant from the original paper


# ════════════════════════════════════════════════════════════════════
# STEP 1 — Download abstracts from PubMed
# ════════════════════════════════════════════════════════════════════

def pubmed_queries(topic: str) -> list[str]:
    """Searches to try, most specific first. If PubMed rejects one, the next is tried."""
    if "[" in topic:  # already written in PubMed syntax — use as is
        return [topic]
    return [
        f'"{topic}"[Title/Abstract] AND hasabstract AND english[Language] '
        f'AND journal article[Publication Type]',
        f'"{topic}"[Title/Abstract] AND hasabstract',
        topic,  # plainest form; records without an abstract are skipped when parsing
    ]


class PubMedError(RuntimeError):
    pass


def search_pubmed(query: str, count: int, email: str) -> list[str]:
    """ESearch: which article IDs (PMIDs) match the query? Max 9,999 per query."""
    import requests
    resp = requests.get(f"{EUTILS}/esearch.fcgi", params={
        "db": "pubmed", "term": query, "retmax": min(count, 9_999),
        "sort": "relevance", "retmode": "json",
        "tool": "medrag-demo", "email": email,
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("esearchresult", {})
    if "idlist" not in result:
        # PubMed sends its reason instead of a list, e.g. {"ERROR": "..."} or a rate-limit note
        reason = result.get("ERROR") or result.get("errorlist") or data.get("error") or data
        raise PubMedError(f"PubMed rejected the search {query!r}: {reason}")
    return result["idlist"]


def fetch_articles(pmids: list[str], email: str) -> list[dict]:
    """EFetch: download full records for up to 200 PMIDs and parse the XML."""
    import requests
    for attempt in range(3):  # retry with backoff: 1 s, 2 s, 4 s
        try:
            resp = requests.post(f"{EUTILS}/efetch.fcgi", data={
                "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
                "tool": "medrag-demo", "email": email,
            }, timeout=120)
            resp.raise_for_status()
            return parse_pubmed_xml(resp.text)
        except Exception:
            time.sleep(2 ** attempt)
    return []


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Turn PubMed XML into simple dicts. Skips records without an abstract."""
    articles = []
    root = ET.fromstring(xml_text)
    for rec in root.iter("PubmedArticle"):
        medline = rec.find("MedlineCitation")
        art = medline.find("Article") if medline is not None else None
        if art is None:
            continue
        title_el = art.find("ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        # Structured abstracts come in labelled parts (BACKGROUND, METHODS, ...)
        parts = []
        for p in art.findall("Abstract/AbstractText"):
            text = "".join(p.itertext()).strip()
            if text:
                label = p.get("Label")
                parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(parts)
        if not title or not abstract:
            continue
        year = (art.findtext("Journal/JournalIssue/PubDate/Year")
                or (art.findtext("Journal/JournalIssue/PubDate/MedlineDate") or "")[:4])
        articles.append({
            "pmid": medline.findtext("PMID"),
            "title": title,
            "abstract": abstract,
            "journal": art.findtext("Journal/Title") or "",
            "year": int(year) if year.isdigit() else 0,
            "mesh": [d.text for d in medline.findall("MeshHeadingList/MeshHeading/DescriptorName") if d.text],
        })
    return articles


def download_articles(topic: str, count: int, email: str, progress=None) -> list[dict]:
    """Search, then download in batches of 200, pausing to respect NCBI's limit (3 req/s)."""
    pmids, problems = [], []
    for query in pubmed_queries(topic):
        try:
            pmids = search_pubmed(query, count, email)
        except PubMedError as e:
            problems.append(str(e))
            time.sleep(1)
            continue
        if pmids:
            break
    if not pmids:
        raise PubMedError("No articles found. " + " | ".join(problems))
    articles: dict[str, dict] = {}  # keyed by PMID, so no duplicates reach the database
    for i in range(0, len(pmids), 200):
        for a in fetch_articles(pmids[i:i + 200], email):
            articles[a["pmid"]] = a
        time.sleep(0.34)
        if progress:
            progress(min(i + 200, len(pmids)) / max(len(pmids), 1),
                     f"Downloaded {len(articles):,} abstracts")
    return list(articles.values())


# ════════════════════════════════════════════════════════════════════
# STEP 2 — Embed each abstract and store everything in Chroma (the database)
# ════════════════════════════════════════════════════════════════════

def article_text(a: dict) -> str:
    """What gets embedded and searched: title + abstract (one chunk per article)."""
    return f"{a['title']}. {a['abstract']}"


def embed(model, texts: list[str]) -> np.ndarray:
    """768 numbers per text, scaled to length 1 so cosine similarity = dot product."""
    return np.asarray(model.encode(texts, batch_size=32, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False),
                      dtype=np.float32)


def open_database():
    """Connect to the Chroma database saved in chroma_db/ (created if missing)."""
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(DB_DIR),
                                     settings=Settings(anonymized_telemetry=False))


def store_articles(client, articles: list[dict], vectors: np.ndarray) -> None:
    """Replace the collection with these articles: ID = PMID, plus text, metadata and vector."""
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(articles), 500):  # Chroma limits batch size
        batch = articles[start:start + 500]
        collection.add(
            ids=[a["pmid"] for a in batch],
            documents=[article_text(a) for a in batch],
            embeddings=vectors[start:start + len(batch)].tolist(),
            # Chroma metadata must be str / int / float / bool, so MeSH becomes one string
            metadatas=[{"title": a["title"], "journal": a["journal"], "year": a["year"],
                        "mesh": ", ".join(a["mesh"])} for a in batch],
        )


# ════════════════════════════════════════════════════════════════════
# STEP 3 — Load Chroma's vectors into FAISS (search engine) and BM25
# ════════════════════════════════════════════════════════════════════

STOPWORDS = frozenset("""a an and are as at be by for from has have in is it its of on or
that the this to was were with what which who how do does can""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase words, hyphenated terms kept whole (sars-cov-2), stopwords dropped."""
    return [t for t in re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())
            if t not in STOPWORDS]


@dataclass
class Library:
    collection: object      # Chroma: the database (source of truth)
    faiss_index: object     # FAISS: the search engine; row i holds the vector of ids[i]
    ids: list[str]          # FAISS row number → Chroma record ID (the PMID)
    row_of: dict[str, int]  # Chroma record ID → FAISS row number
    metas: list[dict]       # metadata per row (for filter options and evaluation)
    bm25: object            # keyword index over the same texts


def load_library(client) -> Library | None:
    """Read every record from Chroma, then build FAISS and BM25 from it."""
    import faiss
    from rank_bm25 import BM25Okapi
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        return None
    if collection.count() == 0:
        return None

    ids, vectors, metas, texts = [], [], [], []
    for offset in range(0, collection.count(), 1000):  # read the database in pages
        page = collection.get(limit=1000, offset=offset,
                              include=["embeddings", "metadatas", "documents"])
        ids += page["ids"]
        vectors.append(np.asarray(page["embeddings"], dtype=np.float32))
        metas += page["metadatas"]
        texts += page["documents"]
    vectors = np.vstack(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])  # exact inner product (= cosine) search
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return Library(collection, index, ids, {pid: i for i, pid in enumerate(ids)}, metas, bm25)


# ════════════════════════════════════════════════════════════════════
# STEP 4 — Search: FAISS (meaning) + BM25 (keywords), merged; records fetched from Chroma
# ════════════════════════════════════════════════════════════════════

@dataclass
class Filters:
    """Optional limits on which articles may be returned."""
    year_min: int | None = None
    year_max: int | None = None
    journal: str | None = None

    def active(self) -> bool:
        return any(v is not None for v in (self.year_min, self.year_max, self.journal))

    def chroma_where(self) -> dict | None:
        """The limits written as a Chroma 'where' clause (a database query)."""
        conds = []
        if self.year_min is not None:
            conds.append({"year": {"$gte": self.year_min}})
        if self.year_max is not None:
            conds.append({"year": {"$lte": self.year_max}})
        if self.journal:
            conds.append({"journal": self.journal})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}


def allowed_rows(lib: Library, filters: Filters | None) -> np.ndarray | None:
    """Ask Chroma which records match the filter; return their FAISS row numbers."""
    if not filters or not filters.active():
        return None  # no filter: everything allowed
    matching_ids = lib.collection.get(where=filters.chroma_where(), include=[])["ids"]
    return np.array([lib.row_of[i] for i in matching_ids], dtype=np.int64)


def faiss_search(lib: Library, query_vec: np.ndarray, k: int,
                 allowed: np.ndarray | None = None) -> list[int]:
    """FAISS: row numbers of the k vectors closest to the question (only allowed rows)."""
    import faiss
    params = None
    if allowed is not None:
        if len(allowed) == 0:
            return []
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
        k = min(k, len(allowed))
    k = min(k, lib.faiss_index.ntotal)
    _, rows = lib.faiss_index.search(query_vec.reshape(1, -1), k, params=params)
    return [int(r) for r in rows[0] if r != -1]


def keyword_search(lib: Library, question: str, k: int,
                   allowed: np.ndarray | None = None) -> list[int]:
    """BM25: row numbers of the k texts with the best keyword score (score > 0)."""
    scores = lib.bm25.get_scores(tokenize(question))
    if allowed is not None:  # rows the filter excludes can't win
        mask = np.full(len(scores), -1.0)
        mask[allowed] = scores[allowed]
        scores = mask
    order = np.argsort(-scores)[:k]
    return [int(r) for r in order if scores[r] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Each appearance at position r (1 = top) adds 1/(60 + r). Found by both → rises."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank)
    return scores


def fetch_records(lib: Library, ids: list[str]) -> list[dict]:
    """Chroma: get full records by ID, put back in the order asked for."""
    got = lib.collection.get(ids=ids, include=["documents", "metadatas"])
    by_id = {pid: (doc, meta) for pid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])}
    records = []
    for pid in ids:  # Chroma doesn't return records in request order
        doc, meta = by_id[pid]
        records.append({"pmid": pid, "title": meta["title"], "journal": meta["journal"],
                        "year": meta["year"], "text": doc})
    return records


def search(lib: Library, model, question: str, top_k: int = 8, mode: str = "hybrid",
           filters: Filters | None = None, timings: dict | None = None) -> list[dict]:
    """Return the top_k articles, fetched from Chroma, with how each method ranked them."""
    t = time.perf_counter()
    allowed = allowed_rows(lib, filters)                 # Chroma: which records qualify
    t_filter = time.perf_counter()

    fetch_k = top_k * 2  # over-fetch so fusion has candidates from both lists
    dense = faiss_search(lib, embed(model, [question])[0], fetch_k, allowed) if mode != "keyword" else []
    keyword = keyword_search(lib, question, fetch_k, allowed) if mode != "dense" else []
    fused = reciprocal_rank_fusion([r for r in (dense, keyword) if r])
    ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
    t_search = time.perf_counter()

    records = fetch_records(lib, [lib.ids[r] for r in ranked]) if ranked else []  # Chroma
    t_fetch = time.perf_counter()

    if timings is not None:
        timings.update(filter_ms=(t_filter - t) * 1000, search_ms=(t_search - t_filter) * 1000,
                       fetch_ms=(t_fetch - t_search) * 1000)
    return [{
        "row": r,
        "article": rec,
        "dense_rank": dense.index(r) + 1 if r in dense else None,
        "keyword_rank": keyword.index(r) + 1 if r in keyword else None,
        "rrf_score": fused[r],
    } for r, rec in zip(ranked, records)]


# ════════════════════════════════════════════════════════════════════
# STEP 5 — Ask Gemini, grounded in the retrieved abstracts
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a medical literature research assistant. Answer using ONLY the research excerpts provided.

Rules:
1. Use only the excerpts. If they don't contain enough information, say "Based on the available literature, I cannot fully answer this question" and say what is missing.
2. Cite every claim as [PMID:number]. Several sources: [PMID:111, PMID:222].
3. If sources disagree, present both sides with citations.
4. Do not use outside knowledge.
5. For treatments or doses, note that clinical guidelines should be checked.

Format: a direct 1–2 sentence answer, then supporting details with citations."""


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        a = h["article"]
        blocks.append(f"[Source {n}] PMID: {a['pmid']} | {a['journal']} | {a['year']}\n{a['text']}")
    context = "\n---\n".join(blocks)
    return (f"Research excerpts:\n{context}\n\nQuestion: {question}\n\n"
            f"Remember: cite every claim with [PMID:number] and use only the excerpts above.")


def _error_code(e: Exception) -> int | None:
    """HTTP status of a Gemini error (the SDK's errors carry .code; fall back to the text)."""
    code = getattr(e, "code", None)
    if isinstance(code, int):
        return code
    for c in (503, 500, 429, 404, 403, 401, 400):
        if str(c) in str(e):
            return c
    return None


def stream_answer(question: str, hits: list[dict], api_key: str, used: dict | None = None):
    """
    Yield Gemini's answer piece by piece as it is written.

    The free tier is sometimes busy (503) or rate-limited (429). Before any text has
    been shown, such errors are retried with a short wait (2 s, then 5 s), and then
    the next model in GEMINI_FALLBACK_MODELS is tried. `used["model"]` records which
    model answered.
    """
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=1500)
    prompt = build_prompt(question, hits)

    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    last_error = None
    for model in models:
        for wait in (0, 2, 5):  # first try, then two retries
            if wait:
                time.sleep(wait)
            started = False
            try:
                for chunk in client.models.generate_content_stream(
                        model=model, contents=prompt, config=config):
                    if chunk.text:  # some chunks carry no text (e.g. the final one)
                        if used is not None:
                            used["model"] = model
                        started = True
                        yield chunk.text
                return
            except Exception as e:
                code = _error_code(e)
                if started or code not in (429, 500, 503, 404):
                    raise  # mid-answer failure, or a problem retrying won't fix (e.g. bad key)
                last_error = e
                if code == 404:
                    break  # this model name doesn't exist: go straight to the next model
    raise last_error


def friendly_error(e: Exception) -> str:
    """Turn common Gemini API errors into plain advice."""
    msg = str(e)
    code = _error_code(e)
    if code in (503, 500) or "UNAVAILABLE" in msg:
        return ("Gemini is overloaded right now (all models tried, with retries). "
                "This is temporary on Google's side: wait 30 seconds and ask again.")
    if code == 429 or "RESOURCE_EXHAUSTED" in msg:
        return ("Gemini's free-tier limit was reached. Wait a minute and ask again "
                "(limits are shown per project in Google AI Studio).")
    if code in (401, 403) or "API key" in msg or "API_KEY_INVALID" in msg:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
    if code == 404:
        return ("None of the Gemini models were found. Set GEMINI_MODEL in .env to a model "
                "listed in Google AI Studio.")
    return f"Gemini request failed: {msg}"


def cited_pmids(answer: str) -> set[str]:
    return set(re.findall(r"PMID:\s*(\d+)", answer))


# ════════════════════════════════════════════════════════════════════
# STEP 6 — Measure search quality (known-item search) and speed
# ════════════════════════════════════════════════════════════════════

def evaluate(lib: Library, model, n_queries: int = 100, k: int = 10, seed: int = 42) -> list[dict]:
    """
    Use each sampled article's TITLE as the question; the right answer is that article.
    NDCG@k = 1/log2(rank+1) when found in the top k; MRR = 1/rank; Recall@k = found or not.
    Caveat: titles share exact words with their own abstracts, so this favours keyword search.
    """
    sample = random.Random(seed).sample(range(len(lib.ids)), min(n_queries, len(lib.ids)))
    rows = []
    for mode, label in (("keyword", "Keyword (BM25)"), ("dense", "Meaning (FAISS)"),
                        ("hybrid", "Hybrid (both, RRF)")):
        ndcg = mrr = recall = 0.0
        start = time.perf_counter()
        for r in sample:
            ranked = [h["row"] for h in search(lib, model, lib.metas[r]["title"], k, mode)]
            if r in ranked:
                rank = ranked.index(r) + 1
                ndcg += 1 / math.log2(rank + 1)
                mrr += 1 / rank
                recall += 1
        n = len(sample)
        rows.append({"Method": label, f"NDCG@{k}": round(ndcg / n, 3),
                     f"Recall@{k}": round(recall / n, 3), "MRR": round(mrr / n, 3),
                     "ms per query": round((time.perf_counter() - start) * 1000 / n, 1)})
    return rows


def compare_search_engines(lib: Library, model, n_queries: int = 100, seed: int = 42) -> list[dict]:
    """Why FAISS does the searching: time FAISS vs Chroma's own vector query, same vectors."""
    sample = random.Random(seed).sample(range(len(lib.ids)), min(n_queries, len(lib.ids)))
    vecs = embed(model, [lib.metas[r]["title"] for r in sample])
    rows = []
    for name, fn in (
        ("FAISS (search engine)", lambda v: faiss_search(lib, v, 10)),
        ("Chroma's own vector query", lambda v: lib.collection.query(
            query_embeddings=[v.tolist()], n_results=10, include=[])),
    ):
        times = []
        for v in vecs:
            t = time.perf_counter()
            fn(v)
            times.append((time.perf_counter() - t) * 1000)
        rows.append({"Searched with": name, "Mean ms": round(float(np.mean(times)), 3),
                     "p95 ms": round(float(np.percentile(times, 95)), 3)})
    return rows


# ════════════════════════════════════════════════════════════════════
# The web page (Streamlit)
# ════════════════════════════════════════════════════════════════════

EXAMPLES = [
    "What are the main risk factors for heart failure?",
    "Do SGLT2 inhibitors like dapagliflozin help heart failure patients?",
    "What is the best treatment for migraine?",  # off-topic: should refuse
]


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Medical Literature Q&A", page_icon="🩺", layout="wide")

    @st.cache_resource(show_spinner="Loading the embedding model (first time downloads ~440 MB)...")
    def get_model():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)

    @st.cache_resource
    def get_client():
        return open_database()

    @st.cache_resource(show_spinner="Loading articles from Chroma into FAISS and BM25...")
    def get_library():
        return load_library(get_client())

    st.title("🩺 Medical Literature Q&A")
    st.caption("Question → FAISS (meaning) + BM25 (keywords) → merge → fetch records from the "
               "Chroma database → Gemini answers using only those abstracts, with citations")

    lib = get_library()

    # ── Sidebar: database + settings ────────────────────────────────
    with st.sidebar:
        st.header("Database")
        if lib:
            st.success(f"{len(lib.ids):,} PubMed articles in Chroma · loaded into FAISS")
        else:
            st.warning("The database is empty. Build it below.")

        with st.expander("Build / rebuild the database", expanded=lib is None):
            topic = st.text_input("Topic", "heart failure")
            count = st.number_input("Number of articles", 100, 5000, 1000, step=100)
            email = st.text_input("Your email (required by PubMed)", os.getenv("NCBI_EMAIL", ""))
            if st.button("Build database", type="primary", disabled=not email):
                bar = st.progress(0.0, "Searching PubMed...")
                try:
                    articles = download_articles(topic, int(count), email,
                                                 progress=lambda p, msg: bar.progress(p * 0.5, msg))
                except Exception as e:
                    st.error(f"Download failed: {e}")
                    st.stop()
                if not articles:
                    st.error("PubMed returned no articles. Try another topic.")
                    st.stop()
                bar.progress(0.5, f"Embedding {len(articles):,} abstracts (a few minutes)...")
                vectors = embed(get_model(), [article_text(a) for a in articles])
                bar.progress(0.9, "Saving to Chroma...")
                store_articles(get_client(), articles, vectors)
                bar.progress(1.0, "Done")
                get_library.clear()  # reload FAISS and BM25 from the new database
                st.rerun()

        st.header("Settings")
        top_k = st.slider("Abstracts sent to Gemini", 3, 15, 8)
        mode_label = st.radio("Search method", ["Hybrid (both)", "Meaning only", "Keywords only"])
        mode = {"Hybrid (both)": "hybrid", "Meaning only": "dense", "Keywords only": "keyword"}[mode_label]
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                   or st.text_input("Gemini API key (free at aistudio.google.com)", type="password"))

        filters = Filters()
        if lib:
            st.header("Filters")
            st.caption("A filter runs as a Chroma database query first; FAISS and BM25 then "
                       "search only the matching records.")
            years = sorted({m["year"] for m in lib.metas if m["year"]})
            if len(years) > 1:
                lo, hi = st.slider("Publication years", years[0], years[-1], (years[0], years[-1]))
                if (lo, hi) != (years[0], years[-1]):
                    filters.year_min, filters.year_max = lo, hi
            counts: dict[str, int] = {}
            for m in lib.metas:
                counts[m["journal"]] = counts.get(m["journal"], 0) + 1
            journal = st.selectbox("Journal", ["All journals"] + sorted(counts, key=counts.get, reverse=True)[:30])
            if journal != "All journals":
                filters.journal = journal

    if not lib:
        st.info("👈 Build the database first: pick a topic, enter your email and click **Build database**.")
        return

    model = get_model()
    ask_tab, eval_tab = st.tabs(["💬 Ask a question", "📊 Evaluate search"])

    # ── Ask tab ─────────────────────────────────────────────────────
    with ask_tab:
        def use_example(q):
            st.session_state.question = q

        cols = st.columns(len(EXAMPLES))
        for col, q in zip(cols, EXAMPLES):
            col.button(q, on_click=use_example, args=(q,), width="stretch")

        with st.form("ask"):
            question = st.text_input("Your question", key="question",
                                     placeholder="e.g. What are the main risk factors for heart failure?")
            submitted = st.form_submit_button("Search and answer", type="primary")

        if submitted and question.strip():
            timings: dict = {}
            hits = search(lib, model, question, top_k, mode, filters, timings)
            pipeline = (f"{'Chroma filter ' + format(timings['filter_ms'], '.0f') + ' ms · ' if filters.active() else ''}"
                        f"FAISS + BM25 search {timings['search_ms']:.0f} ms · "
                        f"Chroma fetch {timings['fetch_ms']:.0f} ms")

            st.subheader("Answer")
            answer = ""
            if not hits:
                st.warning("No matching abstracts found. Try a broader question or remove filters.")
            elif not api_key:
                st.info("Add a free Gemini API key (sidebar or GEMINI_API_KEY in .env) to generate an answer. "
                        "The retrieved sources are shown below.")
                st.caption(pipeline)
            else:
                try:
                    t1 = time.perf_counter()
                    used: dict = {}
                    with st.spinner("Asking Gemini (retries automatically if it's busy)..."):
                        answer = st.write_stream(stream_answer(question, hits, api_key, used))
                    st.caption(f"{pipeline} · {used.get('model', GEMINI_MODEL)} "
                               f"{time.perf_counter() - t1:.1f} s · {mode_label}")
                except Exception as e:
                    st.error(friendly_error(e))

            cited = cited_pmids(answer or "")
            st.subheader("Sources (fetched from Chroma)")
            for n, h in enumerate(hits, 1):
                a = h["article"]
                mark = "✅ cited" if a["pmid"] in cited else "not cited"
                st.markdown(f"**{n}. [{a['title']}](https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/)**  \n"
                            f"PMID {a['pmid']} · {a['journal']} · {a['year']} · {mark}")

            with st.expander("How the search worked"):
                st.markdown(
                    ("1. **Chroma** found the records matching your filter.\n" if filters.active() else "")
                    + "1. **FAISS** ranked articles by meaning; **BM25** ranked them by keywords.\n"
                    "1. **RRF** merged the two rankings: each list adds 1/(60 + rank).\n"
                    "1. The winners' IDs were looked up in **Chroma** to fetch their full records.")
                st.dataframe([{
                    "#": n, "PMID": h["article"]["pmid"],
                    "Title": h["article"]["title"][:90],
                    "FAISS rank": str(h["dense_rank"] or "—"),
                    "BM25 rank": str(h["keyword_rank"] or "—"),
                    "RRF score": round(h["rrf_score"], 5),
                } for n, h in enumerate(hits, 1)], hide_index=True, width="stretch")

    # ── Evaluate tab ────────────────────────────────────────────────
    with eval_tab:
        st.markdown("**Known-item test:** each sampled article's title is used as the question; "
                    "the search succeeds if that article comes back near the top.")
        n = st.slider("Test questions", 20, 300, 100, step=20)
        if st.button("Run evaluation"):
            with st.spinner("Running all three search methods..."):
                st.dataframe(evaluate(lib, model, n), hide_index=True, width="stretch")
            st.caption("NDCG@10: 1.0 if the article is ranked first, lower further down, 0 if missing. "
                       "MRR: 1 / rank of the article. Titles share words with their own abstracts, "
                       "so this test favours keyword search; a labelled benchmark would be fairer.")
            st.markdown("**Why FAISS does the searching:** the same question vectors searched by "
                        "FAISS and by Chroma's own vector query.")
            st.dataframe(compare_search_engines(lib, model, n), hide_index=True, width="stretch")


if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as _running_in_streamlit
    except ImportError:
        sys.exit("Streamlit isn't installed in this Python. Activate the venv and run: "
                 "pip install -r requirements.txt")
    if _running_in_streamlit():
        main()
    else:
        # Started with "python app.py" (e.g. VS Code's Run button): relaunch it the
        # right way, as "streamlit run app.py", using this same Python and venv.
        import subprocess
        print("Starting the app with Streamlit... (press Ctrl+C here to stop it)")
        try:
            sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__,
                                         "--browser.gatherUsageStats", "false"]))
        except KeyboardInterrupt:
            pass

