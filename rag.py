"""
rag.py - SOP retrieval + LLM work orders with citations (NexAI Hack 2026, PS 2.1).
ML decides which machine; RAG tells the technician how to fix it.
Retrieval: TF-IDF over SOP sections (sops/*.md). Generation: Groq (OpenAI-compatible) via requests,
with a template fallback built only from the retrieved SOP text when there is no key or the call fails.
"""
import os
import re
from pathlib import Path

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).parent
SOP_DIR = HERE / "sops"
SOP_ID = {"osf": "SOP-OSF", "twf": "SOP-TWF", "hdf": "SOP-HDF", "pwf": "SOP-PWF"}
MODE_NAME = {"twf": "Tool wear", "hdf": "Heat dissipation", "pwf": "Power", "osf": "Overstrain"}
TOP_K, MODE_BOOST, MIN_SCORE = 3, 0.05, 0.08
ESCALATE = "Insufficient SOP evidence. Escalate to the shift supervisor."
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _load_env_file(path=HERE / ".env"):
    """Minimal KEY=VALUE .env loader (no python-dotenv). Real env vars win."""
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_env_file()


def _api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:  # Streamlit Cloud: App settings -> Secrets
        import streamlit as st
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""


# ---------- knowledge base ----------
def _load_chunks():
    """One chunk per '## ' section. id = 'SOP-OSF §2'."""
    chunks = []
    for f in sorted(SOP_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        m = re.match(r"#\s+(SOP-[A-Z]+):\s*(.+)", text)
        sop, name = m.group(1), m.group(2).strip()
        for sec in re.split(r"^## ", text, flags=re.M)[1:]:
            heading, _, body = sec.partition("\n")
            num = heading.split(".")[0].strip()
            chunks.append({"id": f"{sop} §{num}", "sop": sop, "title": f"{name}: {heading.strip()}",
                           "heading": heading.split(".", 1)[-1].strip(), "body": " ".join(body.split())})
    return chunks


CHUNKS = _load_chunks()
_VEC = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_MAT = _VEC.fit_transform([f"{c['sop']} {c['title']} {c['body']}" for c in CHUNKS])


def retrieve(query, mode=None):
    """Top-k chunks by cosine similarity (+boost for the predicted mode's SOP).
    Returns (status, hits); hits start with the best SOP's Safety section."""
    scores = cosine_similarity(_VEC.transform([query]), _MAT)[0]
    boosted = [s + (MODE_BOOST if c["sop"] == SOP_ID.get(mode) else 0) for c, s in zip(CHUNKS, scores)]
    order = sorted(range(len(CHUNKS)), key=lambda i: -boosted[i])
    if boosted[order[0]] < MIN_SCORE:
        return "escalate", []
    top = order[:TOP_K]
    best_sop = CHUNKS[top[0]]["sop"]
    safety = next(i for i, c in enumerate(CHUNKS) if c["sop"] == best_sop and c["heading"] == "Safety")
    picked = [safety] + [i for i in top if i != safety]
    return "ok", [{**CHUNKS[i], "score": round(float(boosted[i]), 3)} for i in picked]


# ---------- generation ----------
def _diagnosis(meta):
    return (f"{meta.get('machine_id', 'Machine')}: {MODE_NAME.get(meta.get('mode'), 'Unknown')} failure risk "
            f"{float(meta.get('p', 0)):.0%}, driven by {meta.get('causes') or 'unspecified causes'}.")


def _template(meta, hits):
    """Work order from retrieved SOP text only (no LLM)."""
    ctx = [c for c in hits if c["heading"] == "When this applies"]
    rest = sorted((c for c in hits[1:] if c not in ctx), key=lambda c: int(c["id"].split("§")[1]))
    steps = [hits[0]] + rest  # hits[0] is always the Safety section
    out = f"**Diagnosis:** {_diagnosis(meta)}\n\n"
    out += "".join(f"*Context:* {c['body']} [{c['id']}]\n\n" for c in ctx)
    return out + "\n".join(f"{n}. **{c['heading']}:** {c['body']} [{c['id']}]" for n, c in enumerate(steps, 1))


def _llm(meta, hits, key):
    excerpts = "\n\n".join(f"[{c['id']}] {c['title']}\n{c['body']}" for c in hits)
    prompt = (
        f"Machine: {meta.get('machine_id')} | Technician: {meta.get('tech')} | "
        f"Predicted failure: {MODE_NAME.get(meta.get('mode'), meta.get('mode'))} | "
        f"Failure risk: {float(meta.get('p', 0)):.0%} | Top causes (SHAP): {meta.get('causes')}\n\n"
        f"SOP excerpts:\n{excerpts}\n\n"
        "Write a short maintenance work order:\n"
        "- First line: '**Diagnosis:**' followed by a one-line diagnosis mentioning the causes.\n"
        "- Then 4-6 numbered steps, safety first.\n"
        "- Use ONLY the SOP excerpts above. After each step cite the excerpt id in brackets, e.g. [SOP-OSF §2].\n"
        "- If the excerpts don't cover something, write 'escalate to supervisor' instead of inventing it.\n"
        "- Do not invent part numbers, torque specs or tool sizes."
    )
    r = requests.post(
        GROQ_URL, timeout=20,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile",
              "temperature": 0.2, "max_tokens": 500,
              "messages": [{"role": "system", "content": "You are a maintenance planner. You write concise, "
                            "safe work orders grounded only in the SOP excerpts you are given."},
                           {"role": "user", "content": prompt}]},
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    if not text:
        raise ValueError("empty LLM response")
    return text


def work_order(meta):
    """meta: machine_id, tech, mode, p, causes (SHAP cause text).
    Returns {"status", "source", "text", "hits": [{"id", "title", "score"}], "reason"}."""
    try:
        mode = meta.get("mode")
        query = f"{SOP_ID.get(mode, '')} {MODE_NAME.get(mode, mode or '')} {meta.get('causes', '')}"
        status, hits = retrieve(query, mode)
        if status == "escalate":
            return {"status": "escalate", "source": "none", "text": ESCALATE, "hits": [], "reason": "low score"}
        brief = [{"id": c["id"], "title": c["title"], "score": c["score"]} for c in hits]
        key = _api_key()
        if key:
            try:
                return {"status": "ok", "source": "llm", "text": _llm(meta, hits, key), "hits": brief, "reason": ""}
            except Exception:
                reason = "llm failed"  # never surface the error (it could echo request details)
        else:
            reason = "no key"
        return {"status": "ok", "source": "template", "text": _template(meta, hits), "hits": brief, "reason": reason}
    except Exception:
        return {"status": "escalate", "source": "none", "text": ESCALATE, "hits": [], "reason": "error"}
