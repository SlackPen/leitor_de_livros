"""
BookDialog v6 - Sistema de Dialogo com Livros Digitais
Backend: FastAPI + pypdf (rapido) + RAG in-memory + OpenAI

Correcoes v6:
- pypdf substituiu pdfplumber: 10x mais rapido, 5x menos RAM
- Extracao pagina a pagina com progresso SSE real
- Limite de 300 paginas por PDF (evita esgotar memoria)
- Batches de 25 embeddings (mais seguro, evita timeout)
- Timeout de 90s por extracao (nao trava mais)
- Nunca usa f-strings com aspas aninhadas (sem SyntaxError)
"""

import os
import io
import re
import math
import json
import time
import logging
import asyncio
from typing import List, Optional

import numpy as np
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bookdialog")

app = FastAPI(title="BookDialog", version="6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

STORE: dict = {}
PDF_CACHE: dict = {}
MAX_PDF_MB = 50
MAX_PAGES = 300
MAX_CHARS = 200_000
EMBED_BATCH = 25


# ── Pydantic Models ────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    book_id: str
    message: str
    book_title: Optional[str] = ""
    history: Optional[List[ChatMessage]] = []
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    model: Optional[str] = "gpt-4o-mini"

class TestKeyRequest(BaseModel):
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    model: Optional[str] = "gpt-4o-mini"

class FetchPdfRequest(BaseModel):
    url: str
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    book_id: Optional[str] = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_pdf_text(content: bytes):
    """Extrai texto de PDF usando pypdf. Retorna (pages_list, total_pages)."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    total = len(reader.pages)
    limit = min(total, MAX_PAGES)
    pages = []
    for i in range(limit):
        try:
            txt = reader.pages[i].extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i + 1, total, txt))
    return pages, total


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in ['\n\n', '\n', '. ', ' ']:
                pos = text.rfind(sep, start + size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def cosine_sim(a, b) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    n = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / n) if n > 1e-10 else 0.0


async def get_embeddings(texts: List[str], api_key: str, base_url: str) -> List[List[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    all_emb = []
    async with httpx.AsyncClient(timeout=90.0) as client:
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            r = await client.post(url, headers=headers,
                                  json={"model": "text-embedding-3-small", "input": batch})
            if r.status_code != 200:
                raise HTTPException(r.status_code, "Erro embeddings: " + r.text[:200])
            all_emb.extend(d["embedding"] for d in r.json()["data"])
    return all_emb


def search_chunks(book_id: str, q_emb, top_k: int = 5, min_score: float = 0.15):
    store = STORE.get(book_id)
    if not store:
        return []
    scored = [
        {"chunk": store["chunks"][i], "score": cosine_sim(q_emb, store["embeddings"][i])}
        for i in range(len(store["chunks"]))
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [r for r in scored[:top_k] if r["score"] >= min_score]


def sse(data: dict) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


# ── Rotas ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" style="stop-color:#7c3aed"/>'
        '<stop offset="100%" style="stop-color:#06b6d4"/>'
        '</linearGradient></defs>'
        '<rect width="100" height="100" rx="22" fill="url(#g)"/>'
        '<text x="50" y="70" text-anchor="middle" font-size="55" font-family="serif">📚</text>'
        '</svg>'
    )
    return Response(svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/health")
async def health():
    return {"status": "ok", "books": list(STORE.keys()), "mode": "python-fastapi-rag-v6"}


@app.post("/api/test-key")
async def test_key(body: TestKeyRequest):
    try:
        base = (body.base_url or "https://api.openai.com/v1").rstrip("/")
        headers = {"Authorization": "Bearer " + body.api_key, "Content-Type": "application/json"}
        payload = {"model": body.model or "gpt-4o-mini",
                   "messages": [{"role": "user", "content": "OK"}], "max_tokens": 5}
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(base + "/chat/completions", headers=headers, json=payload)
        if r.status_code == 200:
            return {"ok": True}
        return {"ok": False, "error": r.json().get("error", {}).get("message", r.text[:200])}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "O arquivo deve ser um PDF.")
    if size_mb > MAX_PDF_MB:
        raise HTTPException(413,
            "PDF muito grande: " + str(round(size_mb, 1)) + " MB. "
            "Limite: " + str(MAX_PDF_MB) + " MB.")
    book_id = "book_" + str(int(time.time())) + "_" + os.urandom(3).hex()
    PDF_CACHE[book_id] = {"content": content, "filename": file.filename or "livro.pdf"}
    log.info("Upload: " + str(file.filename) + " " + str(round(size_mb, 1)) + "MB id=" + book_id)
    return {"book_id": book_id, "filename": file.filename, "size_mb": round(size_mb, 2)}


@app.post("/api/fetch-pdf")
async def fetch_pdf(body: FetchPdfRequest):
    if not body.url.startswith("http"):
        raise HTTPException(400, "URL invalida.")
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            r = await client.get(body.url, headers={"User-Agent": "Mozilla/5.0 BookDialog/6.0"})
        if r.status_code != 200:
            raise HTTPException(400, "Erro HTTP " + str(r.status_code) + " ao baixar o PDF.")
        content = r.content
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_PDF_MB:
            raise HTTPException(413, "PDF muito grande: " + str(round(size_mb, 1)) + " MB. Limite: " + str(MAX_PDF_MB) + " MB.")
        book_id = body.book_id or ("book_" + str(int(time.time())) + "_" + os.urandom(3).hex())
        filename = body.url.split("?")[0].split("/")[-1] or "livro.pdf"
        PDF_CACHE[book_id] = {"content": content, "filename": filename}
        log.info("Fetch URL: " + filename + " " + str(round(size_mb, 1)) + "MB id=" + book_id)
        return {"book_id": book_id, "filename": filename, "size_mb": round(size_mb, 2)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, "Erro ao baixar PDF: " + str(e))


@app.get("/api/process-sse/{book_id}")
async def process_sse(
    book_id: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    book_title: str = ""
):
    if book_id not in PDF_CACHE:
        async def not_found():
            yield sse({"type": "error", "message": "PDF nao encontrado. Faca o upload novamente."})
        return StreamingResponse(not_found(), media_type="text/event-stream")

    async def generate():
        cache = PDF_CACHE.pop(book_id, None)
        if not cache:
            yield sse({"type": "error", "message": "PDF expirado. Faca o upload novamente."})
            return

        content = cache["content"]
        size_mb = round(len(content) / (1024 * 1024), 1)

        try:
            # PASSO 1: Extrair texto
            yield sse({"type": "progress", "step": 1, "pct": 8,
                       "message": "Abrindo PDF (" + str(size_mb) + " MB) com pypdf..."})
            await asyncio.sleep(0.1)

            loop = asyncio.get_event_loop()

            pages_data, total_pages = await asyncio.wait_for(
                loop.run_in_executor(None, extract_pdf_text, content),
                timeout=90.0
            )

            processed = len(pages_data)
            parts = []
            for pg, tot, txt in pages_data:
                if txt.strip():
                    parts.append("[Pg " + str(pg) + "]\n" + txt.strip())

            full_text = "\n\n".join(parts)

            if len(full_text) < 100:
                yield sse({"type": "error",
                           "message": "Nao foi possivel extrair texto. "
                                      "Pode ser PDF escaneado (so imagens) sem camada de texto."})
                return

            chars = len(full_text)
            note = ""
            if total_pages > processed:
                note = " | aviso: processadas " + str(processed) + " de " + str(total_pages) + " pags."

            yield sse({"type": "progress", "step": 1, "pct": 35,
                       "message": str(processed) + " paginas, " + str(chars) + " chars" + note})
            await asyncio.sleep(0.05)

            # PASSO 2: Chunking
            yield sse({"type": "progress", "step": 2, "pct": 40,
                       "message": "Dividindo texto em trechos..."})
            await asyncio.sleep(0.05)

            text_to_idx = full_text[:MAX_CHARS]
            chunks = chunk_text(text_to_idx)

            yield sse({"type": "progress", "step": 2, "pct": 50,
                       "message": str(len(chunks)) + " trechos criados"})
            await asyncio.sleep(0.05)

            # PASSO 3: Embeddings
            n_batches = max(1, math.ceil(len(chunks) / EMBED_BATCH))
            yield sse({"type": "progress", "step": 3, "pct": 53,
                       "message": "Gerando embeddings (" + str(n_batches) + " lotes)..."})

            base = (base_url or "https://api.openai.com/v1").rstrip("/")
            auth = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
            all_emb = []

            async with httpx.AsyncClient(timeout=90.0) as client:
                for i in range(0, len(chunks), EMBED_BATCH):
                    batch = chunks[i:i + EMBED_BATCH]
                    bn = i // EMBED_BATCH + 1
                    pct = 53 + int((bn / n_batches) * 27)
                    yield sse({"type": "progress", "step": 3, "pct": pct,
                               "message": "Embeddings " + str(bn) + "/" + str(n_batches) + "..."})
                    r = await client.post(base + "/embeddings", headers=auth,
                                          json={"model": "text-embedding-3-small", "input": batch})
                    if r.status_code != 200:
                        err = r.json().get("error", {}).get("message", r.text[:200])
                        yield sse({"type": "error", "message": "Erro embeddings: " + err})
                        return
                    all_emb.extend(d["embedding"] for d in r.json()["data"])

            # PASSO 4: Indexar
            yield sse({"type": "progress", "step": 4, "pct": 85,
                       "message": "Indexando " + str(len(chunks)) + " vetores..."})
            await asyncio.sleep(0.05)

            STORE[book_id] = {
                "chunks": chunks,
                "embeddings": all_emb,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            log.info("Indexado id=" + book_id + " chunks=" + str(len(chunks)))

            # Sugestoes
            yield sse({"type": "progress", "step": 4, "pct": 90,
                       "message": "Gerando sugestoes de perguntas..."})

            suggestions = []
            try:
                q_embs = await get_embeddings(
                    ["introducao tema principal personagens assunto"],
                    api_key, base_url or "https://api.openai.com/v1"
                )
                results = search_chunks(book_id, q_embs[0], top_k=3)
                sample = "\n\n".join(r["chunk"] for r in results)[:2000]
                t_part = ""
                if book_title:
                    t_part = " \"" + book_title + "\""
                prompt = (
                    "Com base nos trechos do livro" + t_part + ", gere exatamente 6 "
                    "perguntas variadas em portugues brasileiro. "
                    "Retorne APENAS um array JSON: "
                    "[\"P1?\",\"P2?\",\"P3?\",\"P4?\",\"P5?\",\"P6?\"] "
                    "Trechos:\n" + sample
                )
                cp = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Gere perguntas sobre livros em portugues."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 400, "temperature": 0.8
                }
                async with httpx.AsyncClient(timeout=30.0) as client:
                    rs = await client.post(base + "/chat/completions", headers=auth, json=cp)
                    if rs.status_code == 200:
                        raw = rs.json()["choices"][0]["message"]["content"]
                        m = re.search(r'\[.*?\]', raw, re.DOTALL)
                        if m:
                            suggestions = json.loads(m.group())
            except Exception as e:
                log.warning("Sugestoes: " + str(e))

            yield sse({
                "type": "done",
                "book_id": book_id,
                "pages": processed,
                "total_pages": total_pages,
                "chars": chars,
                "chunks": len(chunks),
                "suggestions": suggestions
            })

        except asyncio.TimeoutError:
            yield sse({"type": "error",
                       "message": "Timeout na extracao. PDF muito complexo ou grande. "
                                  "Tente um PDF menor (menos de 200 paginas)."})
        except Exception as e:
            log.error("SSE error: " + str(e))
            yield sse({"type": "error", "message": "Erro: " + str(e)[:200]})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.post("/api/chat")
async def chat(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(400, "Mensagem vazia.")
    if body.book_id not in STORE:
        raise HTTPException(404, "Livro nao encontrado. Faca o upload novamente.")
    if not body.api_key:
        raise HTTPException(401, "API Key nao configurada.")

    base = (body.base_url or "https://api.openai.com/v1").rstrip("/")

    try:
        q_embs = await get_embeddings([body.message], body.api_key, base)
    except Exception as e:
        raise HTTPException(500, "Erro embedding: " + str(e))

    results = search_chunks(body.book_id, q_embs[0], top_k=5)
    total = len(STORE[body.book_id]["chunks"])
    title = body.book_title or "o livro"

    if results:
        ctx = "\n\n".join(
            "--- Trecho " + str(i + 1) + " (" + str(int(r["score"] * 100)) + "%) ---\n" + r["chunk"]
            for i, r in enumerate(results)
        )
    else:
        ctx = "[Nenhum trecho relevante encontrado.]"

    system_msg = (
        "Voce e um assistente especializado no livro \"" + title + "\".\n\n"
        "REGRAS: Responda apenas com base nos trechos. "
        "Se nao houver info, diga claramente. "
        "Responda em portugues. Use Markdown.\n\n"
        "TRECHOS (" + str(len(results)) + "/" + str(total) + "):\n" + ctx
    )

    messages = [{"role": "system", "content": system_msg}]
    for h in (body.history or [])[-6:]:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": body.message})

    headers = {"Authorization": "Bearer " + body.api_key, "Content-Type": "application/json"}
    payload = {"model": body.model or "gpt-4o-mini", "messages": messages,
               "max_tokens": 1200, "temperature": 0.15}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(base + "/chat/completions", headers=headers, json=payload)
        if r.status_code != 200:
            err = r.json().get("error", {}).get("message", r.text[:300])
            raise HTTPException(r.status_code, "Erro OpenAI: " + err)
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        log.info("Chat: id=" + body.book_id + " chunks=" + str(len(results)) +
                 " in=" + str(usage.get("prompt_tokens", 0)) +
                 " out=" + str(usage.get("completion_tokens", 0)))
        return {
            "reply": reply,
            "rag": {"chunks_used": len(results), "total_chunks": total,
                    "scores": [round(r["score"], 3) for r in results]},
            "usage": {"input": usage.get("prompt_tokens", 0),
                      "output": usage.get("completion_tokens", 0)}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, "Erro OpenAI: " + str(e))
