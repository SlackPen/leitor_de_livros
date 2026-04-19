"""
BookDialog v5 - Sistema de Dialogo com Livros Digitais
Backend: FastAPI + pdfplumber + RAG in-memory + OpenAI
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

import pdfplumber
import numpy as np
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (
    HTMLResponse, JSONResponse, FileResponse,
    StreamingResponse, Response
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("bookdialog")

app = FastAPI(title="BookDialog", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Armazenamento in-memory ───────────────────────────────────
STORE: dict = {}       # book_id -> {chunks, embeddings, created_at}
PDF_CACHE: dict = {}   # book_id -> {content: bytes, filename: str}
MAX_PDF_MB = 50


# ── Modelos Pydantic ──────────────────────────────────────────
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


class IngestRequest(BaseModel):
    book_id: str
    text: str
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"


class TestKeyRequest(BaseModel):
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    model: Optional[str] = "gpt-4o-mini"


class FetchPdfRequest(BaseModel):
    url: str
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    book_id: Optional[str] = ""


# ── Utilitarios ───────────────────────────────────────────────

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text).strip()
    chunks = []
    start = 0
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
    if n < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / n)


async def get_embeddings(texts: List[str], api_key: str, base_url: str) -> List[List[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json"
    }
    all_emb = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            payload = {
                "model": "text-embedding-3-small",
                "input": batch
            }
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                raise HTTPException(r.status_code, "Erro embeddings: " + r.text[:200])
            data = r.json()
            all_emb.extend(d["embedding"] for d in data["data"])
    return all_emb


def search_chunks(book_id: str, q_emb, top_k: int = 5, min_score: float = 0.15):
    store = STORE.get(book_id)
    if not store:
        return []
    scored = []
    for i in range(len(store["chunks"])):
        score = cosine_sim(q_emb, store["embeddings"][i])
        scored.append({"chunk": store["chunks"][i], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [r for r in scored[:top_k] if r["score"] >= min_score]


def sse_event(data: dict) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


# ── Rotas ─────────────────────────────────────────────────────

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
    return Response(
        svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"}
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "books": list(STORE.keys()),
        "mode": "python-fastapi-rag-v5"
    }


@app.post("/api/test-key")
async def test_key(body: TestKeyRequest):
    try:
        base = (body.base_url or "https://api.openai.com/v1").rstrip("/")
        headers = {
            "Authorization": "Bearer " + body.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "model": body.model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": "OK"}],
            "max_tokens": 5
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(base + "/chat/completions", headers=headers, json=payload)
        if r.status_code == 200:
            return {"ok": True}
        err_msg = r.json().get("error", {}).get("message", r.text[:200])
        return {"ok": False, "error": err_msg}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Upload de PDF ─────────────────────────────────────────────

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "O arquivo deve ser um PDF.")

    if size_mb > MAX_PDF_MB:
        raise HTTPException(
            413,
            "PDF muito grande: " + str(round(size_mb, 1)) + " MB. Limite: " + str(MAX_PDF_MB) + " MB. "
            "Tente um PDF menor ou comprimido."
        )

    book_id = "book_" + str(int(time.time())) + "_" + os.urandom(3).hex()
    PDF_CACHE[book_id] = {"content": content, "filename": file.filename}
    log.info("PDF recebido: " + str(file.filename) + " (" + str(round(size_mb, 1)) + " MB) -> book_id=" + book_id)
    return {
        "book_id": book_id,
        "filename": file.filename,
        "size_mb": round(size_mb, 2)
    }


# ── Baixar PDF de URL ─────────────────────────────────────────

@app.post("/api/fetch-pdf")
async def fetch_pdf(body: FetchPdfRequest):
    url = body.url
    if not url.startswith("http"):
        raise HTTPException(400, "URL invalida.")
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0 BookDialog/5.0"})
        if r.status_code != 200:
            raise HTTPException(400, "Erro HTTP " + str(r.status_code) + " ao baixar o PDF.")

        content = r.content
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_PDF_MB:
            raise HTTPException(
                413,
                "PDF muito grande: " + str(round(size_mb, 1)) + " MB. Limite: " + str(MAX_PDF_MB) + " MB."
            )

        book_id = body.book_id or ("book_" + str(int(time.time())) + "_" + os.urandom(3).hex())
        filename = url.split("?")[0].split("/")[-1] or "livro.pdf"
        PDF_CACHE[book_id] = {"content": content, "filename": filename}
        log.info("PDF baixado de URL: " + filename + " (" + str(round(size_mb, 1)) + " MB) -> book_id=" + book_id)
        return {"book_id": book_id, "filename": filename, "size_mb": round(size_mb, 2)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, "Erro ao baixar PDF: " + str(e))


# ── SSE: Processamento com progresso em tempo real ────────────

@app.get("/api/process-sse/{book_id}")
async def process_sse(
    book_id: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    book_title: str = ""
):
    if book_id not in PDF_CACHE:
        async def err_gen():
            yield sse_event({"type": "error", "message": "PDF nao encontrado. Faca o upload novamente."})
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def generate():
        cache = PDF_CACHE.pop(book_id, None)
        if not cache:
            yield sse_event({"type": "error", "message": "PDF expirado. Faca o upload novamente."})
            return

        content = cache["content"]

        try:
            # PASSO 1: Extrair texto
            yield sse_event({"type": "progress", "step": 1, "pct": 10, "message": "Extraindo texto do PDF..."})
            await asyncio.sleep(0.05)

            loop = asyncio.get_event_loop()

            def extract_pdf():
                pages_text = []
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    total_pages = len(pdf.pages)
                    for i, page in enumerate(pdf.pages):
                        t = page.extract_text() or ""
                        pages_text.append((i + 1, total_pages, t))
                return pages_text

            pages_data = await loop.run_in_executor(None, extract_pdf)
            num_pages = pages_data[-1][1] if pages_data else 0

            text_parts = []
            for pg_num, total, txt in pages_data:
                text_parts.append("[Pagina " + str(pg_num) + "]\n" + txt)

            full_text = "\n\n".join(text_parts).strip()

            if len(full_text) < 100:
                yield sse_event({
                    "type": "error",
                    "message": "Nao foi possivel extrair texto do PDF. Pode ser um PDF escaneado (so imagens) sem camada de texto."
                })
                return

            chars = len(full_text)
            yield sse_event({
                "type": "progress", "step": 1, "pct": 30,
                "message": "Texto extraido: " + str(num_pages) + " paginas, " + str(chars) + " caracteres"
            })
            await asyncio.sleep(0.05)

            # PASSO 2: Chunking
            yield sse_event({"type": "progress", "step": 2, "pct": 40, "message": "Dividindo texto em trechos..."})
            await asyncio.sleep(0.05)

            text_to_index = full_text[:200000]
            chunks = chunk_text(text_to_index)

            yield sse_event({
                "type": "progress", "step": 2, "pct": 50,
                "message": str(len(chunks)) + " trechos criados"
            })
            await asyncio.sleep(0.05)

            # PASSO 3: Embeddings
            total_batches = max(1, math.ceil(len(chunks) / 50))
            yield sse_event({
                "type": "progress", "step": 3, "pct": 55,
                "message": "Gerando embeddings (" + str(total_batches) + " lote(s))..."
            })

            base = (base_url or "https://api.openai.com/v1").rstrip("/")
            emb_url = base + "/embeddings"
            headers = {
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json"
            }

            all_emb = []
            async with httpx.AsyncClient(timeout=120.0) as client:
                for i in range(0, len(chunks), 50):
                    batch = chunks[i:i + 50]
                    batch_num = i // 50 + 1
                    pct = 55 + int((batch_num / total_batches) * 25)
                    yield sse_event({
                        "type": "progress", "step": 3, "pct": pct,
                        "message": "Embeddings lote " + str(batch_num) + "/" + str(total_batches) + "..."
                    })
                    payload = {
                        "model": "text-embedding-3-small",
                        "input": batch
                    }
                    r = await client.post(emb_url, headers=headers, json=payload)
                    if r.status_code != 200:
                        err_msg = r.json().get("error", {}).get("message", r.text[:200])
                        yield sse_event({"type": "error", "message": "Erro na API de embeddings: " + err_msg})
                        return
                    data = r.json()
                    all_emb.extend(d["embedding"] for d in data["data"])

            # PASSO 4: Indexar
            yield sse_event({"type": "progress", "step": 4, "pct": 85, "message": "Indexando banco vetorial..."})
            await asyncio.sleep(0.05)

            STORE[book_id] = {
                "chunks": chunks,
                "embeddings": all_emb,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

            log.info("RAG indexado: book_id=" + book_id + " | " + str(len(chunks)) + " chunks")

            # Sugestoes de perguntas
            yield sse_event({"type": "progress", "step": 4, "pct": 90, "message": "Gerando sugestoes de perguntas..."})

            suggestions = []
            try:
                q_emb_list = await get_embeddings(
                    ["introducao tema principal personagens assunto"],
                    api_key,
                    base_url or "https://api.openai.com/v1"
                )
                results = search_chunks(book_id, q_emb_list[0], top_k=3)
                sample = "\n\n".join(r["chunk"] for r in results)[:2500]

                title_part = ""
                if book_title:
                    title_part = " \"" + book_title + "\""

                prompt = (
                    "Com base nos trechos do livro" + title_part + ", gere exatamente 6 "
                    "perguntas variadas e instigantes em portugues brasileiro.\n\n"
                    "Retorne APENAS um array JSON com 6 strings, sem mais nada.\n"
                    "Exemplo: [\"Pergunta 1?\",\"Pergunta 2?\",\"Pergunta 3?\","
                    "\"Pergunta 4?\",\"Pergunta 5?\",\"Pergunta 6?\"]\n\n"
                    "Trechos:\n" + sample
                )

                messages_payload = [
                    {"role": "system", "content": "Voce gera perguntas sobre livros em portugues brasileiro."},
                    {"role": "user", "content": prompt}
                ]

                chat_payload = {
                    "model": "gpt-4o-mini",
                    "messages": messages_payload,
                    "max_tokens": 400,
                    "temperature": 0.8
                }

                async with httpx.AsyncClient(timeout=30.0) as client:
                    rs = await client.post(
                        base + "/chat/completions",
                        headers=headers,
                        json=chat_payload
                    )
                    if rs.status_code == 200:
                        raw = rs.json()["choices"][0]["message"]["content"]
                        match = re.search(r'\[.*?\]', raw, re.DOTALL)
                        if match:
                            suggestions = json.loads(match.group())
            except Exception as e:
                log.warning("Sugestoes falhou: " + str(e))

            # Concluido
            yield sse_event({
                "type": "done",
                "book_id": book_id,
                "pages": num_pages,
                "chars": chars,
                "chunks": len(chunks),
                "suggestions": suggestions
            })

        except Exception as e:
            log.error("Erro no processamento SSE: " + str(e))
            yield sse_event({"type": "error", "message": "Erro ao processar: " + str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Chat RAG ──────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(400, "Mensagem vazia.")
    if body.book_id not in STORE:
        raise HTTPException(404, "Livro nao encontrado. Faca o upload novamente.")
    if not body.api_key:
        raise HTTPException(401, "API Key nao configurada.")

    base = (body.base_url or "https://api.openai.com/v1").rstrip("/")

    # Embedding da pergunta
    try:
        q_embs = await get_embeddings([body.message], body.api_key, base)
    except Exception as e:
        raise HTTPException(500, "Erro embedding: " + str(e))

    # Busca RAG
    results = search_chunks(body.book_id, q_embs[0], top_k=5)
    total = len(STORE[body.book_id]["chunks"])
    book_title = body.book_title or "o livro"

    if results:
        ctx_parts = []
        for i, r in enumerate(results):
            relevancia = str(int(r["score"] * 100))
            ctx_parts.append("--- Trecho " + str(i + 1) + " (relevancia " + relevancia + "%) ---\n" + r["chunk"])
        ctx = "\n\n".join(ctx_parts)
    else:
        ctx = "[Nenhum trecho relevante encontrado.]"

    system_content = (
        "Voce e um assistente especializado no livro \"" + book_title + "\".\n\n"
        "REGRAS:\n"
        "1. Responda APENAS com base nos trechos abaixo.\n"
        "2. Se nao houver info suficiente, diga claramente.\n"
        "3. Cite o numero do trecho quando relevante.\n"
        "4. Responda em portugues brasileiro.\n"
        "5. Use Markdown (negrito, listas, citacoes).\n\n"
        "TRECHOS RELEVANTES (" + str(len(results)) + " de " + str(total) + "):\n" + ctx
    )

    messages = [{"role": "system", "content": system_content}]
    for h in (body.history or [])[-6:]:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": body.message})

    headers = {
        "Authorization": "Bearer " + body.api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "model": body.model or "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 1200,
        "temperature": 0.15
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(base + "/chat/completions", headers=headers, json=payload)

        if r.status_code != 200:
            err = r.json().get("error", {}).get("message", r.text[:300])
            raise HTTPException(r.status_code, "Erro OpenAI: " + err)

        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        log.info(
            "Chat: book_id=" + body.book_id +
            " | chunks=" + str(len(results)) + "/" + str(total) +
            " | tokens in=" + str(usage.get("prompt_tokens", 0)) +
            " out=" + str(usage.get("completion_tokens", 0))
        )

        return {
            "reply": reply,
            "rag": {
                "chunks_used": len(results),
                "total_chunks": total,
                "scores": [round(r["score"], 3) for r in results]
            },
            "usage": {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, "Erro ao chamar OpenAI: " + str(e))
