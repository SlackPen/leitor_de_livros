"""
BookDialog – Sistema de Diálogo com Livros Digitais
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
import hashlib
from typing import List, Optional

import pdfplumber
import numpy as np
import httpx

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bookdialog")

app = FastAPI(title="BookDialog", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Store in-memory ───────────────────────────────────────────
# book_id -> { chunks, embeddings, title, pages, chars, created_at }
STORE: dict = {}

# ─── Modelos Pydantic ──────────────────────────────────────────
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

class SuggestionsRequest(BaseModel):
    book_id: str
    book_title: Optional[str] = ""
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    model: Optional[str] = "gpt-4o-mini"

class FetchPdfRequest(BaseModel):
    url: str

# ─── Utilitários ───────────────────────────────────────────────

def chunk_text(text: str, size: int = 900, overlap: int = 180) -> List[str]:
    """Divide o texto em chunks sobrepostos, respeitando parágrafos."""
    # Normaliza quebras de linha
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text).strip()

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)

        # Tenta quebrar em parágrafo ou frase
        if end < length:
            for sep in ['\n\n', '\n', '. ', ' ']:
                pos = text.rfind(sep, start + size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)

        start = end - overlap
        if start >= length:
            break

    return chunks


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity entre dois vetores."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / norm)


async def get_embeddings(texts: List[str], api_key: str, base_url: str) -> List[List[float]]:
    """Gera embeddings via OpenAI API (em lotes de 50)."""
    base_url = base_url.rstrip("/")
    url = f"{base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_embeddings = []
    batch_size = 50

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            log.info(f"Embeddings lote {i//batch_size + 1}/{math.ceil(len(texts)/batch_size)} ({len(batch)} chunks)")

            resp = await client.post(url, headers=headers, json={
                "model": "text-embedding-3-small",
                "input": batch
            })

            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code,
                                    detail=f"Erro na API de embeddings: {resp.text[:300]}")

            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(embeddings)

    return all_embeddings


def build_sug_prompt(title: str, sample: str) -> str:
    title_part = f' "{title}"' if title else ""
    return (
        f"Com base nos trechos do livro{title_part}, gere exatamente 6 perguntas variadas.\n\n"
        'Retorne APENAS um array JSON:\n'
        '["Pergunta 1?","Pergunta 2?","Pergunta 3?","Pergunta 4?","Pergunta 5?","Pergunta 6?"]\n\n'
        f"Trechos:\n{sample}"
    )


def search_chunks(book_id: str, query_embedding: List[float],
                  top_k: int = 5, min_score: float = 0.15):
    """Busca os chunks mais relevantes por cosine similarity."""
    store = STORE.get(book_id)
    if not store or not store["chunks"]:
        return []

    scored = [
        {
            "chunk": store["chunks"][i],
            "score": cosine_similarity(query_embedding, store["embeddings"][i]),
            "index": i,
        }
        for i in range(len(store["chunks"]))
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [r for r in scored[:top_k] if r["score"] >= min_score]


# ─── Rotas ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" style="stop-color:#7c3aed"/>
    <stop offset="100%" style="stop-color:#06b6d4"/>
  </linearGradient></defs>
  <rect width="100" height="100" rx="22" fill="url(#g)"/>
  <text x="50" y="70" text-anchor="middle" font-size="55" font-family="serif">📚</text>
</svg>'''
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "books": list(STORE.keys()),
        "mode": "python-fastapi-rag"
    }


@app.post("/api/test-key")
async def test_key(body: TestKeyRequest):
    """Testa se a API Key é válida."""
    try:
        base_url = (body.base_url or "https://api.openai.com/v1").rstrip("/")
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {body.api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": body.model or "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "OK"}],
                    "max_tokens": 5
                }
            )
        if resp.status_code == 200:
            return {"ok": True}
        else:
            detail = resp.json().get("error", {}).get("message", resp.text[:200])
            return {"ok": False, "error": detail}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Recebe o PDF, extrai o texto com pdfplumber (Python puro).
    Retorna o texto extraído + metadados.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Arquivo deve ser um PDF.")

    try:
        content = await file.read()
        log.info(f"PDF recebido: {file.filename} ({len(content):,} bytes)")

        text_pages = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            num_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_pages.append(f"[Página {i+1}]\n{page_text}")

        full_text = "\n\n".join(text_pages).strip()

        if len(full_text) < 50:
            raise HTTPException(422, "Não foi possível extrair texto. O PDF pode ser baseado em imagens (escaneado).")

        log.info(f"Texto extraído: {len(full_text):,} chars | {num_pages} páginas")

        return {
            "success": True,
            "text": full_text,
            "pages": num_pages,
            "chars": len(full_text),
            "filename": file.filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Erro ao processar PDF: {e}")
        raise HTTPException(500, f"Erro ao processar PDF: {str(e)}")


@app.post("/api/fetch-pdf")
async def fetch_pdf(body: FetchPdfRequest):
    """Proxy para baixar PDF de URL (contorna CORS no browser)."""
    url = body.url
    if not url.startswith("http"):
        raise HTTPException(400, "URL inválida.")
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 BookDialog/2.0",
                "Accept": "application/pdf,*/*"
            })
        if resp.status_code != 200:
            raise HTTPException(400, f"Erro HTTP {resp.status_code} ao baixar o PDF.")

        content = resp.content
        # Extrai texto
        text_pages = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            num_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_pages.append(f"[Página {i+1}]\n{page_text}")

        full_text = "\n\n".join(text_pages).strip()
        if len(full_text) < 50:
            raise HTTPException(422, "PDF sem texto extraível (pode ser escaneado).")

        return {
            "success": True,
            "text": full_text,
            "pages": num_pages,
            "chars": len(full_text),
            "filename": url.split("/")[-1] or "livro.pdf",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erro ao baixar PDF: {str(e)}")


@app.post("/api/ingest")
async def ingest(body: IngestRequest):
    """
    Divide o texto em chunks, gera embeddings e indexa in-memory.
    NUNCA armazena o texto completo para reenvio à OpenAI.
    """
    if not body.book_id or not body.text or not body.api_key:
        raise HTTPException(400, "book_id, text e api_key são obrigatórios.")

    # Remove livro anterior
    STORE.pop(body.book_id, None)

    # Divide em chunks
    chunks = chunk_text(body.text, size=900, overlap=180)
    log.info(f"[INGEST] book_id={body.book_id} | {len(chunks)} chunks")

    if not chunks:
        raise HTTPException(422, "Texto vazio após processamento.")

    # Gera embeddings
    try:
        embeddings = await get_embeddings(chunks, body.api_key, body.base_url or "https://api.openai.com/v1")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar embeddings: {str(e)}")

    # Armazena
    STORE[body.book_id] = {
        "chunks": chunks,
        "embeddings": embeddings,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    log.info(f"[INGEST] ✅ {len(chunks)} chunks indexados para book_id={body.book_id}")
    return {
        "success": True,
        "book_id": body.book_id,
        "chunks": len(chunks),
        "message": f"{len(chunks)} trechos indexados no banco vetorial.",
    }


@app.post("/api/chat")
async def chat(body: ChatRequest):
    """
    RAG Chat:
    1. Gera embedding da pergunta
    2. Busca os chunks mais relevantes
    3. Monta prompt enxuto (só os chunks relevantes)
    4. Chama OpenAI — NUNCA o livro inteiro
    """
    if not body.message.strip():
        raise HTTPException(400, "Mensagem vazia.")
    if not body.book_id:
        raise HTTPException(400, "book_id não informado.")
    if body.book_id not in STORE:
        raise HTTPException(404, "Livro não encontrado. Faça o upload novamente.")
    if not body.api_key:
        raise HTTPException(401, "API Key não configurada.")

    base_url = (body.base_url or "https://api.openai.com/v1").rstrip("/")

    # ── 1. Embedding da pergunta ──────────────────────────────
    try:
        query_embs = await get_embeddings([body.message], body.api_key, base_url)
        query_embedding = query_embs[0]
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar embedding: {str(e)}")

    # ── 2. Busca RAG ──────────────────────────────────────────
    results = search_chunks(body.book_id, query_embedding, top_k=5)
    total_chunks = len(STORE[body.book_id]["chunks"])

    # ── 3. Monta contexto ENXUTO ─────────────────────────────
    if not results:
        context_block = "[Nenhum trecho relevante encontrado para esta pergunta.]"
    else:
        parts = []
        for i, r in enumerate(results):
            pct = int(r["score"] * 100)
            parts.append(f"--- Trecho {i+1} (relevância {pct}%) ---\n{r['chunk']}")
        context_block = "\n\n".join(parts)

    book_title = body.book_title or "o livro"
    system_prompt = f"""Você é um assistente especializado em responder perguntas sobre o livro "{book_title}".

REGRAS OBRIGATÓRIAS:
1. Responda EXCLUSIVAMENTE com base nos trechos abaixo.
2. Se os trechos não tiverem informação suficiente, diga: "Os trechos recuperados não contêm informação suficiente para responder isso."
3. Cite o número do trecho quando relevante.
4. Responda sempre em português brasileiro.
5. Use Markdown para formatar (negrito, listas, citações).
6. Seja preciso e baseado apenas nos trechos fornecidos.

TRECHOS RELEVANTES ({len(results)} de {total_chunks} total):
{context_block}"""

    # ── 4. Chama OpenAI com contexto mínimo ──────────────────
    messages = [{"role": "system", "content": system_prompt}]
    for h in (body.history or [])[-6:]:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": body.message})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {body.api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": body.model or "gpt-4o-mini",
                    "messages": messages,
                    "max_tokens": 1200,
                    "temperature": 0.15,
                }
            )

        if resp.status_code != 200:
            err = resp.json().get("error", {}).get("message", resp.text[:300])
            raise HTTPException(resp.status_code, f"Erro da OpenAI: {err}")

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        log.info(f"[CHAT] book_id={body.book_id} | chunks={len(results)}/{total_chunks} | "
                 f"tokens entrada={usage.get('prompt_tokens',0)} saída={usage.get('completion_tokens',0)}")

        return {
            "reply": reply,
            "rag": {
                "chunks_used": len(results),
                "total_chunks": total_chunks,
                "scores": [round(r["score"], 3) for r in results],
            },
            "usage": {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erro ao chamar OpenAI: {str(e)}")


@app.post("/api/suggestions")
async def suggestions(body: SuggestionsRequest):
    """Gera sugestões de perguntas usando RAG."""
    if body.book_id not in STORE:
        return {"suggestions": []}

    base_url = (body.base_url or "https://api.openai.com/v1").rstrip("/")

    try:
        # Busca trechos introdutórios
        query_embs = await get_embeddings(
            ["introdução tema principal personagens assunto do livro"],
            body.api_key, base_url
        )
        results = search_chunks(body.book_id, query_embs[0], top_k=4)
        sample = "\n\n".join(r["chunk"] for r in results)[:3000]

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {body.api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": body.model or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Você gera perguntas instigantes sobre livros em português brasileiro."},
                        {"role": "user", "content": build_sug_prompt(body.book_title, sample)}
                    ],
                    "max_tokens": 400,
                    "temperature": 0.8,
                }
            )

        if resp.status_code != 200:
            return {"suggestions": []}

        raw = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        sugs = json.loads(match.group()) if match else []
        return {"suggestions": sugs}

    except Exception as e:
        log.error(f"[SUGGESTIONS] {e}")
        return {"suggestions": []}


@app.get("/api/stats/{book_id}")
async def stats(book_id: str):
    store = STORE.get(book_id)
    if not store:
        raise HTTPException(404, "Livro não encontrado.")
    total_chars = sum(len(c) for c in store["chunks"])
    return {
        "book_id": book_id,
        "chunks": len(store["chunks"]),
        "total_chars": total_chars,
        "avg_chunk_len": total_chars // len(store["chunks"]),
        "created_at": store["created_at"],
    }
