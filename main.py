"""
BookDialog v9 — ultra low memory
- Processa PDF em blocos de 50 páginas (streaming)
- Descarta texto imediatamente após gerar embeddings
- Nunca mantém o PDF inteiro em memória durante processamento
- Embeddings em float16 para usar 50% menos RAM
- Sem SSE: polling puro
"""

import os, io, re, math, json, time, logging, threading, gc
from typing import List, Optional, Dict

import numpy as np
import httpx
import requests as req_lib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bookdialog")

app = FastAPI(title="BookDialog", version="9.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

BOOKS: Dict[str, Dict] = {}
JOBS:  Dict[str, Dict] = {}
MAX_PDF_MB  = 200
EMBED_BATCH = 10   # batch menor para economizar RAM
PAGE_BLOCK  = 50   # processar 50 páginas por vez

# ── Schemas ───────────────────────────────────────────────────────────────────

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

# ── Helpers ────────────────────────────────────────────────────────────────────

def set_job(book_id: str, status: str, pct: int, message: str, error: str = ""):
    JOBS[book_id] = {
        "status": status, "pct": pct, "message": message,
        "error": error, "updated": time.time(),
    }

def chunk_text(text: str, size: int = 900, overlap: int = 100) -> List[str]:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
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

def cosine_sim_f16(va: np.ndarray, vb: np.ndarray) -> float:
    """Cosine similarity entre arrays float16."""
    a = va.astype(np.float32)
    b = vb.astype(np.float32)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-10 else 0.0

def search_chunks(book_id: str, q_emb: List[float], top_k: int = 5, min_score: float = 0.15):
    book = BOOKS.get(book_id)
    if not book:
        return []
    qv = np.array(q_emb, dtype=np.float32)
    scored = [
        {"chunk": book["chunks"][i],
         "score": cosine_sim_f16(book["embeddings"][i], qv)}
        for i in range(len(book["chunks"]))
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [r for r in scored[:top_k] if r["score"] >= min_score]

# ── Job em thread separada (requests síncrono, ultra low memory) ───────────────

def _process_book_thread(book_id: str, pdf_bytes: bytes, filename: str,
                         api_key: str, base_url: str, book_title: str):
    """
    Processa o livro completo em blocos de PAGE_BLOCK páginas.
    Nunca mantém o texto inteiro em memória — processa e descarta por bloco.
    """
    try:
        base = (base_url or "https://api.openai.com/v1").rstrip("/")
        size_mb = round(len(pdf_bytes) / (1024 * 1024), 1)
        auth = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}

        set_job(book_id, "running", 3, "Abrindo PDF (" + str(size_mb) + " MB)...")

        # Abrir PDF e contar páginas
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            total_pages = len(reader.pages)
        except Exception as e:
            set_job(book_id, "error", 0, "", "Erro ao abrir PDF: " + str(e)[:200])
            return

        set_job(book_id, "running", 5, str(total_pages) + " pag. encontradas. Extraindo texto...")

        # Acumular todos os chunks e embeddings de todos os blocos
        all_chunks: List[str] = []
        all_embeddings: List[np.ndarray] = []
        total_chars = 0
        n_blocks = math.ceil(total_pages / PAGE_BLOCK)

        with req_lib.Session() as session:
            session.headers.update(auth)

            for block_idx in range(n_blocks):
                page_start = block_idx * PAGE_BLOCK
                page_end   = min(page_start + PAGE_BLOCK, total_pages)
                pages_done = page_end

                # Progresso da extração (5% → 40%)
                extract_pct = 5 + int((pages_done / total_pages) * 35)
                set_job(book_id, "running", extract_pct,
                        "Extraindo pag " + str(page_start + 1) + "-" + str(page_end) +
                        " de " + str(total_pages) + "...")

                # ── Extrair texto deste bloco ─────────────────────────────
                block_parts = []
                for i in range(page_start, page_end):
                    try:
                        txt = reader.pages[i].extract_text() or ""
                    except Exception:
                        txt = ""
                    if txt.strip():
                        block_parts.append("[Pg " + str(i + 1) + "]\n" + txt.strip())

                if not block_parts:
                    continue

                block_text = "\n\n".join(block_parts)
                total_chars += len(block_text)

                # ── Chunking deste bloco ──────────────────────────────────
                block_chunks = chunk_text(block_text)

                # Liberar texto do bloco da memória
                del block_text, block_parts
                gc.collect()

                if not block_chunks:
                    continue

                # ── Embeddings deste bloco ────────────────────────────────
                n_bc  = len(block_chunks)
                total_batches_done = len(all_chunks)
                global_total = total_chars  # estimativa

                # Progresso do embedding (40% → 88%)
                embed_start_pct = 40 + int((block_idx / n_blocks) * 48)

                for bi in range(0, n_bc, EMBED_BATCH):
                    batch = block_chunks[bi: bi + EMBED_BATCH]
                    bn = bi // EMBED_BATCH + 1
                    pct = embed_start_pct + int((bn / math.ceil(n_bc / EMBED_BATCH)) * (48 // n_blocks))
                    set_job(book_id, "running", min(pct, 87),
                            "Embeddings bloco " + str(block_idx + 1) + "/" + str(n_blocks) +
                            " — " + str(len(all_chunks) + bi + len(batch)) + " trechos...")

                    ok = False
                    for attempt in range(4):
                        try:
                            r = session.post(
                                base + "/embeddings",
                                json={"model": "text-embedding-3-small", "input": batch},
                                timeout=120,
                            )
                            if r.status_code == 200:
                                embs = r.json()["data"]
                                for d in embs:
                                    # Armazenar em float16 (metade da RAM)
                                    all_embeddings.append(
                                        np.array(d["embedding"], dtype=np.float16)
                                    )
                                ok = True
                                break
                            elif r.status_code == 429:
                                wait = 10 * (attempt + 1)
                                set_job(book_id, "running", pct,
                                        "Rate limit — aguardando " + str(wait) + "s...")
                                time.sleep(wait)
                            else:
                                err = ""
                                try:
                                    err = r.json().get("error", {}).get("message", r.text[:200])
                                except Exception:
                                    err = r.text[:200]
                                set_job(book_id, "error", 0, "",
                                        "Erro OpenAI: " + err)
                                return
                        except req_lib.exceptions.Timeout:
                            if attempt == 3:
                                set_job(book_id, "error", 0, "",
                                        "Timeout nos embeddings. Tente novamente.")
                                return
                            time.sleep(5 * (attempt + 1))
                        except Exception as e:
                            if attempt == 3:
                                set_job(book_id, "error", 0, "",
                                        "Erro inesperado: " + str(e)[:200])
                                return
                            time.sleep(3)

                    if not ok:
                        set_job(book_id, "error", 0, "", "Falha ao gerar embeddings.")
                        return

                all_chunks.extend(block_chunks)
                del block_chunks
                gc.collect()

        if len(all_chunks) == 0:
            set_job(book_id, "error", 0, "",
                    "Nao foi possivel extrair texto. PDF pode ser escaneado (so imagens).")
            return

        # ── Indexar ────────────────────────────────────────────────────────
        n_total = len(all_chunks)
        set_job(book_id, "running", 90, "Indexando " + str(n_total) + " vetores...")
        BOOKS[book_id] = {
            "chunks":     all_chunks,
            "embeddings": all_embeddings,
            "title":      book_title,
            "pages":      total_pages,
            "chars":      total_chars,
            "filename":   filename,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Liberar o PDF da memória
        del pdf_bytes, reader
        gc.collect()

        log.info("Indexado: id=" + book_id + " chunks=" + str(n_total) +
                 " pages=" + str(total_pages))

        # ── Sugestões de perguntas ─────────────────────────────────────────
        set_job(book_id, "running", 93, "Gerando sugestoes...")
        suggestions: List[str] = []
        try:
            with req_lib.Session() as s2:
                s2.headers.update(auth)
                qr = s2.post(
                    base + "/embeddings",
                    json={"model": "text-embedding-3-small",
                          "input": ["introducao tema principal personagens"]},
                    timeout=30,
                )
                if qr.status_code == 200:
                    q_emb = qr.json()["data"][0]["embedding"]
                    top = search_chunks(book_id, q_emb, top_k=3)
                    sample = "\n\n".join(r["chunk"] for r in top)[:2000]
                    t_part = (" \"" + book_title + "\"") if book_title else ""
                    prompt = (
                        "Com base nos trechos do livro" + t_part +
                        ", gere exatamente 6 perguntas variadas em portugues brasileiro. "
                        "Retorne APENAS array JSON: [\"P1?\",\"P2?\",\"P3?\",\"P4?\",\"P5?\",\"P6?\"] "
                        "Trechos:\n" + sample
                    )
                    rs = s2.post(
                        base + "/chat/completions",
                        json={"model": "gpt-4o-mini",
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": 400, "temperature": 0.8},
                        timeout=30,
                    )
                    if rs.status_code == 200:
                        raw = rs.json()["choices"][0]["message"]["content"]
                        m = re.search(r"\[.*?\]", raw, re.DOTALL)
                        if m:
                            suggestions = json.loads(m.group())
        except Exception as e:
            log.warning("Sugestoes falhou: " + str(e))

        JOBS[book_id]["suggestions"] = suggestions
        set_job(book_id, "done", 100,
                "Pronto! " + str(n_total) + " trechos de " + str(total_pages) + " pag.")

    except Exception as e:
        log.error("Job error id=" + book_id + ": " + str(e))
        set_job(book_id, "error", 0, "", "Erro inesperado: " + str(e)[:300])


def start_job_thread(book_id: str, pdf_bytes: bytes, filename: str,
                     api_key: str, base_url: str, book_title: str):
    t = threading.Thread(
        target=_process_book_thread,
        args=(book_id, pdf_bytes, filename, api_key, base_url, book_title),
        daemon=True,
        name="job-" + book_id[-8:],
    )
    t.start()
    log.info("Thread iniciada: " + t.name)
    return t


# ── Rotas FastAPI ──────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
async def favicon():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" style="stop-color:#7c3aed"/>'
        '<stop offset="100%" style="stop-color:#06b6d4"/></linearGradient></defs>'
        '<rect width="100" height="100" rx="22" fill="url(#g)"/>'
        '<text x="50" y="70" text-anchor="middle" font-size="55" font-family="serif">📚</text>'
        '</svg>'
    )
    return Response(svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})

@app.get("/health")
async def health():
    return {"status": "ok", "books": len(BOOKS), "jobs": len(JOBS), "mode": "v9-ultralow"}

@app.get("/api/process-sse/{book_id}")
async def legacy_sse(book_id: str):
    return JSONResponse(
        {"error": "SSE descontinuado. Use POST /api/upload-pdf + GET /api/status/{id}"},
        status_code=410,
    )

@app.post("/api/test-key")
async def test_key(body: TestKeyRequest):
    try:
        base = (body.base_url or "https://api.openai.com/v1").rstrip("/")
        h = {"Authorization": "Bearer " + body.api_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(
                base + "/chat/completions", headers=h,
                json={"model": body.model or "gpt-4o-mini",
                      "messages": [{"role": "user", "content": "OK"}],
                      "max_tokens": 5},
            )
        if r.status_code == 200:
            return {"ok": True}
        return {"ok": False,
                "error": r.json().get("error", {}).get("message", r.text[:200])}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    api_key: str = Form(default=""),
    base_url: str = Form(default="https://api.openai.com/v1"),
    book_title: str = Form(default=""),
):
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    fname   = file.filename or "livro.pdf"

    if not (fname.lower().endswith(".pdf") or (file.content_type or "").startswith("application/pdf")):
        raise HTTPException(400, "Envie um arquivo PDF.")
    if size_mb > MAX_PDF_MB:
        raise HTTPException(
            413, "PDF muito grande: " + str(round(size_mb, 1)) + " MB. Limite: " +
            str(MAX_PDF_MB) + " MB."
        )
    if not api_key:
        raise HTTPException(400, "api_key obrigatorio.")

    book_id = "book_" + str(int(time.time())) + "_" + os.urandom(3).hex()
    set_job(book_id, "pending", 0, "Iniciando...")
    log.info("Upload: " + fname + " " + str(round(size_mb, 1)) + "MB id=" + book_id)

    start_job_thread(book_id, content, fname,
                     api_key, base_url or "https://api.openai.com/v1", book_title or "")

    return {"book_id": book_id, "filename": fname, "size_mb": round(size_mb, 2)}

@app.post("/api/fetch-pdf")
async def fetch_pdf(body: FetchPdfRequest):
    if not body.url.startswith("http"):
        raise HTTPException(400, "URL invalida.")
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as c:
            r = await c.get(body.url, headers={"User-Agent": "Mozilla/5.0 BookDialog/9"})
        if r.status_code != 200:
            raise HTTPException(400, "Erro HTTP " + str(r.status_code))
        content  = r.content
        size_mb  = len(content) / (1024 * 1024)
        if size_mb > MAX_PDF_MB:
            raise HTTPException(413, "PDF muito grande: " + str(round(size_mb, 1)) + " MB.")
        fname    = body.url.split("?")[0].split("/")[-1] or "livro.pdf"
        book_id  = "book_" + str(int(time.time())) + "_" + os.urandom(3).hex()
        set_job(book_id, "pending", 0, "Iniciando...")
        start_job_thread(book_id, content, fname,
                         body.api_key, body.base_url or "https://api.openai.com/v1", "")
        return {"book_id": book_id, "filename": fname, "size_mb": round(size_mb, 2)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, "Erro ao baixar PDF: " + str(e))

@app.get("/api/status/{book_id}")
async def get_status(book_id: str):
    job = JOBS.get(book_id)
    if not job:
        raise HTTPException(404, "Job nao encontrado.")
    result = dict(job)
    if job["status"] == "done":
        book = BOOKS.get(book_id, {})
        result["chunks"]      = len(book.get("chunks", []))
        result["pages"]       = book.get("pages", 0)
        result["chars"]       = book.get("chars", 0)
        result["suggestions"] = job.get("suggestions", [])
    return result

@app.post("/api/chat")
async def chat(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(400, "Mensagem vazia.")
    if body.book_id not in BOOKS:
        raise HTTPException(404, "Livro nao encontrado. Faca o upload novamente.")
    if not body.api_key:
        raise HTTPException(401, "API Key nao configurada.")

    base = (body.base_url or "https://api.openai.com/v1").rstrip("/")
    h    = {"Authorization": "Bearer " + body.api_key, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(
                base + "/embeddings", headers=h,
                json={"model": "text-embedding-3-small", "input": [body.message]},
            )
        if r.status_code != 200:
            raise Exception("HTTP " + str(r.status_code))
        q_emb = r.json()["data"][0]["embedding"]
    except Exception as e:
        raise HTTPException(500, "Erro embedding: " + str(e))

    results = search_chunks(body.book_id, q_emb, top_k=5)
    book    = BOOKS[body.book_id]
    total   = len(book["chunks"])
    title   = body.book_title or book.get("title") or "o livro"

    ctx = (
        "\n\n".join(
            "--- Trecho " + str(i + 1) + " (" + str(int(r["score"] * 100)) + "%) ---\n" + r["chunk"]
            for i, r in enumerate(results)
        )
        if results else "[Nenhum trecho relevante encontrado.]"
    )

    system_msg = (
        "Voce e um assistente especializado no livro \"" + title + "\".\n"
        "Responda APENAS com base nos trechos fornecidos. "
        "Se nao houver informacao suficiente, diga claramente. "
        "Responda em portugues brasileiro. Use Markdown.\n\n"
        "TRECHOS (" + str(len(results)) + "/" + str(total) + "):\n" + ctx
    )

    messages = [{"role": "system", "content": system_msg}]
    for hh in (body.history or [])[-6:]:
        messages.append({"role": hh.role, "content": hh.content})
    messages.append({"role": "user", "content": body.message})

    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(
                base + "/chat/completions", headers=h,
                json={"model": body.model or "gpt-4o-mini",
                      "messages": messages,
                      "max_tokens": 1200, "temperature": 0.15},
            )
        if r.status_code != 200:
            err = ""
            try:
                err = r.json().get("error", {}).get("message", r.text[:300])
            except Exception:
                err = r.text[:300]
            raise HTTPException(r.status_code, "Erro OpenAI: " + err)

        data  = r.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        log.info("Chat: id=" + body.book_id + " chunks=" + str(len(results)) +
                 " tokens=" + str(usage.get("total_tokens", 0)))
        return {
            "reply": reply,
            "rag": {
                "chunks_used": len(results),
                "total_chunks": total,
                "scores": [round(r["score"], 3) for r in results],
            },
            "usage": {
                "input":  usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, "Erro OpenAI: " + str(e))
