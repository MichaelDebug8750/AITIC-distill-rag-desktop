r"""
webui.py — 知识蒸馏 RAG 管线 · 本地 Web 界面（FastAPI 后端）
================================================================
设计原则：
  1. **不改 main.py**：本文件只 import 并复用其函数（embed / _run_once / is_abstain / ...），
     所有检索、打包、生成、动态预算逻辑与 CLI 完全一致，避免"演示和实测两套代码"。
  2. **结构化溯源**：CLI 的 ask() 把来源 print 出来，Web 需要结构化数据，
     故此处复刻 ask() 的流程但返回 dict（答案 + 来源 + tokens + 是否升配）。
  3. **真流式**：检索完成后逐 token 推送生成结果（SSE），不是"先算完再假装打字"。
     动态升配也会实时告知前端（"正在补充上下文重试…"）。

跑法（在 code 目录下）：
    uvicorn webui:app --host 127.0.0.1 --port 8000
然后浏览器打开 http://127.0.0.1:8000

依赖：
    pip install fastapi uvicorn
"""
import json
import os
import sys
import time
import asyncio

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# 文件上传依赖 python-multipart（可选）。没装也能正常问答，只是不能从网页上传文件。
try:
    from fastapi import UploadFile, File, Form
    import multipart  # noqa: F401  仅用于探测依赖是否存在
    _UPLOAD_OK = True
except Exception:
    _UPLOAD_OK = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M   # noqa: E402  复用 CLI 的全部管线逻辑

app = FastAPI(title="知识蒸馏 RAG · 本地问答", version="1.0")

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "webui_index.html")


def _resolve_db_path():
    """定位向量库（解决 main.DB_PATH 是相对路径、随启动目录漂移的问题）。
       优先级：环境变量 DISTILL_DB > 当前工作目录 ./vectordb > ../data/vectordb > code/vectordb。
       找到后固化为绝对路径写回 M.DB_PATH，Web 与 CLI 从此指向同一个库。"""
    cands = []
    env = os.environ.get("DISTILL_DB")
    if env:
        cands.append(env)
    cands += [
        M.DB_PATH,                                        # ./vectordb（CLI 默认，跟随 cwd）
        os.path.join(HERE, "..", "data", "vectordb"),     # data 目录（建库常在这里跑）
        os.path.join(HERE, "vectordb"),                   # code/vectordb
        os.path.join(HERE, "..", "vectordb"),             # 项目根/vectordb
    ]
    for c in cands:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(M.DB_PATH)                     # 都没有 → 保持默认，前端会提示未建库


M.DB_PATH = _resolve_db_path()                            # 固化绝对路径
print("[webui] 向量库路径 →", M.DB_PATH)
print("[webui] Ollama     →", M._ollama_host())


# ----------------------------- 工具 -----------------------------
def _src_of(meta):
    """把一条 metadata 转成人类可读的来源标签 + 结构化字段（与 main.ask 的 _src 一致）。"""
    t = meta.get("type")
    if t == "audio":
        return {"label": "audio %s" % meta.get("time", "?"), "type": "audio",
                "loc": meta.get("time", "?"), "page": None}
    if t in ("epub", "image"):
        return {"label": meta.get("loc", t), "type": t,
                "loc": meta.get("loc", ""), "page": None}
    return {"label": "p%d" % meta["page"], "type": t or "text",
            "loc": "", "page": meta.get("page")}


def _collection():
    """取集合。注意：main.get_collection() 在没库时会调 sys.exit()，
       抛的是 SystemExit（继承 BaseException，不被 except Exception 捕获），
       故这里显式转成普通异常，避免整个请求 500。"""
    try:
        return M.get_collection()
    except SystemExit as e:
        raise RuntimeError(str(e) or "还没建库")


def _retrieve(question):
    """检索：返回 (docs, metas, dists)，完整复用 main 的扩写与 VL 配额。"""
    qv = M.embed([question])[0]
    return M._retrieve(_collection(), qv, question)


def _sources_from(metas, packed_idx, docs=None):
    """按实际打包的块下标取来源（相关度裁剪下打包的不一定是前几块）。"""
    out, seen = [], set()
    for i in packed_idx:
        if i >= len(metas):
            continue
        s = _src_of(metas[i])
        if s["label"] in seen:
            continue
        seen.add(s["label"])
        if docs is not None and i < len(docs):
            s["snippet"] = docs[i][:220]      # 给前端展开看原文片段
        out.append(s)
    return sorted(out, key=lambda x: x["label"])


# ----------------------------- API -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists(INDEX_HTML):
        return HTMLResponse(open(INDEX_HTML, encoding="utf-8").read())
    return HTMLResponse("<h3>webui_index.html 不存在，请与 webui.py 放在同一目录。</h3>")


@app.get("/api/status")
def status():
    """系统状态：模型、知识块数、是否离线、关键配置 + Ollama 探活诊断。"""
    host = M._ollama_host()
    info = {
        "ollama": "未连接",
        "ollama_host": host,          # 暴露出来，连不上时一眼看出地址对不对
        "chunks": 0,
        "llm_model": M.LLM_MODEL,
        "vl_model": M.VL_MODEL,
        "embed_model": M.EMBED_MODEL,
        "offline": True,
        "relevance_trim": bool(M.RELEVANCE_TRIM),
        "context_budget": M.CONTEXT_BUDGET,
        "budget_escalated": M.BUDGET_ESCALATED,
        "top_k": M.TOP_K,
        "db_path": os.path.abspath(M.DB_PATH),
        "cwd": os.getcwd(),           # 相对路径 DB_PATH 跟随工作目录，暴露出来便于排查
        "ready": False,
        "upload_enabled": _UPLOAD_OK,
    }
    # 探活：直接打 Ollama 根路径
    try:
        import urllib.request
        with urllib.request.urlopen(host, timeout=3) as r:
            r.read(64)
        info["ollama"] = "已连接"
    except Exception as e:
        info["ollama"] = "连不上"
        info["ollama_error"] = "%s → %s" % (host, str(e)[:120])
    # 知识库
    try:
        info["chunks"] = _collection().count()
        info["ready"] = info["chunks"] > 0 and info["ollama"] == "已连接"
    except Exception as e:
        info["db_error"] = str(e)[:160]
    return info


@app.post("/api/ask")
def api_ask(payload: dict):
    """非流式问答：一次返回答案 + 结构化来源。逻辑与 CLI ask() 完全一致。"""
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)

    t0 = time.time()
    docs, metas, dists = _retrieve(question)

    # 第一档：常态预算（保 Token 效率）
    answer, toks, packed_idx = M._run_once(docs, question, M.CONTEXT_BUDGET, metas)
    escalated = False
    # 动态升配：仅当"检索有命中却拒答"时升配重答（与 CLI 同策略）
    if M.should_escalate(answer, docs, dists, M.DYNAMIC_BUDGET):
        ans2, toks2, idx2 = M._run_once(docs, question, M.BUDGET_ESCALATED, metas)
        toks += toks2
        escalated = True
        if not M.is_abstain(ans2):
            answer, packed_idx = ans2, idx2

    return {
        "answer": answer,
        "abstained": M.is_abstain(answer),
        "sources": _sources_from(metas, packed_idx, docs),
        "tokens": toks,
        "escalated": escalated,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "budget": M.BUDGET_ESCALATED if escalated else M.CONTEXT_BUDGET,
        "cite_check": M.verify_citations(answer, packed_idx, metas),
    }


@app.post("/api/brief")
def api_brief(payload: dict):
    """非流式简报生成：围绕 topic 综合检索资料，产出带出处的结构化 brief 文档。
       复用 main.brief 的生成逻辑（_run_once_brief），返回结构与 /api/ask 一致，前端可复用渲染。"""
    topic = (payload.get("topic") or payload.get("question") or "").strip()
    if not topic:
        return JSONResponse({"error": "主题为空"}, status_code=400)

    t0 = time.time()
    docs, metas, _dists = _retrieve(topic)
    answer, toks, packed_idx = M._run_once_brief(docs, topic, M.BUDGET_ESCALATED, metas)

    return {
        "answer": answer,
        "abstained": False,
        "sources": _sources_from(metas, packed_idx, docs),
        "tokens": toks,
        "escalated": False,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "budget": M.BUDGET_ESCALATED,
        "cite_check": M.verify_citations(answer, packed_idx, metas),
        "mode": "brief",
    }


def _sse(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))


@app.get("/api/ask/stream")
async def api_ask_stream(q: str):
    """流式问答（SSE）。真流式：检索完先推来源，再逐 token 推生成结果。
       若首答拒答且检索有命中 → 推 escalate 事件，再流式重答（把动态预算演给用户看）。"""
    question = (q or "").strip()
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)

    async def gen():
        t0 = time.time()
        loop = asyncio.get_event_loop()
        try:
            docs, metas, dists = await loop.run_in_executor(None, _retrieve, question)
        except Exception as e:
            msg = str(e)[:200]
            if "10061" in msg or "refused" in msg.lower() or "urlopen" in msg.lower():
                msg = ("连不上 Ollama（%s）。请确认：① Ollama 已启动（ollama list 能列出模型）；"
                       "② 若设过 OLLAMA_HOST 环境变量，需与 Ollama 实际监听地址一致。"
                       % M._ollama_host())
            yield _sse("error", {"msg": msg})
            return
        yield _sse("retrieved", {"n": len(docs)})

        async def run(budget, tag):
            """按预算打包 → 流式生成。返回 (完整答案, packed_idx, tokens)。"""
            if M.RELEVANCE_TRIM:
                packed, packed_idx = M._pack_relevance(docs, question, budget)
            else:
                packed, packed_idx = M._pack_truncate(docs, budget)
            context = M._labeled_context(packed, packed_idx, metas)   # 每块带真页标签，让模型引用真页码
            prompt = M._format_prompt(context, question, packed_idx, metas)

            yield ("meta", {"budget": budget, "tag": tag,
                            "sources": _sources_from(metas, packed_idx, docs)})

            q_out = asyncio.Queue()

            def produce():
                """在线程里跑 Ollama 流式，把分片塞进队列。

                ⚠️ 必须走 /api/generate（补全模式），与 main.py 的 _generate 一致。
                   走 /api/chat（对话模式）会套用聊天模板，模型天性"乐于助人"，
                   会脱离 Material 用自身知识作答、还硬安一个页码 —— 直接破坏
                   Citation Grounding 的防幻觉保证。这是实测踩过的坑，勿改。
                """
                try:
                    import ollama
                    stream = ollama.generate(
                        model=M.LLM_MODEL,
                        prompt=prompt,                    # prompt 以 "Answer:" 结尾 → 补全模式
                        think=False,
                        stream=True,
                        options={"temperature": M.TEMPERATURE,
                                 "num_predict": M.NUM_PREDICT},
                    )
                    for part in stream:
                        piece = part.get("response", "")
                        if piece:
                            loop.call_soon_threadsafe(q_out.put_nowait, ("d", piece))
                        if part.get("done"):
                            tk = (part.get("prompt_eval_count", 0) or 0) + \
                                 (part.get("eval_count", 0) or 0)
                            loop.call_soon_threadsafe(q_out.put_nowait, ("t", tk))
                except Exception as e:
                    loop.call_soon_threadsafe(q_out.put_nowait, ("e", str(e)[:200]))
                loop.call_soon_threadsafe(q_out.put_nowait, (None, None))

            loop.run_in_executor(None, produce)

            buf, toks = [], 0
            while True:
                kind, val = await q_out.get()
                if kind is None:
                    break
                if kind == "d":
                    buf.append(val)
                    yield ("delta", {"text": val})
                elif kind == "t":
                    toks = val
                elif kind == "e":
                    yield ("error", {"msg": val})
                    break
            full = M._strip_think("".join(buf)).strip()
            yield ("__done__", (full, packed_idx, toks))

        # ---- 第一档 ----
        answer = ""
        packed_idx, toks = [], 0
        async for ev, data in run(M.CONTEXT_BUDGET, "first"):
            if ev == "__done__":
                answer, packed_idx, toks = data
            else:
                yield _sse(ev, data)

        # ---- 动态升配（仅当检索有命中却拒答）----
        escalated = False
        if M.should_escalate(answer, docs, dists, M.DYNAMIC_BUDGET):
            escalated = True
            yield _sse("escalate", {"from": M.CONTEXT_BUDGET, "to": M.BUDGET_ESCALATED,
                                    "reason": "首答拒答且检索有命中，补充上下文重试"})
            a2, idx2, tk2 = "", [], 0
            async for ev, data in run(M.BUDGET_ESCALATED, "escalated"):
                if ev == "__done__":
                    a2, idx2, tk2 = data
                else:
                    yield _sse(ev, data)
            toks += tk2
            if not M.is_abstain(a2):          # 升配后答出来了才采用
                answer, packed_idx = a2, idx2

        yield _sse("done", {
            "answer": answer,
            "abstained": M.is_abstain(answer),
            "sources": _sources_from(metas, packed_idx, docs),
            "tokens": toks,
            "escalated": escalated,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "cite_check": M.verify_citations(answer, packed_idx, metas),
        })

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/build")
async def api_build(payload: dict):
    """建库（按服务器本地路径）。始终可用，不依赖 python-multipart。
       pdf = 替换建库；epub/image/audio = 追加入库（与 CLI 语义一致）。"""
    kind = (payload.get("kind") or "").strip()
    path = (payload.get("path") or "").strip()
    max_pages = int(payload.get("max_pages") or 120)
    vl_limit = int(payload.get("vl_limit") or 15)
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "文件不存在：%s" % path}, status_code=400)
    try:
        loop = asyncio.get_event_loop()
        if kind == "pdf":
            await loop.run_in_executor(None, lambda: M.build(path, max_pages, vl_limit, True))
        elif kind == "epub":
            await loop.run_in_executor(None, lambda: M.build_epub(path, None))
        elif kind == "image":
            await loop.run_in_executor(None, lambda: M.build_image(path))
        elif kind == "audio":
            await loop.run_in_executor(None, lambda: M.build_audio(path, None, None))
        else:
            return JSONResponse({"error": "未知类型：%s（支持 pdf/epub/image/audio）" % kind},
                                status_code=400)
        return {"ok": True, "chunks": _collection().count()}
    except Exception as e:
        return JSONResponse({"error": str(e)[:300]}, status_code=500)


if _UPLOAD_OK:
    @app.post("/api/upload")
    async def api_upload(kind: str = Form(...), max_pages: int = Form(120),
                         vl_limit: int = Form(15), file: UploadFile = File(...)):
        """从网页上传文件建库。需要 python-multipart：
               pip install python-multipart
           未安装时本端点不注册，其余功能不受影响。"""
        try:
            up = os.path.join(HERE, "_uploads")
            os.makedirs(up, exist_ok=True)
            filename = os.path.basename(file.filename or "upload.bin")
            path = os.path.join(up, filename)
            with open(path, "wb") as f:
                f.write(await file.read())
            loop = asyncio.get_event_loop()
            if kind == "pdf":
                await loop.run_in_executor(None, lambda: M.build(path, max_pages, vl_limit, True))
            elif kind == "epub":
                await loop.run_in_executor(None, lambda: M.build_epub(path, None))
            elif kind == "image":
                await loop.run_in_executor(None, lambda: M.build_image(path))
            elif kind == "audio":
                await loop.run_in_executor(None, lambda: M.build_audio(path, None, None))
            else:
                return JSONResponse({"error": "未知类型：%s" % kind}, status_code=400)
            return {"ok": True, "chunks": _collection().count(), "saved": path}
        except Exception as e:
            return JSONResponse({"error": str(e)[:300]}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
