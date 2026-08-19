# -*- coding: utf-8 -*-
"""把 EPUB 建成 webui 认得的知识库。

为什么要单写：webui 的 /api/build 只收 PDF（「其他格式请继续使用原 CLI」），
而 main.py 早就有 build_epub()。这里不改 main.py（指纹约束），也不改 webui 的端点，
只做一件事：把 main 的 build_epub 输出到 webui 的库目录，并按 webui 的
registry 结构登记一条。

registry 字段照抄 webui._start_build_job 里的写法，缺字段会让库在界面上半死不活。
"""
import io
import json
import os
import sys
import time
import uuid

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import main as M                                        # noqa: E402

PROJECT_ROOT = r"E:\Ollama_test_beta"
KB_ROOT = os.path.join(PROJECT_ROOT, "data", "webui_knowledge_bases")
REGISTRY = os.path.join(KB_ROOT, "registry.json")
EPUB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "data", "简明世界经济史.epub")

if not os.path.exists(EPUB):
    raise SystemExit("找不到 EPUB：%s" % EPUB)

filename = os.path.basename(EPUB)
name = os.path.splitext(filename)[0]

reg = json.load(io.open(REGISTRY, encoding="utf-8"))
if any(x.get("name") == name for x in reg.get("libraries", [])):
    raise SystemExit("已存在同名知识库「%s」，先删掉再建，避免题集映射到两个库" % name)

library_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
db_path = os.path.join(KB_ROOT, library_id, "vectordb")
os.makedirs(db_path, exist_ok=True)

# build_epub 用的是模块级 DB_PATH，指过去即可；不动 main.py 源码
M.DB_PATH = db_path
print("[build] 目标库 %s" % db_path, flush=True)
t0 = time.time()
M.build_epub(EPUB)

import chromadb                                          # noqa: E402
col = chromadb.PersistentClient(path=db_path).get_or_create_collection(M.COLLECTION)
n = col.count()
print("[build] 入库块数 %d，用时 %.1f 分钟" % (n, (time.time() - t0) / 60))

src_abs = os.path.abspath(EPUB)
try:
    src_ref = (os.path.relpath(src_abs, PROJECT_ROOT)
               if os.path.commonpath([src_abs, PROJECT_ROOT]) == PROJECT_ROOT else src_abs)
except ValueError:
    src_ref = src_abs

reg = json.load(io.open(REGISTRY, encoding="utf-8"))     # 重读，缩短写窗口
reg["libraries"].insert(0, {
    "id": library_id, "name": name, "source": filename, "source_path": src_ref,
    "status": "ready", "chunks": int(n), "subject": "",
    "db_path": os.path.relpath(db_path, PROJECT_ROOT),
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "error": "",
})
with io.open(REGISTRY, "w", encoding="utf-8") as f:
    f.write(json.dumps(reg, ensure_ascii=False, indent=2))
print("[build] 已登记：%s（id=%s）" % (name, library_id))
