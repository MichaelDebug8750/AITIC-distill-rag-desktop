# Docker 一键部署说明

本项目提供 Docker 部署，满足交付物①"Docker/Shell 一键脚本"。

## 放哪
把这些文件全放到项目根目录 `E:\Ollama_test`（和 `code/`、`data/` 同级）：

```
E:\Ollama_test\
├── code\                     # 你的源码（已有）
├── data\                     # PDF/音频/vectordb（已有）
├── models\                   # faster-whisper 权重（已有）
├── Dockerfile                # 新
├── docker-compose.yml        # 新（路线A）
├── docker-compose.gpu.yml    # 新（路线B，可选）
├── .dockerignore             # 新
├── deploy.ps1                # 新（Windows 一键）
└── run.sh                    # 新（Linux/Mac 一键）
```

## 两条路线（选一条）

### 路线A（推荐，默认）— 容器跑管线，连宿主 Ollama
- Ollama 留在 Windows 上，用它已经认到的原生 5090，**不碰 GPU 直通**。
- 容器里只跑 Python 管线，通过 `host.docker.internal:11434` 连宿主 Ollama。
- 优点：不折腾 GPU 直通，起得来是确定的；管线本身容器化了，属于真·Docker 部署，满足"必须用 Docker"的要求。
- 这也是 Ollama + 应用最常见的生产拆分方式（重服务和业务逻辑分离）。

一键：
```powershell
# Windows，项目根目录
powershell -ExecutionPolicy Bypass -File .\deploy.ps1
```
或手动：
```powershell
docker compose build
docker compose run --rm pipeline --help
```

### 路线B（可选）— Ollama 也进容器 + GPU 直通
- 完整隔离，但要 GPU 直通真的能用。先验证：
```powershell
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
- 能列出 5090 才继续；列不出说明 NVIDIA Container Toolkit / WSL GPU 没到位，别硬上，用路线A。
- 启动：
```powershell
docker compose -f docker-compose.gpu.yml up -d
# 首次要在容器里的 ollama 拉模型（除非按注释把宿主 .ollama 挂进去）：
docker compose -f docker-compose.gpu.yml exec ollama ollama pull qwen3:8b
docker compose -f docker-compose.gpu.yml exec ollama ollama pull qwen3-vl:8b
docker compose -f docker-compose.gpu.yml exec ollama ollama pull bge-m3
```

## 日常命令（两条路线通用）
```powershell
# 建库
docker compose run --rm pipeline build --pdf med.pdf
docker compose run --rm pipeline build --audio Starmer.mp3 --max-seconds 300
# 提问
docker compose run --rm pipeline ask "What is a process in an operating system?"
# 生成智能体
docker compose run --rm pipeline agent --pdf med.pdf
```
> 路线B 把上面命令里的 `docker compose` 换成 `docker compose -f docker-compose.gpu.yml`。

## 一个必要的代码改动（连接地址走环境变量）
容器里 `127.0.0.1` 指的是容器自己，不是宿主。所以 `main.py` 里 Ollama 的地址要能读 `OLLAMA_HOST` 环境变量（compose 已经把它设好了）。

在 `main.py` 顶部加：
```python
import os
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
```
然后把 HTTP 兜底里写死的 `http://127.0.0.1:11434` 换成 `OLLAMA_BASE`（就是你之前修 502 时加的 urllib 直连那几处：`embed()` / `_generate()` / `_chat_vl()` 里的 `_post_json`）。

> ollama 官方 python 库本身会自动读 `OLLAMA_HOST`，所以库这条通道不改也能连上；改的是你手写的 HTTP 兜底那条，避免它 fallback 时打到容器自己。

## 排错
- **`host.docker.internal` 连不上**：确认宿主 Ollama 在跑（`http://127.0.0.1:11434/api/tags` 浏览器能打开）。Linux 才需要 compose 里的 `extra_hosts`，Windows/Mac 自带。
- **构建很慢/上下文很大**：检查 `.dockerignore` 是否生效，别把 `vectordb/`、`models/` 带进构建上下文。
- **音频/PDF 找不到**：文件要在 `data/`（容器工作目录就是挂载进来的 `/app/data`），命令里用相对文件名即可（如 `med.pdf`，不用写路径）。
- **路线B 认不到 GPU**：回路线A，不影响任何功能，Ollama 在宿主照样用满 5090。
