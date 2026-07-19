# 知识蒸馏管线 · Pipeline 镜像
# 只装 Python 管线，Ollama 走宿主机（路线A）或同一 compose 里的 ollama 服务（路线B）
FROM python:3.11-slim

# apt 换阿里云源（deb.debian.org 从国内下大包会断线），只装运行时真正需要的 libgomp1
#   （ctranslate2 / onnxruntime 运行时依赖它；ffmpeg/编译器都不需要——相关 wheel 自带）
RUN set -eux; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
      [ -f "$f" ] && sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' "$f" || true; \
    done; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends libgomp1; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip 也换阿里云源，避免从 pypi.org 拉 wheel 慢/断
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONUNBUFFERED=1

# 先装依赖（单独一层，改代码不必重装依赖）
COPY code/requirements.txt /app/code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/code/requirements.txt

# 再拷源码
COPY code/ /app/code/

# 默认连宿主机 Ollama；路线B 由 compose 覆盖成 http://ollama:11434
ENV OLLAMA_HOST=http://host.docker.internal:11434

# 工作目录设成挂载进来的 data，vectordb / vl_cache.json 就落在这里、可持久化
WORKDIR /app/data

ENTRYPOINT ["python", "/app/code/main.py"]
CMD ["--help"]
