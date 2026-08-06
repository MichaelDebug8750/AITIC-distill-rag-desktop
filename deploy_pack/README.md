# 部署包 · 说明

任务书交付物②：**预编译 GGUF 量化权重 + 自定义 Modelfile + 启动脚本**。

## 里面有啥

| 文件 | 是什么 |
|---|---|
| `qwen3-8b-Q4_K_M.gguf` | 预编译的 Q4_K_M 量化权重（5,225,374,496 字节，约 4.87 GiB） |
| `Modelfile` | 自定义模型配置，引用上面的 GGUF + 引用锚定 system prompt |
| `deploy.ps1` | Windows 一键脚本：从 GGUF 创建可运行模型 |
| `run.sh` | Linux/Mac 一键脚本，同上 |

## 怎么用（一键）

前提：装好 Ollama。然后在本目录下——

**Windows：**
```powershell
powershell -ExecutionPolicy Bypass -File .\deploy.ps1
```

**Linux/Mac：**
```bash
bash run.sh
```

脚本会用本地这个 GGUF 权重创建出模型 `distill-assistant`。创建完直接对话：
```
ollama run distill-assistant "What is a process in an operating system?"
```

## 说明

- 这个 GGUF 是 Q4_K_M 量化；建议至少准备 6GB 可用显存，纯 CPU 运行建议至少 8GB 可用系统内存。
- Modelfile 里固化了引用锚定规则（只依据材料答、标来源、没依据就 `[NO REFERENCE FOUND]`）。
- 这个包是**自包含**的——不依赖原机器上已有的 Ollama 模型，换台装了 Ollama 的机器也能一键起。
- 完整的知识蒸馏管线（建库/问答/智能体生成）见源码仓库；本包只负责「模型权重 + 一键起模型」这一交付项。
