# 智能体包：med-assistant

- 学科：med
- 工具链：检索（RAG + 引用溯源）、计算器（白名单 AST 安全求值）；规则路由
- 一键启动：双击 `run.bat`（先 `ollama create` 专属模型，再进入对话）
- 手动启动对话：
  `python "E:\Ollama_test\code\agent_runtime.py" --db "E:\Ollama_test\data\vectordb" --prompt system_prompt.txt`

对话中：普通问题走检索问答（带 [p.X] / [audio mm:ss] 引用）；输入算式（如 `3*(4+5)`、`sqrt(144)`）走计算器，返回精确结果。
