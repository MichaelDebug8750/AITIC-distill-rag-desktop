@echo off
chcp 65001 >nul
echo == 1) 创建专属 Ollama 模型（system prompt + 参数）==
ollama create med-assistant -f "E:\Ollama_test\data\agent_med\Modelfile"
echo == 2) 启动智能体对话（工具：检索 + 计算器）==
python "E:\Ollama_test\code\agent_runtime.py" --db "E:\Ollama_test\data\vectordb" --collection knowledge_base --prompt "E:\Ollama_test\data\agent_med\system_prompt.txt"
