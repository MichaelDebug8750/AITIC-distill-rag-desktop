# AITIC Desktop 1.0

AITIC Desktop 是 Beta Plus 分支的 Windows 原生版本。界面由 Qt Widgets 绘制，
问答、检索、引用、拒答、建库和评测逻辑直接在程序进程内调用；运行时不启动
FastAPI/Uvicorn、不打开浏览器，也不监听本地端口。

## 普通用户安装

1. 双击 `AITIC-Desktop-1.0.0-Setup-x64.exe`，先选择“简体中文”或 English。
2. 按 Telegram 风格向导选择安装路径、开始菜单文件夹和附加任务。“桌面快捷方式”
   默认勾选，不需要时可取消；也可以选择不创建开始菜单文件夹。
3. 等待向导进度条完成，从桌面或开始菜单打开 **AITIC Desktop**。
4. 首次启动按提示进入“设置 → 模型管理”，点击“一键配置推荐环境”。
   完整安装包已经包含 Ollama Windows 运行时，不需要另装 Python、Ollama 或 Web 服务；
   程序只会联网下载没有打进安装包的模型权重。
5. 在“资料库”中一次选择一本或多本 PDF / EPUB；程序会依次建库，完成后即可在
   “问答”中使用。EPUB 会先在本机转换成可检索的分页文本，再复用 PDF 的引用链路。

模型权重和既有教材库体积很大，不塞入安装包；它们由 Ollama 下载或由用户在本机
建库。断网使用前应先把三个模型和所需教材准备好。
三档模型切换、模型目录迁移和本地 GGUF/Modelfile 导入见
[`MODEL_SETUP_GUIDE.md`](MODEL_SETUP_GUIDE.md)。

## 安装位置与数据安全

- 程序文件：`%LOCALAPPDATA%\Programs\AITIC Desktop`
- 用户知识库、反馈与配置：`%LOCALAPPDATA%\AITIC Desktop`

程序与数据目录相互独立。卸载程序会移除程序和快捷方式，但不会删除用户知识库。
需要彻底清除个人数据时，应先确认无须保留，再手动删除数据目录。

## 界面功能

- 问答：已核验答案流式显示、多资料选择、三档回答方式、页范围、工作区角色说明、
  标准/混合检索、教材外补充、专注模式、本轮统计和引用原文抽屉。
- 本地会话：最近对话搜索、恢复、置顶、重命名、删除、复制和 Markdown 导出。
- 资料库：PDF / EPUB 多选队列导入、进度跟踪、未完成任务恢复、激活切换和 PDF
  图表页识别。每本失败独立记录，不会阻断同批次后续教材。
- 证据审查、分块浏览和知识库诊断：独立检查召回、入库原文与库健康，不调用回答模型。
- 更多工具：带出处简报、自动出题、批量评测、概念对照、检索 A/B 和反馈回归集。
- 运行状态与设置：本机服务、依赖、数据目录、模型检查和模型下载。
- 模型管理：Qwen3 4B/8B/14B 三档下载与切换、推荐模型一键配置、GGUF/Modelfile
  一键导入、模型目录选择和 VC++ 运行库修复。

## 便携版

解压 `AITIC-Desktop-1.0.0-Portable.zip` 后运行目录中的
`AITIC Desktop.exe`。便携版和安装版默认共用上述用户数据目录；两者不要同时运行。

## 从源码运行

```powershell
powershell -ExecutionPolicy Bypass -File .\build_desktop.ps1 -InstallOnly
powershell -ExecutionPolicy Bypass -File .\start_desktop.ps1 -CheckOnly
powershell -ExecutionPolicy Bypass -File .\start_desktop.ps1
```

源码启动时数据根默认是当前项目目录，避免污染已安装版本的数据。也可用
`AITIC_PROJECT_ROOT` 指向其他绝对路径。

## 构建发布物

```powershell
powershell -ExecutionPolicy Bypass -File .\build_desktop.ps1
```

脚本会创建隔离 Python 环境、生成图标、用 PyInstaller 冻结 onedir EXE、下载并
校验固定版本的官方 Ollama x64 基础运行时和 AMD ROCm 附加包、用 Inno Setup 7
生成中英文大众版 Setup EXE、用本地 WiX 4.0.6 构建并校验企业备用 MSI，并生成
便携 ZIP 和 `SHA256SUMS.txt`。若机器没有
.NET SDK，脚本会按微软官方方式把 .NET 8 SDK 安装到 `packaging\.dotnet`，不修改
系统 PATH。

开发时只想快速检查冻结包，可加 `-SkipOllamaRuntime`；面向普通用户发布时不得使用
该选项。

## 发布前检查

- Setup EXE、MSI 与应用 EXE 当前未做 Authenticode 签名，公开分发时 Windows 可能
  显示 SmartScreen 提示；正式发布应使用受信任的代码签名证书签名。
- 本目录的 `packaging\THIRD_PARTY_NOTICES.md` 记录关键依赖许可提示。尤其是
  PyMuPDF 和 Qt/PySide6，分发方式改变时必须重新做许可合规检查。
- 模型许可、教材版权和模型输出责任不由安装包自动解决，发布方仍需单独核对。

## 已执行的桌面验收

- Python 回归：`code/test_pipeline.py` 452 项通过。
- 原生源码功能：12 个功能页、会话持久化、问答、证据、分块、库诊断、简报和测试题通过。
- PDF / EPUB 多选导入：两文件串行队列通过；生成 EPUB 的转换、真实建库、检索和
  原始文件名保留端到端通过。
- 模型一键导入：真实 Modelfile 已完成创建、Ollama show、切换、恢复原模型和测试模型清理。
- 冻结包：默认知识库问答与精确拒答通过，引用格式通过，且未加载 Uvicorn。
- MSI：WiX ICE 校验通过；安装路径选择、可选桌面快捷方式、开始菜单、进度页、
  默认/自定义路径真实安装、已安装 EXE 启动和完整卸载均已在最终完整包上通过。
- Setup EXE：中英文选择、现代向导、路径页、开始菜单、桌面快捷方式、摘要与进度页
  的静态契约通过；真实安装和卸载结果见最新验收记录。
- 数据保留：卸载删除程序与快捷方式，同时保留用户数据目录中的探针文件。
