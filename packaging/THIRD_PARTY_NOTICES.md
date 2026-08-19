# AITIC Desktop 第三方组件提示

本文件是发布复核清单，不替代各项目随包元数据中的完整许可证，也不构成法律意见。
正式分发前应由发布方按实际用途、是否改动和是否公开源码做最终合规审查。

## 随桌面程序分发的关键组件

- **Python**：Python Software Foundation License。
- **PySide6 / Qt for Python 6.10.1**：Qt 官方提供 LGPLv3、GPLv3 和商业许可路径。
  当前 onedir 包保留独立 Qt DLL 动态链接布局；分发方仍须满足所选许可的通知、
  源码/替换能力等要求，或取得商业许可。见 <https://doc.qt.io/qtforpython-6/licenses.html>。
- **PyInstaller 6.21.0**：GPL 许可证及 bootloader 例外；该例外允许分发由
  PyInstaller 生成的可执行文件。见 <https://pyinstaller.org/en/stable/license.html>。
- **PyMuPDF 1.27.2.3**：AGPL 或商业许可。若发布方式不能满足 AGPL，应在分发前
  取得适用的商业许可。见 <https://pymupdf.readthedocs.io/en/latest/about.html#license>。
- **ChromaDB 1.5.9**：Apache License 2.0。
- **FastAPI 0.139.2**：MIT License。桌面版复用其数据模型/响应对象，但不启动服务。
- **Starlette、Uvicorn、Ollama Python、ONNX Runtime、tokenizers**：各自许可证以
  安装包元数据和上游仓库为准。Uvicorn 不在冻结运行时中启动或导入。
- **Ollama Windows 0.32.4**：MIT License；完整安装包包含官方 x64 独立 Windows
  基础运行时及同版本 AMD ROCm 附加运行包，不包含模型权重。见
  <https://github.com/ollama/ollama/blob/v0.32.4/LICENSE>。
- **Microsoft Visual C++ v14 Redistributable x64 14.51.36247.0**：完整包附带由
  Microsoft Corporation 有效签名的官方安装器，仅在用户从模型管理页明确选择修复时
  启动。许可条款随微软安装器显示，官方入口见
  <https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist>。

## 仅构建时使用

- **Inno Setup 7.1.0 x64**：用于生成面向普通用户的中英文单文件 Setup EXE。
  下载的编译器在构建前校验为 Pyrsys B.V. 有效签名。Inno Setup 官方请求符合其定义的
  商业用户购买商业许可证；准备商业生产发布时应按实际主体核对并办理。见
  <https://jrsoftware.org/isorder.php>。
- **WiX Toolset 4.0.6**：仅用于生成 MSI，不随最终程序运行。见
  <https://github.com/wixtoolset/wix/blob/main/LICENSE.TXT>。
- **.NET 8 SDK**：仅放在 `packaging\.dotnet` 供 WiX 构建使用，不装入 MSI。

## 未包含在安装包中的内容

Qwen3/Qwen3-VL/bge-m3 模型权重和教材原文不包含在 Setup EXE、MSI 或便携 ZIP 中。用户下载、
导入或分发模型及教材前，仍须分别核对模型许可证和内容权利。
