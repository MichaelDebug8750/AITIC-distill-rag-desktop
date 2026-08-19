# Codex 工作断点（2026-08-18 晚间）

## 暂停状态

- 已按用户要求暂停；本断点后不再启动模型评测、代码优化或安装包构建。
- 英文无固定 seed 全量已经正常退出：1007 / 1007，退出码 0，用时 152.8 分钟。
- 当前没有需要续接的 `desktop_full_eval.py` 会话；恢复时不要重跑这次有效全量。
- Ollama 常驻服务未停止；它不是本轮创建的临时任务。
- 恢复口令：用户说“继续”后，从“恢复后的下一步”开始。

## 项目边界

- 当前开发目录：`E:\Ollama_test_beta_plus`
- 稳定目录 `E:\Ollama_test`、Beta 目录 `E:\Ollama_test_beta` 均不得修改。
- 不修改任何 `main.py`，不提交。
- Plus / Beta / stable 共六份 `code\main.py`、`data\main.py` 当前 SHA-256 均为：
  `FE179DB3D558A65D6452767FED3F19ED5671696D0446D2FF4B54E13D85BF52C4`

## 本轮已经完成的代码修复

### 1. 固定 seed 从默认行为改为显式选择

本地记录已经证明：固定 seed 并没有提高当前 Ollama / GPU 路径的重复稳定性，反而使 8 个敏感问题中仅 2 个保持字节级一致；无 seed 时 8 / 8 一致。因此已修复 `code\webui.py`：

- `DISTILL_MODEL_SEED` 默认空值，产品默认不再注入 seed。
- 只有用户明确设置 `DISTILL_MODEL_SEED=<整数>` 时才向模型 options 注入 seed。
- 非法 seed 值安全回落为不注入。
- `/api/status` 增加 `model_seed`，使有效配置可审计。
- 所有模型 options 统一经过 `_llm_options(...)`。

### 2. 测试和评测工具同步

- `code\test_pipeline.py` 增加守卫：默认不带 seed，显式 seed 能正确生效。
- `packaging\desktop_full_eval.py` 的 manifest 指纹增加 `model_seed`。
- `packaging\desktop_seed_pilot.py` 增加 `--unseeded`，会清除环境中的 seed。
- 修复 `desktop_seed_pilot.py` 的评分元数据 bug：案例类型和 `keywords` 必须取当前标准题集，旧运行文件只提供旧 outcome。旧 Web rows 不含 `keywords`，此前会把一批可评分命中误记为“未判定”。
- 最新完整单元测试：`470 passed, 1 warning`。warning 仅为 Chroma 对未来 Python 3.16 的弃用提示。

## 已验证的桌面 / Web 功能等价性

静态契约均通过：

- Web 业务路由 23 项全部映射到原生后端。
- 原生 UI 导航、资料库、证据审查、分块浏览、诊断、设置、学习工具、批量评测、A/B 对比、概念对照、反馈与回归集、导出/复制对话均有对应入口。
- 丰富结果载荷覆盖：答案、引用、Agent 升档、补充检索、引用校验、检索证据卡、逐句语义审查、证据链关系、不确定项、可信度、Agent trace、诊断、被丢弃资料库、复制/收藏/重新生成。
- PDF / EPUB 多选批量导入、会话持久化、模型管理、高 DPI 和原生字体栅格化契约均通过。

注意：当前公开安装包仍是 2026-08-17 19:09 的旧构建，早于最新源码和本轮修复。用户之前截图中的灰框、旧结果卡和 Agent 展示缺失不能代表最新源码。必须在全量指标达标后重建，再做 frozen / installed smoke。

## 当前有效全量结果

### 中文当前构建（v4，已达到并超过 Web 命中）

- 标签：`desktop_anchor_directness_v4_cn_20260818`
- 题数：104 / 104
- 正确拒答：40
- 编造：0
- 命中：60
- 过度拒答：4
- 未命中：0
- HTTP 错误：0
- 引用失败：0
- Web 参考命中为 59；当前为 60，已通过。

### 英文固定 seed 42 当前构建（v4，仅作对照，不应成为产品默认）

- 标签：`desktop_anchor_directness_v4_en_20260818`
- 报告：`docs\全量跑分_20260812\desktop_anchor_directness_v4_en_20260818_analysis.json`
- 题数：1007 / 1007
- 正确拒答：282
- 编造：18
- 命中：537
- 过度拒答：82
- 未命中：88
- HTTP 错误：0
- 引用失败：0
- 达到旧项目汇报命中下限 534，但低于 Web 当前档 548。
- 由于固定 seed 的重复稳定性实验更差，该结果只能作为对照，不能以恢复默认 seed 42 的方式进入产品。

### 英文无固定 seed 当前构建（v5，刚完成）

- 标签：`desktop_unseeded_anchor_directness_v5_en_20260818`
- rows：`docs\全量跑分_20260812\desktop_unseeded_anchor_directness_v5_en_20260818_rows.jsonl`
- manifest：`docs\全量跑分_20260812\desktop_unseeded_anchor_directness_v5_en_20260818_manifest.json`
- analysis：`docs\全量跑分_20260812\desktop_unseeded_anchor_directness_v5_en_20260818_analysis.json`
- 题数：1007 / 1007
- 正确拒答：280
- 编造：20
- 命中：524
- 过度拒答：99
- 未命中：84
- HTTP 错误：0
- 引用失败：0
- rows SHA-256：`6caa2315e3fff6e9beb3b145418304671286c2c6171e12e6f82e6ed4bc4ddf3e`
- 完整性全部通过：题数、唯一键、manifest、rows 哈希、代码指纹、知识库、服务、桌面适配器均稳定。
- 安全指标优于 Web 当前档：280 >= 277，20 <= 23。
- 命中未达标：524 < 项目汇报档 534，且 524 < Web 当前档 548。
- 因此当前源码不能进入最终打包，下一步必须先做配对失败分析和有依据的命中修复。

## 已排除 / 禁止重试

- 不再通过搜索随机 seed 追指标；这会把噪声当优化，而且固定 seed 已被重复稳定性实验否定。
- 不恢复“英文通用定义锚点”候选。此前全量/定向实验证明它会引入副作用，已经完整回退。
- 不放宽 `_EVIDENCE_FLOOR = 0.99`、引用校验或拒答安全闸门。
- 混合检索默认保持关闭；此前收益只覆盖英文且会引入答非所问式编造。
- 不用跨时段单次差值宣称显著改善；必要时采用配对分类和同条件复核。

## 恢复后的下一步

### A. 先做三方配对分析，不跑模型

对以下三份英文 rows 以 `(book, question)` 为键配对：

1. Web 当前档：`deep_current5_en_20260817_rows.jsonl`（命中 548、正确拒答 277、编造 23）。
2. 固定 seed v4：`desktop_anchor_directness_v4_en_20260818_rows.jsonl`（命中 537、正确拒答 282、编造 18）。
3. 无 seed v5：`desktop_unseeded_anchor_directness_v5_en_20260818_rows.jsonl`（命中 524、正确拒答 280、编造 20）。

输出至少包括：

- 按书、题型、outcome 的汇总。
- Web 命中 -> v5 过度拒答 / 未命中的新增损失。
- v4 命中 -> v5 失败，以及 v5 命中 -> v4 失败的双向翻转。
- 对新增过度拒答检查：top-k 是否已有直接证据、是否模型直接拒答、是否逐句裁剪、是否引用孤立、是否重组裁剪。
- 对新增未命中检查：答案是否实际正确但评分关键词不完整，排除评分器假阴性。

### B. 只实现可推广且不削弱安全性的修法

- 优先找“证据已召回但生成/裁剪杀掉答案”的确定性共因。
- 每个候选先跑新增损失定向集 + 300 道不可答安全集抽样/守卫。
- 候选无明确净改善立即回退。
- 禁止靠放宽闸门换命中。

### C. 达标后再收尾

1. 英文 1007 题重新全量，要求至少达到项目汇报档，目标达到或超过 Web 当前档，同时安全指标不得恶化。
2. 若改动触及中文路径，重跑中文 104 题并保持 40 拒答、0 编造、命中 >= 60。
3. 运行 470 单测、UI contract、Web/桌面路由等价、PDF/EPUB、多选、模型配置、会话持久化、Windows 高 DPI 富界面 smoke。
4. 用最新源码重建完整公共包；不得使用 `-SkipOllamaRuntime`。
5. 对 Setup EXE、MSI、portable zip 做 frozen / installed smoke，核对可选桌面快捷方式、安装进度、多语言、安装路径和一键模型部署。
6. 重新截图，与 Web 富结果逐项对照清晰度和 Agent 呈现。
7. 最后再次核对六份 `main.py` 哈希和稳定/Beta 目录未变化。

## 恢复原则

- 当前英文 v5 安全性通过但命中未达标，不能打包为最终版。
- 中文 v4 已达标，不要为英文优化破坏中文零编造和 60 命中。
- 每个候选必须先做定向验证，失败候选立即回退。
- 不提交，除非用户明确授权。

## 2026-08-18 临时暂停增量断点

### 已完成的新分析

- 已完成 Web current / fixed-seed v4 / no-seed v5 三方按 `(book, question)` 配对。
- Web -> no-seed v5 的净命中为 `-24`：Web 命中丢失 51 条（34 过度拒答、17 未命中），同时从 Web 失败中新增命中 27 条。
- fixed-seed v4 -> no-seed v5 的净命中为 `-13`。大量题在三次运行间双向翻转，不能把单条差异直接当作确定性代码回归。
- 英文 no-seed v5 中，答案中文字符占比 >= 35% 的共有 22 条：14 命中、8 未命中。故英文题混入中文是真缺陷，但不能用粗暴的语言拒答闸门，否则会误杀 14 条现有命中。
- 8 条高中文占比失败全部位于 `fuzzy_desc` / `fuzzy_kw`；典型形态是答案用中文解释正确概念却漏掉评测要求的英文术语，或检索/辨识到了相邻但错误的概念。

### 已修复的新真 Bug

- 非流式 `/api/ask` 在 `support_audit` 创建前就调用 `_enforce_final_directness(...)`，普通非流式请求会抛 `UnboundLocalError`；流式全量没有覆盖到这条路径。
- 已把非流式调用顺序改成与流式一致：先 `_finalize_agent_answer(...)` 得到真实 `support_audit`，再执行最终正面性校验。
- 已新增 `test_non_stream_ask_finalizes_before_enforcing_directness` 守卫，定向复测通过。

### 当前候选（默认关闭）

- `code\webui.py` 新增环境开关 `AITIC_TERM_DIRECTNESS`，默认 `0`，所以当前生产默认行为未改变。
- 开启后仅对纯英文问题增加两条提示约束：整段必须使用英文；若问题是描述后识别概念，首句必须点出材料中明确出现的英文原术语，材料没有命名则精确拒答。
- 不改检索、不放宽 `_EVIDENCE_FLOOR`、不绕过引用/逐句核验。
- 新增两条单测，锁定英文约束和中文问题不受影响；与非流式守卫合跑结果：`7 passed, 466 deselected, 1 warning`。

### 被用户暂停的运行

- 正在跑 40 条 no-seed v5 当前未命中题的默认臂时收到“先暂停”。
- 已向评测进程发送 Ctrl+C 并终止；脚本尚未到最终写报告阶段，`term_directness_pilot_miss_baseline_20260818.json` 不存在，不得把这次中断当成有效结果。
- 下次恢复应从这 40 条默认臂重新开始，再紧接开启 `AITIC_TERM_DIRECTNESS=1` 的同题候选臂；之后补至少 20 条现有命中和 20 条库外拒答对照。
- 若候选没有明确净命中或任一库外安全恶化，立即回退该候选；只有三类小样本都通过才跑英文 1007 全量。
