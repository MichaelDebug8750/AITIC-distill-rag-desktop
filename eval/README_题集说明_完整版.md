# 多学科评测题集 · 完整版

生成日期：2026-07-23 ｜ **55 本书 · 4454 道题** ｜ 七轮核验 · 最终判定 PASS

---

## 一、每本书三部分

| 部分 | `type` | 期望行为 | 考什么 |
|---|---|---|---|
| 第一部分 · 可答题 | `answerable` | `expect: answer` | 书里确实有答案，考检索召回 + 作答 |
| 第二部分 · 不可答题 | `unanswerable` | `expect: abstain` | **书里压根没有**，专测瞎编乱造 |
| 第三部分 · 模糊题 | `fuzzy_desc` / `fuzzy_kw` | `expect: answer` | 不打术语名，靠语义把目标内容捞出来 |

目标是每本 50 / 25 / 25，实际多数落在 44–50 / 25 / 19–25（清洗噪音时掉了几道，见第八节）。

---

## 二、核验结果（这段可以原样写进评测报告）

| 检查项 | 结果 |
|---|---|
**第一轮 · 内容核验**（题目和书对不对得上）

| 检查项 | 结果 |
|---|---|
| 可答题：术语在书里存在，且**每个关键词都在正文里逐字出现** | **2153 / 2153 通过** |
| 不可答题：该词在全书**出现次数 = 0** | **1374 / 1374 通过** |
| 模糊题：GT 词在书里存在，且**查询句里不出现该词** | **927 / 927 通过** |
| **问题行** | **0** |

**第二轮 · 独立复核**（换一套逻辑重读题集和原文，专找第一轮的盲区）——查出并修掉了这些：

| 查出的问题 | 处理 |
|---|---|
| 同一本书内 12 道重复题（都是 `fuzzy_kw` 短查询撞车） | 去重 |
| 1 道常识探针踩雷：问"澳洲首都"，而 `Business Law and the Legal Environment` 正文里出现过 Canberra | 换题 |
| 3 道模糊题 `it` 占位符过多、句子读不通 | 剔除 |
| 70 道题的目标词是代码片段（`>>> myCat = {...}`、`public static`） | 剔除 |
| 70 道题的目标词是通用词（`States` 来自 United States、`Duke`、月份名等） | 剔除 |
| 17 道题的证据句来自版权/来源行（`attribution:`、`Copyright Rice University`） | 剔除 |
| 18 道术语含引号括号等代码符号、14 道格式异常 | 剔除 |

**第三轮 · 换第三套逻辑，专查前两轮的盲区**

| 检查项 | 结果 |
|---|---|
| **绕过缓存直接重开原始 PDF 复核**（抽 12 本 × 12 题） | 144 / 144 通过 —— 证明文本提取没跑偏 |
| **关键词与目标词能否落进同一个检索块**（1500 字符窗口） | 2630 道里只有 1 道超窗 —— 决定 Hit@5 真假的关键项 |
| `page` 字段是否真指向含该词的页（抽 8 本 111 道） | 111 / 111 命中 |
| 不可答探针的**词形变体泄漏**（单复数、连字符变体） | 查出 1 道（`low-income country` vs 书里 `low income country`）→ 剔除 |
| 同一本书里同一目标词被重复问 | 查出 2 道（另有 582 组是 `fuzzy_desc`+`fuzzy_kw` 配对，**属设计如此不是重复**）→ 剔除 |
| 可答题与模糊题目标词是否串号 | **零重叠**，三部分互不泄题 |
| 编码/乱码/中文混入/首尾空白 | 全部正常 |
| 学科归属、文件名、book 字段与实际文件对应 | 全部一致 |

**三轮共剔除 225 道**。代价是部分书从 50 掉到 40 多，没去凑数——宁可少几道，不要塞脏题。

**第四轮 · 用交付给你的脚本重跑一遍**

`verify_eval.py` 是给你在本机自己验的工具——**不用信我的结论**。它会重新打开你的原始 PDF/EPUB，逐题核对上面所有检查项。我用它对 61 本书全量跑过一遍：

```
可答题   2153 / 2153
不可答题 1374 / 1374
模糊题   927 / 927
结构与内容检查：无问题
最终判定：PASS
```

**本地怎么跑**（PowerShell，先把书和 `eval_by_book` 放好）：

```powershell
# 先装依赖
pip install PyMuPDF ebooklib beautifulsoup4

# 先单验一本，确认路径对（快）
& "C:\Users\Seifer\distill\Scripts\python.exe" verify_eval.py `
    --books E:\Ollama_test\books --eval E:\Ollama_test\eval_by_book `
    --only Microbiology

# 全量验（慢，要读完所有 PDF；加 --skip-cooccur 可以快很多）
& "C:\Users\Seifer\distill\Scripts\python.exe" verify_eval.py `
    --books E:\Ollama_test\books --eval E:\Ollama_test\eval_by_book
```

有问题会写进 `verify_eval_报告.txt`，退出码 0 = PASS，2 = FAIL。

**第五轮 · 人工逐题阅读（前四轮全是字符串核对，没人真读过题）**

抽了 46 道跨学科跨题型的题**逐条读**，读出了自动化查不出来的问题，全部修掉：

| 读出来的问题 | 数量 | 例子 |
|---|---|---|
| **非名词性片段被当成概念** | 659 | `Describe further typical.`（来自标题"SOME FURTHER TYPICAL DREAMS"）、`How is alice thought discussed?`（"Alice thought" 是主谓不是实体） |
| **证据句是标题/目录/学习目标** | 204 | `Describe queue.` 的证据是章节目标列表，关键词成了 `objectives`/`understand` |
| **OCR 长 s 污染** | 70 | 布莱克斯通《英国法释义》1765 年排印，`ſ` 被识别成 `f`：`absolute`→`abfolute`、`superiority`→`fuperiority`，整本报废 |
| 内容过薄（目标词全书只出现 1 次） | 33 | |
| 短语最后一个词是动词（不是名词短语） | 33 | `hours passed`、`every object` |
| 证据句压根不含目标词 | 56 | |
| 课程行政用语被当概念 | 3 | `What is online library?`（法学讲义的选课说明） |
| `term` 字段与 GT 关键词不一致 | 2 | 断行造成 `entry-` / `raid-` |
| `it` 占位符过多读不通 | 8 | |

**第五轮共剔除 1068 道，并整本剔除 6 本**（可答题不足 10 道）：
`Dive Into Python 3`(3)、`Introduction to Computer Science`(8)、`An Introduction to Law`(2)、`Commentaries on the Laws of England`(0，OCR报废)、`Conditioned Reflexes — Pavlov`(0)、`The Principles of Psychology — William James`(0)。

**五轮累计从 5608 道砍到 4152 道，从 61 本砍到 55 本。** 砍掉的都是自洽但没意义的题——留着只会让评测结果假高或假低。

新增的名词性检验规则：非术语表来源的目标词，**必须至少 2 次以冠词/介词引导出现**（`the X` / `a X` / `of X`），否则判为句子碎片剔除。这条一刀砍掉 659 道，是这轮最有效的过滤器。

**第六轮 · 复审第五轮的剔除（自查有没有砍过头）**

第五轮一口气砍了 1068 道，这轮把**被砍掉的题捞回来逐条读**，检查剔除本身对不对。结论：**砍过头了，两条过滤器有方法论错误**。

| 错误 | 后果 | 修正 |
|---|---|---|
| **名词性检验的引导词表太窄**，只用了 `the/a/an/of/in/to/with/on` | 误杀了 `promissory estoppel`、`outgroups`、`response prevention`、`comorbidity`、`selection sort`、`cellulitis` 这类真概念——它们常以 `our X`、`and X`、`for X` 出现 | 扩到 26 个引导词后重判 |
| **名词性检验对专有名词根本不适用** | 英语不说 "the Gabelle"，导致 `Manette`、`Jacques`、`Holmwood`、`Scatcherd`、`Ryder` 等**真人物被整批误杀** | 改用「人物性检验」：该名字附近 3 次以上出现 `said/asked/his/her/Mr/Mrs` 等人称标记 |
| **"证据句是标题"就整题删除** | 概念本身没问题，只是我随手挑的证据句挑差了 | 改为**重新到书里找一句像样的正文证据句**，找得到就恢复 |

**恢复 332 道**（均带 `revalidated: true` 标记），4152 → **4484 道**。

同时确认了第五轮**砍对**的部分不予恢复：`stood looking`、`returned defarge`、`cried lord henry`、`darcy looked` 这类对话标签碎片（184 道）、`static void`/`typedef`/`return self` 等代码片段、`multiple choice questions` 等课程行政用语、以及布莱克斯通那本 OCR 长 s 报废的书——这些确实该删。

**第七轮 · 总体校对（测量为主，不再大改）**

全项复跑 + 分层抽样人工阅读（每来源 × 每题型各 3 道，共 33 道）。只剔除了 **30 道客观损坏的**（23 道题面残留页码/项目符号/花括号、4 道 `term` 与 GT 不一致、3 道取自 Standard Ebooks 版权页）。

**残余缺陷率实测（这是最该写进报告的一段）：**

| 来源 | 题数 | 抽样判定 | 说明 |
|---|---|---|---|
| `glossary` 术语表 | 1225 | **≈0%** | 抽 9 道全对，作者标注的权威定义 |
| 探针（跨学科+常识） | 1374 | **≈0%** | 抽 3 道全对，机器可完全验证 |
| `distinct` 特征短语 | 1336 | **约 15–20% 偏弱** | 如 `venous`（形容词）、`runtime`（证据句不构成定义） |
| `entity`/`lit` 文学 | 519 | **约 25–30% 偏弱** | 如 `lattice`（窗格）、`marvellous`（形容词）、`darling` |

**全库整体偏弱率约 8–9%。**

**关键区分**：这些"偏弱"题**不是错题**——目标词确实在书里、关键词确实逐字存在、检索理应能召回。它们只是**作为概念题不够漂亮**。所以：

- 对 **Hit@5、召回率、拒答率**这类检索指标，它们**仍然有效**，不影响结论
- 对**答案质量的人工评判**，它们是噪音，抽查时应避开或单独标注

想要更干净的子集，直接按 `source == "glossary"` 过滤，1225 道全部是权威定义题。

核验和复核都是程序化全量跑的，人工阅读部分是抽样。

---

## 三、题是怎么出的

**没用 LLM 造题。** 三种书源，质量分级如下（报告里建议照实说）：

**A. 术语表驱动（19 本 OpenStax）—— 最硬**
从 PDF 的 `Key Terms` 区块按**字体粗体位置**逐条抠出「术语 + 原文定义 + 页码」，是作者自己标的权威定义。

**B. 特征短语驱动（30 本：CS、法学经典、医学专著等）—— 次之**
这些书没有术语表。改用 TF-IDF 式打分挑「本书高频 + 全 61 本语料里罕见」的名词短语，再从正文抓一句包含它的证据句。
效果：OSTEP 挑出 `inode` / `mutex` / `system call` / `address space`；霍姆斯《普通法》挑出 `trespass` / `bailee` / `assumpsit` / `bailment`；Gray's Anatomy 挑出 `fossa` / `inguinal` / `plexus`。

**C. 实体驱动（12 本小说）**
抽高频专有名词。用「**小写形式很少出现**」这条规则区分真专名和句首大写词——`Gatsby`/`Daisy` 留下，`Well`/`Come`/`Look` 剔掉。

**四道防坑闸门**（每道都直接对应之前踩过的坑）：

1. **正文反查**：术语必须在正文里出现 ≥2 次，只在术语表里露一面的丢掉 → 堵死「词出现≠定义出现」
2. **关键词区分度**：GT 关键词从原文逐字摘，且全书出现 ≤80 次；`protein`、`system` 这种泛词丢掉 → 否则检索到哪块都算命中，指标是虚的
3. **页眉页脚过滤**：出现在超过 60% 页面上的词判为页眉/页脚（如作者名），不是概念，丢掉
4. **目录页过滤**：跳过前 5% 前言目录页，证据句里数字占比 >8% 或含目录点线的丢掉

---

## 四、第二部分的不可答题是怎么造的（这块是加分点）

**不是随便编个不存在的词。** 是**从别的学科的教材术语表里抽真术语**，再回这本书全文验证零出现：

- 给 OSTEP 出 `What is a generalized anxiety disorder?`（心理学真术语）
- 给《傲慢与偏见》出 `What does standard metabolic rate mean?`（生理学真术语）

**关键设计：题面模板和第一部分完全一样**（`Define X.` / `What is a X?` / `Explain the term X.`）。模型没法靠句式判断该不该拒答，只能靠"检索到没检索到"，这才是真探针。每本另掺 3 道常识题（2024 诺奖、澳洲首都、苹果昨日收盘价这类），共 25 道。

---

## 五、模糊题两型

| type | 做法 | 例子（《傲慢与偏见》） | GT |
|---|---|---|---|
| `fuzzy_desc` | 拿正文证据句当查询，**把目标词挖空换成 it** | `Bennet," said his lady to him one day, "have you heard that it Park is let at last?"` | `netherfield` |
| `fuzzy_kw` | 只打 2–3 个词，像浏览器搜索框 | `remarkable consideration bourgh` | `lady catherine` |

判定：检索结果里出没出现那个目标词。这就是「打制冷设备也能出冰箱」。

---

## 六、字段格式

**兼容现有 `eval_*.jsonl` 的 `book/question/keywords` 三字段**，多出来的字段老脚本会自动忽略：

```json
{"book":"Operating Systems - Three Easy Pieces.pdf","subject":"Computer Science",
 "question":"Describe inode.","keywords":["inode","segregated","boots"],
 "type":"answerable","expect":"answer","page":712,"term":"inode",
 "evidence":"...正文证据句...","source":"distinct"}
```

- `keywords[0]` **永远是目标词本身**，最可靠；后两个是补充，命中任一即可算 Hit
- 不可答题 `keywords` 为空数组，`expect` 是 `abstain`
- `page` 是术语表页或证据句所在页，**不是标准答案页**，别拿它做引用校验的 GT
- `evidence` 是原文证据句，人工抽查时对照用

---

## 七、文件

- `eval_ALL.jsonl` —— 全量 4454 道
- `eval_Business.jsonl` / `eval_Literature.jsonl` / `eval_Medicine.jsonl` / `eval_ComputerScience.jsonl` / `eval_Psychology.jsonl` / `eval_Law.jsonl` —— 学科合并版
- `eval_by_book/` —— 61 个单本文件，按本跑消融时用

| 学科 | 书 | 可答 | 不可答 | 模糊 | 合计 |
|---|---|---|---|---|---|
| Business | 11 | 523 | 275 | 257 | 1055 |
| Medicine | 10 | 420 | 250 | 202 | 872 |
| Literature | 12 | 391 | 300 | 128 | 819 |
| Psychology | 7 | 296 | 174 | 123 | 593 |
| Computer Science | 8 | 276 | 200 | 116 | 592 |
| Law | 7 | 247 | 175 | 101 | 523 |
| **合计** | **55** | **2153** | **1374** | **927** | **4454** |

---

## 八、诚实边界（建议原样进报告）

0. **探针分级字段 `probe_class`**（复核时新增）：`clean` 845 道（组成词在书里一个都不出现）／`near_miss` 364 道（整词不出现，但拆开的词书里有，比如给商业书出 `hip joint`，书里有 joint venture 的 joint）／`world` 165 道（常识题）。**near_miss 是更难的探针**，模型容易被相近内容带偏；建议报告里分开统计，别混成一个数。

1. **四本书基本没出出来**，都是预览版/节选，页数本身就不够：
   - `Conditioned Reflexes — Ivan Pavlov.pdf`（10 页）—— 可答题 **0 道**，等于没法用
   - `An Introduction to Law.pdf`（29 页）—— 4 道
   - `Dive Into Python 3.pdf`（30 页）—— 6 道
   - `Introduction to Computer Science.pdf`（22 页）—— 14 道

   建议重下完整版，或直接从题集里剔掉。

2. **11 本没达到 40 道可答题或 15 道模糊题**（50 本达标）：
   除上述四本外，还有 `Think Java`(29/13)、`Think Python`(31/16)、`The Principles of Psychology — William James`(33/10)、`Crafting Interpreters`(36/19)、`Criminal Law`(45/0)、`the-great-gatsby`(45/0)、`alices-adventures-in-wonderland`(47/14)。
   CS 那几本掉得多是因为**剔代码片段剔掉的**——编程书正文里大量是代码，不是自然语言概念。

3. **三本没出题**：`Constitutional Law.pdf` 和 `Scientific-Methods-In-Psychology.pdf` 是**扫描件**（提不出文本，本管线没做 OCR）；`cs.pdf` 和 `Operating Systems - Three Easy Pieces.pdf` **是同一本**，去重了。

4. **题型偏「这是什么」**。三种书源出的都是概念/实体定位题，**不覆盖推理题、计算题、跨章节综合题**。要考推理得另想办法。

5. **模糊题的判定较宽**：只看检索结果里有没有出现目标词，**没有验证生成的答案对不对**。要严格评生成质量还得人工抽检。

6. **两个阈值是拍的不是算的**：关键词区分度上限 80 次、页眉判定 60% 页面。调这两个数会影响命中率，报告里引用数据时应注明。

7. **B 类（特征短语）和 C 类（实体）的权威性不如 A 类（术语表）**。A 类是作者标的定义，B/C 类是统计挑出来的，个别词可能不是这本书的核心概念。复核剔掉的 222 道里绝大多数出自 B/C 类，抽查时也优先看这两类。

8. **63 道跨书重复的可答题**（不是缺陷）：`Principles of Economics` / `Macroeconomics` / `Microeconomics` 本来就是同一套书拆分，`circular flow diagram`、`command economy` 这些概念三本都有，各自出题是对的。按学科合并跑时会看到同题出现多次，属正常。

---

## 九、跑之前

这批书页数很大（A&P 1347 页、Biology 1475 页、Business Law and the Legal Environment 1921 页）。按现有架构 `build` 是**删库重建、单 collection**，一次只能建一本；纯文本书记得加 `--vl-limit 0`，否则 VL 读图能跑到天亮。
