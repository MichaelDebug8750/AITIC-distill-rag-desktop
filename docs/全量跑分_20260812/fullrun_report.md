# webui 路径全量 · 结果报告

> 口径：题集中落在已建知识库上的全部题目（Think Python / Dreams / Criminal Law）。
> **与 v8final 的 93.7%/96.4%/3.6% 不可并列**——那是 CLI 口径，不同 PROMPT、不同 num_predict、不同代码路径。
> **耗时数据作废**：本次机器 GPU 锁频 225MHz/7W（约正常 1/30）。

已完成 **152 / 196** 题。

## 一、主要指标

| 指标 | 值 |
|---|---|
| 可答题命中率 | 80.8%（63/78） |
| 可答题未命中 | 5.1%（4） |
| 过度拒答 | 14.1%（11） |
| 不可答题精确拒答 | 100.0%（73/73） |
| 编造（不可答题给了实质答案） | 0.0%（0） |
| 请求失败 | 1 |

## 二、无引用句（本轮核心问题）

已作答的 67 题共 289 句，其中 **162 句没有引用（56.1%）**。

界面副标题承诺「每句话可溯源到原文页码」；无引用句会进逐句核验、多数判不出来，从而把可信度拖到「低」。

无引用占比最高的 8 题：

| 占比 | 书 | 问题 |
|---|---|---|
| 100% | Think Python | What is stack diagram? |
| 100% | Criminal Law | What is blood transfusion? |
| 100% | Think Python | What is a benchmarking? |
| 100% | The Interpretation of Dreams | Summarize what the text covers on censorship. |
| 86% | Criminal Law | Summarize what the text covers on prohibited conduct. |
| 83% | The Interpretation of Dreams | What is dream-content? |
| 83% | Criminal Law | What is manslaughter? |
| 83% | Criminal Law | Explain factual cause. |

## 三、可信度分布

- 低：48（71.6%）
- 中：18（26.9%）
- 高：1（1.5%）

## 四、Agent 与逐句核验

- 轮次分布：{1: 61, 3: 12, 2: 78, 0: 1}
- 逐句核验累计：裁剪 135 ｜ 悬空剔除 6 ｜ 未判定 195

## 五、分书

| 书 | 结果 |
|---|---|
| Think Python | {'拒答正确': 25, '命中': 24, '过度拒答': 2, '未命中': 1} |
| Criminal Law | {'拒答正确': 25, '未命中': 2, '命中': 16, '过度拒答': 7} |
| The Interpretation of Dreams | {'拒答正确': 23, '命中': 23, '过度拒答': 2, '请求失败': 1, '未命中': 1} |

## 六、需人工复核的题（16 条）

- **未命中** ｜ Criminal Law ｜ Explain offences.
  - 答案：The sources of criminal law, including statutes and legal principles, provide the framework for defining what constitutes an offence. [p.1] The material also me
- **过度拒答** ｜ Criminal Law ｜ How is actus discussed in this book?
  - 答案：[NO REFERENCE FOUND]
- **过度拒答** ｜ Criminal Law ｜ What is the role of mental element?
  - 答案：[NO REFERENCE FOUND]
- **过度拒答** ｜ The Interpretation of Dreams ｜ What does this book say about condensation?
  - 答案：[NO REFERENCE FOUND]
- **过度拒答** ｜ Criminal Law ｜ What does this book say about operative?
  - 答案：[NO REFERENCE FOUND]
- **过度拒答** ｜ The Interpretation of Dreams ｜ What is the role of wish-ful?
  - 答案：[NO REFERENCE FOUND]
- **过度拒答** ｜ Criminal Law ｜ Describe module guide.
  - 答案：[NO REFERENCE FOUND]
- **过度拒答** ｜ Think Python ｜ What is a shape?
  - 答案：[NO REFERENCE FOUND]
- **过度拒答** ｜ Criminal Law ｜ What is the role of result crimes?
  - 答案：[NO REFERENCE FOUND]
- **过度拒答** ｜ Criminal Law ｜ How is actual bodily discussed in this book?
  - 答案：[NO REFERENCE FOUND]
- **请求失败** ｜ The Interpretation of Dreams ｜ Summarize what the text covers on wishful impulses.
  - 答案：
- **未命中** ｜ Criminal Law ｜ Explain requires proof.
  - 答案：In criminal law, the prosecution must prove all elements of the offence, including actus reus, mens rea, and disprove defences. [p.14] The principle that the pr
- **未命中** ｜ Think Python ｜ justify spaces write
  - 答案：Adding spaces to control indentation in output can be achieved by using a string of space characters, as demonstrated in the example of the factorial function w
- **过度拒答** ｜ Criminal Law ｜ What is online library?
  - 答案：[NO REFERENCE FOUND]
- **未命中** ｜ The Interpretation of Dreams ｜ How is present dream discussed in this book?
  - 答案：Dreams are discussed as using the present tense in the same way as day-dreams, with wishes represented as fulfilled, but they differ in that their ideational co
- **过度拒答** ｜ Think Python ｜ written contains letters
  - 答案：[NO REFERENCE FOUND]
