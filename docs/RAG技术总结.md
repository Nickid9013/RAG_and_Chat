# 基于 RAG 的金融年报智能问答系统

> 基于检索增强生成（RAG）的**企业年报 / 研报问答系统**：支持 PDF 解析、向量检索、可选 LLM 重排与多题型结构化答案生成，适用于单公司问答与多公司比较场景。

---

## 一、项目概述

### 1.1 背景与目标

面向**金融年报、研报**等长文档，实现「给定问题 → 检索相关片段 → 基于上下文生成答案」的闭环。要求答案**可追溯**（引用页码）、**可配置**（多种检索与生成策略），并支持**单公司问答**与**多公司比较题**（如“哪家公司营收更高”）。

### 1.2 应用场景

- 单公司事实性问答：高管姓名、财务指标、是否具备某披露等（name / number / boolean / string）
- 多公司比较：将比较题拆解为各公司子问题，分别检索与生成后汇总
- 支持按公司维度的多文档管理（多份 PDF 对应多公司），便于扩展为更大规模文档集

### 1.3 端到端流程

```
PDF 报告 → MinerU 解析为 Markdown → 文本分块（LangChain）→ 向量化建库（FAISS）
    → 问题路由（按公司）→ 向量检索（± LLM 重排）→ 构造 RAG 上下文 → LLM 生成结构化答案（含推理与引用）
```

---

## 二、技术架构

### 2.1 技术栈总览

| 模块         | 技术选型 | 说明 |
|--------------|----------|------|
| 文档解析     | MinerU API | PDF → Markdown，支持 OCR，输出统一 Markdown 便于分块 |
| 文本分块     | LangChain RecursiveCharacterTextSplitter | 中文友好分隔符，按块 + 按页双结构输出 |
| 向量化       | DashScope text-embedding-v1 / OpenAI text-embedding-3-large | 可配置多提供商，批量调用 + 重试 |
| 向量索引     | FAISS (IndexFlatIP) | 按报告建索引，内积检索（等价归一化后余弦相似度） |
| 检索         | VectorRetriever + 可选 HybridRetriever | 按公司 + query 的 top-k；可选 LLM 重排（向量分与语义分加权融合） |
| 生成         | 多提供商 LLM（DashScope / OpenAI / Gemini / IBM） | 按题型 schema 生成结构化答案，支持分步推理与页码引用 |
| 结构化输出   | Pydantic 定义 Schema；OpenAI 用 Structured Outputs API，其余用 Prompt + 解析/修复 + 校验 | 保证答案格式稳定，便于引用校验与提交 |

### 2.2 核心流程说明

- **文档侧**：PDF 经 MinerU 转为 Markdown → 分块并写入 `chunked_reports/*.json`（含 chunks、pages、metainfo）→ 对 chunks 做 embedding 并写入 `vector_dbs/*.faiss`，元数据与 subset（公司名、sha1）一致。
- **问答侧**：从问题中识别公司（单公司或比较题拆解）→ 按公司定位对应向量库 → query embedding + FAISS top-k（可选父文档模式：按页聚合）→ 可选 LLM 对候选块重排并加权 → 将检索结果格式化为 RAG 上下文 → 调用 LLM 按题型生成带 `step_by_step_analysis`、`relevant_pages`、`final_answer` 的结构化答案 → 对引用页码做校验与补全，输出符合评测格式的 references。

---

## 三、核心功能与实现

### 3.1 文档解析与分块

- **PDF → Markdown**：调用 MinerU 云端 API，支持 OCR；PDF 通过 OSS URL 提交，轮询任务状态后下载并解压得到 `full.md`，支持批量处理目录下全部 PDF。
- **分块策略**：使用 LangChain `RecursiveCharacterTextSplitter`，分隔符适配中文（如 `\n\n`、`。`、`？` 等）；输出同时保留**块级**（chunks）与**页级**（pages）结构，便于后续「父文档检索」时按页返回完整内容。
- **元数据**：从 `subset.csv` 读取 `file_name` → `company_name` / `sha1`，写入每份报告的 `metainfo`，保证检索与引用时能按公司维度隔离与定位。

### 3.2 向量检索与可选重排

- **向量检索**：按 `company_name` 定位报告 → 对 query 做 embedding → FAISS 检索 top-k 块；支持 **父文档模式**（`return_parent_pages=True`）：将同一页的多个块合并为整页文本返回，减少上下文割裂。
- **LLM 重排（可选）**：先用向量检索取较多候选（如 30 条），再用 LLM 对每个/每批块打相关性分（0–1），与向量分数加权融合后重排序，取最终 top-n 作为 RAG 上下文，提升与问题语义的匹配度。
- **多提供商**：Embedding 与检索逻辑支持 DashScope / OpenAI 切换；重排模块支持 DashScope / OpenAI，便于在不同环境下复用同一套流程。

### 3.3 问题路由与多题型生成

- **单公司**：从问题文本中匹配 `subset.csv` 的 `company_name`（长匹配优先），确定唯一公司后执行检索 + 生成。
- **多公司比较**：使用 LLM 将比较题拆解为「每公司一题」（RephrasedQuestions），分别检索与生成后，再用 Comparative 类 prompt 汇总为比较答案。
- **题型与 Schema**：按问题类型选择不同生成 schema（name / number / boolean / names / string / comparative），统一要求输出**分步推理**、**简要总结**、**引用页码**和**最终答案**，便于可解释性与评测。

### 3.4 引用校验与可追溯性

- **页码校验**：对 LLM 返回的 `relevant_pages` 做校验：仅保留在本次检索结果中真实出现的页码，剔除幻觉引用；若引用页数过少则从检索结果中补足，过多则截断，保证引用与检索一致。
- **references 输出**：根据公司从 subset 取 `sha1`，生成 `{ pdf_sha1, page_index }` 列表，满足「答案可追溯到具体 PDF 与页码」的评测与审计需求。

### 3.5 结构化输出

- **Schema 定义**：使用 **Pydantic (BaseModel)** 在代码中统一定义各类答案与重排的 JSON 结构（如 `AnswerSchema`、`RephrasedQuestions`、`RetrievalRankingSingleBlock`），并将 schema 描述与示例写入 system prompt，保证多题型、多提供商下输出格式一致。
- **按提供商实现**：
  - **OpenAI**：使用 **Structured Outputs API**（`beta.chat.completions.parse` + Pydantic），由 API 保证返回符合 schema。
  - **DashScope / IBM / Gemini**：通过 **Prompt 约束 + 后处理**（去 markdown 包裹、`json.loads`、必要时 `json_repair` + Pydantic `model_validate`）；失败时可用「让模型改写成合法 JSON」的 reparse（AnswerSchemaFixPrompt）兜底，保证在无原生 structured output 的接口下仍能稳定拿到结构化结果。

---

## 四、难点与解决方案

| 难点 | 解决方案 |
|------|----------|
| 模型返回的引用页码可能不在检索结果中（幻觉） | 实现 `_validate_page_references`：仅保留检索结果中存在的页码，不足则从检索结果补足，过多则截断，保证 references 与上下文一致。 |
| 多 LLM 提供商下结构化输出接口不统一 | 用 Pydantic 统一定义 schema；OpenAI 走原生 Structured Outputs API；DashScope 等用 Prompt + 解析/修复 + 校验，并增加 reparse 兜底，保证输出格式稳定。 |
| 检索块过细导致上下文割裂 | 支持「父文档检索」：同一页的多个 chunk 合并为整页文本再参与构造 RAG 上下文，兼顾检索精度与阅读连贯性。 |
| 纯向量检索有时与问题语义匹配不足 | 增加可选「LLM 重排」：向量初筛 + LLM 对候选块打相关性分，加权融合后取 top-n，在可控成本下提升检索质量。 |

---

## 五、项目亮点总结

1. **完整 RAG 链路**：从 PDF 解析、分块、向量建库、检索、到多题型生成与引用校验，形成可配置、可复用的 pipeline，便于扩展数据与模型。
2. **检索策略可配置**：支持纯向量检索、父文档检索、LLM 重排、全文上下文等多种模式，通过 `RunConfig` 统一切换，适合做消融与优化实验。
3. **多提供商兼容**：Embedding / 生成 / 结构化输出均支持 DashScope、OpenAI、Gemini、IBM 等，便于在不同环境与成本约束下部署。
4. **答案可追溯**：通过页码校验与 `pdf_sha1 + page_index` 的 references 输出，满足「有据可查」的合规与评测需求。
5. **结构化输出与鲁棒性**：用 Pydantic 统一定义 schema，并结合 OpenAI 原生 API 与 Prompt+解析/修复/reparse 两种方式，在多提供商下保证答案结构稳定、便于下游使用。

---

## 六、配置与入口速查

| 能力 | 配置 / 入口 | 说明 |
|------|-------------|------|
| PDF 批量转 Markdown | `Pipeline.export_reports_to_markdown_batch()` | 解析 pdf_reports 目录下全部 PDF |
| 分块 | `Pipeline.chunk_reports()` | 依赖 subset.csv 元数据 |
| 向量建库 | `Pipeline.create_vector_dbs()` | DashScope embedding + FAISS |
| 父文档检索 | `RunConfig.parent_document_retrieval=True` | 按页返回完整文本 |
| LLM 重排 | `RunConfig.llm_reranking=True` | 向量初筛 + LLM 打分融合 |
| 全文上下文 | `RunConfig.full_context=True` | 不检索，用该公司全部页作上下文 |
| 生成模型 | `RunConfig.api_provider` / `answering_model` | 多提供商与模型名 |

---

*本文档基于当前代码库整理，描述均为实际实现。*
