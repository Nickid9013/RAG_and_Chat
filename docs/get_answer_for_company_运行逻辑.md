# `get_answer_for_company` 完整运行逻辑

本文档梳理 `QuestionsProcessor.get_answer_for_company(company_name, question, schema)` 的端到端执行流程，以及所涉及的 `retrieval`、`api_requests`、`reranking`、`prompts` 等模块的实现要点。

---

## 一、入口与参数

- **入口**：`src/questions_processing.py` → `QuestionsProcessor.get_answer_for_company(company_name, question, schema)`
- **含义**：针对**单个公司**，用 `question` 做检索，再基于检索到的上下文调用 LLM 按 `schema` 生成答案。
- **调用前依赖**：`QuestionsProcessor` 已根据 `llm_reranking`、`full_context`、`return_parent_pages`、`top_n_retrieval`、`llm_reranking_sample_size` 等完成初始化；`vector_db_dir`、`documents_dir`、`openai_processor`、`subset_path`、`new_challenge_pipeline` 等可用。

---

## 二、整体流程概览

```
1. 根据 llm_reranking 选择检索器（VectorRetriever 或 RerankedRetriever）
2. 根据 full_context 选择检索方式：全文 或 按 query 检索
3. 检索得到 retrieval_results（列表，每项 { page, text [, distance] }）
4. _format_retrieval_results → 拼成 rag_context 字符串
5. openai_processor.get_answer_from_rag_context(question, rag_context, schema, model) → answer_dict
6. 若 new_challenge_pipeline：校验页码、生成 references，写回 answer_dict
7. 返回 answer_dict
```

下面按步骤展开，并标出涉及的其他模块与实现。

---

## 三、分步逻辑与涉及模块

### 3.1 选择并初始化检索器（本文件）

```python
if self.llm_reranking:
    retriever = RerankedRetriever(vector_db_dir=..., documents_dir=...)
else:
    retriever = VectorRetriever(vector_db_dir=..., documents_dir=...)
```

- **VectorRetriever**（`src/retrieval.py`）  
  - `__init__`：读 `documents_dir` 下所有 `*.json`（chunked_reports），用 `metainfo.sha1` 找到对应 `vector_db_dir/{sha1}.faiss`，调用 `faiss.read_index`，得到 `all_dbs = [ { name, vector_db, document } ]`；并根据 `embedding_provider`（dashscope/openai）初始化 embedding 客户端（DashScope TextEmbedding 或 OpenAI embeddings）。  
  - 不在本步做检索，只做“加载所有向量库 + 文档”的准备工作。

- **RerankedRetriever**（`src/retrieval.py`）  
  - `__init__`：内部创建一个 `VectorRetriever` 和一个 `LLMReranker()`（`src/reranking.py`）。  
  - `LLMReranker` 会加载重排用的 system prompt（单块/多块）和 Pydantic schema，并初始化 DashScope/OpenAI 的生成客户端。

---

### 3.2 执行检索：全文 vs 按 query

**分支 A：`self.full_context == True`（全文上下文）**

- 调用：`retrieval_results = retriever.retrieve_all(company_name)`
- **VectorRetriever.retrieve_all**（`retrieval.py`）：  
  - 在 `all_dbs` 里按 `metainfo.company_name == company_name` 找到唯一报告。  
  - 从该报告的 `document["content"]["pages"]` 按页号排序，逐页构造 `{ "distance": 0.5, "page", "text" }`，**不做 query 的向量检索**，直接返回该公司全部页的列表。  
- **RerankedRetriever** 没有实现 `retrieve_all`，因此若开启 `llm_reranking` 又开 `full_context`，会走 VectorRetriever 的 `retrieve_all`（因为 RerankedRetriever 内部只有 vector_retriever，没有自己实现 retrieve_all，需要看代码是否在 full_context 时改用 vector_retriever）—— 当前 `get_answer_for_company` 里是同一个 `retriever` 调 `retrieve_all`，所以若 `retriever` 是 RerankedRetriever，会报错（RerankedRetriever 无 retrieve_all）。实际配置通常不会同时开 llm_reranking 和 full_context，这里逻辑上是“全文时用 retriever.retrieve_all”，只有 VectorRetriever 支持 retrieve_all。

**分支 B：常规按 query 检索（`full_context == False`）**

- 调用：  
  `retrieval_results = retriever.retrieve_by_company_name(company_name=..., query=question, llm_reranking_sample_size=..., top_n=self.top_n_retrieval, return_parent_pages=self.return_parent_pages)`

- **VectorRetriever.retrieve_by_company_name**（`retrieval.py`）：  
  1. 在 `all_dbs` 中按 `company_name`（或 `file_name` 包含）定位到该公司的报告（document + vector_db）。  
  2. 用 `_get_embedding(query)` 得到 query 的向量（DashScope text-embedding-v1 或 OpenAI text-embedding-3-large）。  
  3. `vector_db.search(embedding_array, k=actual_top_n)`（FAISS IndexFlatIP）得到 `distances, indices`。  
  4. 根据 `indices` 从 `document["content"]["chunks"]` 取对应 chunk；若 `return_parent_pages` 且存在 `content.pages`，则按 chunk 的 `page` 找到整页，**以页为单位去重**后放入结果（同一页只出现一次，text 为整页内容），否则每条结果为一个 chunk 的 `page` + `text`。  
  5. 返回列表：`[ { "distance", "page", "text" }, ... ]`。

- **RerankedRetriever.retrieve_by_company_name**（`retrieval.py`）：  
  1. 调用内部 `vector_retriever.retrieve_by_company_name(..., top_n=llm_reranking_sample_size, return_parent_pages=...)`，得到较多候选（如 28 条）。  
  2. 调用 `self.reranker.rerank_documents(query, documents=vector_results, documents_batch_size, llm_weight)`。  
  3. **LLMReranker.rerank_documents**（`src/reranking.py`）：  
     - 将 documents 按 `documents_batch_size` 分批；每批把 `doc["text"]` 和 query 一起送给 LLM（单块用 `get_rank_for_single_block`，多块用 `get_rank_for_multiple_blocks`），拿到 `relevance_score`（0–1）。  
     - 融合分：`combined_score = llm_weight * relevance_score + (1 - llm_weight) * doc["distance"]`（此处 distance 为向量距离，越大越相似时需注意符号；代码里按 combined_score 降序排，即认为分数越高越相关）。  
     - 对所有文档按 `combined_score` 降序排序后，由 RerankedRetriever 再取前 `top_n`（即 `self.top_n_retrieval`）条返回。  
  4. 返回列表：`[ { "distance", "page", "text", "relevance_score", "combined_score" }, ... ]`，顺序为重排后的顺序。

无论哪种检索路径，**retrieval_results** 都是“列表，元素至少含 `page`、`text`”，供后续拼上下文和校验页码使用。

---

### 3.3 格式化检索结果为 RAG 上下文（本文件）

```python
rag_context = self._format_retrieval_results(retrieval_results)
```

- **_format_retrieval_results**（`questions_processing.py`）：  
  - 遍历 `retrieval_results`，对每条 `result` 拼成：  
    `Text retrieved from page {page}: \n"""\n{text}\n"""`  
  - 再用 `"\n\n---\n\n".join(...)` 连成一个大字符串。  
- 得到的就是传给 LLM 的“仅基于此上下文作答”的 **rag_context** 文本。

---

### 3.4 调用 LLM 生成结构化答案（api_requests）

```python
answer_dict = self.openai_processor.get_answer_from_rag_context(
    question=question, rag_context=rag_context, schema=schema, model=self.answering_model
)
```

- **APIProcessor.get_answer_from_rag_context**（`src/api_requests.py`）：  
  1. **选 prompt 和 schema**：`_build_rag_context_prompts(schema)` 根据 `schema`（name / number / boolean / names / string / comparative）选择对应的：  
     - system_prompt（来自 `prompts.AnswerWithRAGContext*Prompt` 或 `ComparativeAnswerPrompt`）  
     - response_format（Pydantic 类，如 `AnswerWithRAGContextNamePrompt.AnswerSchema`）  
     - user_prompt 模板（通常含 `{context}`、`{question}`）  
  2. **发请求**：`self.processor.send_message(model=model, system_content=system_prompt, human_content=user_prompt.format(context=rag_context, question=question), is_structured=True, response_format=response_format)`  
     - 实际会路由到当前 provider（如 BaseDashscopeProcessor）：用 system + user 拼成 messages，调用 DashScope Generation.call；若为结构化，会对返回的 content 做 JSON 解析（去 markdown 包裹、json.loads 等），再按 response_format 校验/兜底。  
  3. **兜底**：若返回的 dict 缺少 `step_by_step_analysis` 等字段，会用默认值或从 `final_answer` 里尝试解析，保证返回的 answer_dict 至少包含：`step_by_step_analysis`、`reasoning_summary`、`relevant_pages`、`final_answer`。  

- **answer_dict** 中与后续逻辑直接相关的是 **relevant_pages**（LLM 声称引用的页码列表）和 **final_answer** 等。

---

### 3.5 引用校验与 references 生成（本文件，仅 new_challenge_pipeline）

```python
if self.new_challenge_pipeline:
    pages = answer_dict.get("relevant_pages", [])
    validated_pages = self._validate_page_references(pages, retrieval_results)
    answer_dict["relevant_pages"] = validated_pages
    answer_dict["references"] = self._extract_references(validated_pages, company_name)
```

- **_validate_page_references**（`questions_processing.py`）：  
  - 从 `retrieval_results` 里抽出真实出现过的页码集合。  
  - 只保留 LLM 返回的 `claimed_pages` 中**在检索结果里存在的页码**，去掉“幻觉页码”；若保留的页数少于 `min_pages`（默认 2），则从 `retrieval_results` 里按顺序补页直到够；若超过 `max_pages`（默认 8）则截断。  
  - 返回 **validated_pages**（整型页码列表）。

- **_extract_references**（`questions_processing.py`）：  
  - 读 `subset_path`（subset.csv），按 `company_name` 查到该公司对应的 **sha1**。  
  - 对每个 `validated_pages` 中的页码生成 `{ "pdf_sha1": company_sha1, "page_index": page }`。  
  - 返回 references 列表，用于评测/提交格式。

最后把 **validated_pages** 写回 `answer_dict["relevant_pages"]`，**references** 写回 `answer_dict["references"]`，并返回 **answer_dict**。

---

## 四、数据流小结

| 阶段           | 输入                         | 输出 / 产物 |
|----------------|------------------------------|-------------|
| 检索器选择     | llm_reranking                | VectorRetriever 或 RerankedRetriever |
| 检索           | company_name, question, 配置 | retrieval_results: List[{ page, text, ... }] |
| 格式化上下文   | retrieval_results            | rag_context: str |
| LLM 生成       | question, rag_context, schema, model | answer_dict: { step_by_step_analysis, reasoning_summary, relevant_pages, final_answer } |
| 引用校验与生成 | answer_dict.relevant_pages, retrieval_results, company_name | validated_pages, answer_dict.references |

---

## 五、涉及模块与文件一览

| 模块/类/函数 | 文件 | 作用 |
|-------------|------|------|
| QuestionsProcessor.get_answer_for_company | questions_processing.py | 入口与流程编排 |
| QuestionsProcessor._format_retrieval_results | questions_processing.py | 检索结果 → rag_context 字符串 |
| QuestionsProcessor._validate_page_references | questions_processing.py | 页码校验与补全/截断 |
| QuestionsProcessor._extract_references | questions_processing.py | 根据 subset 生成 pdf_sha1 + page_index |
| VectorRetriever | retrieval.py | 加载 FAISS + 文档；retrieve_by_company_name（向量 top-k）、retrieve_all（全文页） |
| RerankedRetriever | retrieval.py | 向量检索 + LLMReranker.rerank_documents，再取 top_n |
| LLMReranker.rerank_documents | reranking.py | 对候选文档用 LLM 打 relevance_score，与向量分加权排序 |
| APIProcessor.get_answer_from_rag_context | api_requests.py | 按 schema 选 prompt，调 processor.send_message，结构化解析与兜底 |
| APIProcessor._build_rag_context_prompts | api_requests.py | 根据 schema 返回 (system_prompt, response_format, user_prompt) |
| prompts.AnswerWithRAGContext*Prompt / ComparativeAnswerPrompt | prompts.py | RAG 作答的 system/user 模板与 Pydantic AnswerSchema |
| BaseDashscopeProcessor.send_message（等） | api_requests.py | 实际发 LLM 请求、解析 JSON、返回 dict |

按上述顺序阅读代码即可还原 `get_answer_for_company` 的完整运行逻辑及各模块职责。
