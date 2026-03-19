# 项目运行参数表（以 answer_single_question 为入口）

入口示例：

```python
pipeline = Pipeline(root_path, run_config=max_config)  # __main__ 当前用法
pipeline.answer_single_question(question="中芯国际2025年年度报告的营收是多少？", kind="string")
```

**实际流程约定**：下文「实际取值」以 `if __name__ == "__main__"` 中 `Pipeline(here()/ "data"/ "stock_data", run_config=max_config)` 且调用上述单问为例。

---

## 1. 入口层：Pipeline

### 1.1 Pipeline 构造参数（创建 pipeline 时）


| 参数                     | 类型        | 默认值                | 实际取值                             | 说明                    |
| ---------------------- | --------- | ------------------ | -------------------------------- | --------------------- |
| `root_path`            | Path      | —                  | `here() / "data" / "stock_data"` | 数据根目录                 |
| `subset_name`          | str       | `"subset.csv"`     | `"subset.csv"`                   | 公司列表 CSV 文件名          |
| `questions_file_name`  | str       | `"questions.json"` | `"questions.json"`               | 问题列表 JSON 文件名（单问时未使用） |
| `pdf_reports_dir_name` | str       | `"pdf_reports"`    | `"pdf_reports"`                  | PDF 报告目录名             |
| `run_config`           | RunConfig | `RunConfig()`      | `max_config`                     | 运行配置，见下表              |


### 1.2 RunConfig（run_config 各字段）


| 参数                          | 类型   | 默认值           | 实际取值（max_config）         | 说明                                       |
| --------------------------- | ---- | ------------- | ------------------------ | ---------------------------------------- |
| `use_serialized_tables`     | bool | False         | False                    | 是否使用序列化表格（影响 `databases` 目录名等）           |
| `parent_document_retrieval` | bool | False         | True                     | 检索是否返回整页文本（当前 JSON 无 pages，实际按 chunk 返回） |
| `use_vector_dbs`            | bool | True          | True                     | 是否使用向量检索                                 |
| `llm_reranking`             | bool | False         | True                     | 是否在向量检索后用 LLM 重排                         |
| `llm_reranking_sample_size` | int  | 30            | 30                       | 参与 LLM 重排的候选数量                           |
| `top_n_retrieval`           | int  | 10            | 10                       | 检索返回的 top-k 条 chunk 数量                   |
| `parallel_requests`         | int  | 1             | 4（单问时仍为 1）               | 批量问题时的并发数                                |
| `pipeline_details`          | str  | ""            | 见 pipeline.py max_config | pipeline 描述，写答案文件时使用                     |
| `submission_file`           | bool | True          | True                     | 是否按提交格式写答案文件                             |
| `full_context`              | bool | False         | False                    | True 时返回该公司全部文本，不做检索                     |
| `api_provider`              | str  | `"dashscope"` | `"dashscope"`            | LLM 提供商                                  |
| `answering_model`           | str  | `"qwen-turbo"` | `"qwen-turbo"`            | 作答使用的模型名                                 |
| `config_suffix`             | str  | ""            | `"_qwen_plus"`           | 配置后缀，用于 answers 文件名等                     |


### 1.3 answer_single_question 直接参数


| 参数         | 类型  | 默认值        | 实际取值                     | 说明               |
| ---------- | --- | ---------- | ------------------------ | ---------------- |
| `question` | str | —          | `"中芯国际2025年年度报告的营收是多少？"` | 问题文本             |
| `kind`     | str | `"string"` | `"string"`               | 答案类型，对应作答 schema |


---

## 2. 路径层：PipelineConfig（由 root_path + run_config 推导）


| 路径/配置                 | 说明        | 实际取值（root_path=…/stock_data, max_config）                   |
| --------------------- | --------- | ---------------------------------------------------------- |
| `subset_path`         | 公司名列表 CSV | `…/data/stock_data/subset.csv`                             |
| `questions_file_path` | 单问时未用     | None                                                       |
| `pdf_reports_dir`     | PDF 报告目录  | `…/data/stock_data/pdf_reports`                            |
| `answers_file_path`   | 答案输出 JSON | `…/data/stock_data/answers_qwen_plus.json`                 |
| `debug_data_path`     | 调试数据目录    | `…/data/stock_data/debug_data`                             |
| `databases_path`      | 数据库根目录    | `…/data/stock_data/databases`（use_serialized_tables=False） |
| `vector_db_dir`       | 向量库目录     | `…/data/stock_data/databases/vector_dbs`                   |
| `documents_dir`       | 分块报告目录    | `…/data/stock_data/databases/chunked_reports`              |


---

## 3. 问题处理层：QuestionsProcessor

由 `answer_single_question` 内部创建，参数来自 `run_config` + 路径：


| 参数                          | 类型        | 默认值/来源                                 | 实际取值（max_config + 单问）         | 说明                   |
| --------------------------- | --------- | -------------------------------------- | ----------------------------- | -------------------- |
| `vector_db_dir`             | Path      | `paths.vector_db_dir`                  | `…/databases/vector_dbs`      | 向量库目录                |
| `documents_dir`             | Path      | `paths.documents_dir`                  | `…/databases/chunked_reports` | 分块报告 JSON 目录         |
| `questions_file_path`       | Path/None | None（单问）                               | None                          | 问题文件路径               |
| `new_challenge_pipeline`    | bool      | True                                   | True                          | 使用 subset 抽取公司名等新流程  |
| `subset_path`               | Path      | `paths.subset_path`                    | `…/stock_data/subset.csv`     | 公司列表 CSV             |
| `parent_document_retrieval` | bool      | `run_config.parent_document_retrieval` | True                          | 是否按“整页”返回（当前按 chunk） |
| `llm_reranking`             | bool      | `run_config.llm_reranking`             | True                          | 是否启用 LLM 重排          |
| `llm_reranking_sample_size` | int       | `run_config.llm_reranking_sample_size` | 30                            | 重排候选数量               |
| `top_n_retrieval`           | int       | `run_config.top_n_retrieval`           | 10                            | 检索 top-k             |
| `parallel_requests`         | int       | 1（单问固定）                                | 1                             | 并发请求数                |
| `api_provider`              | str       | `run_config.api_provider`              | `"dashscope"`                 | LLM 提供商              |
| `answering_model`           | str       | `run_config.answering_model`           | `"qwen-turbo"`                 | 作答模型                 |
| `full_context`              | bool      | `run_config.full_context`              | False                         | 是否全量上下文不检索           |


---

## 4. 检索层

### 4.1 检索器选择

- `llm_reranking == True` → 使用 **HybridRetriever**（VectorRetriever + LLMReranker）
- `llm_reranking == False` → 使用 **VectorRetriever**

**实际取值**：max_config 下 `llm_reranking=True` → 使用 **HybridRetriever**。

### 4.2 VectorRetriever 构造参数


| 参数                   | 类型   | 默认值           | 实际取值                          | 说明                         |
| -------------------- | ---- | ------------- | ----------------------------- | -------------------------- |
| `vector_db_dir`      | Path | —             | `…/databases/vector_dbs`      | 向量库目录                      |
| `documents_dir`      | Path | —             | `…/databases/chunked_reports` | 分块报告目录                     |
| `embedding_provider` | str  | `"dashscope"` | `"dashscope"`                 | 向量化提供商（代码未从 run_config 传入） |


### 4.3 retrieve_by_company_name 调用参数（来自 get_answer_for_company）


| 参数                          | 类型   | 默认值/来源                             | 实际取值                     | 说明                         |
| --------------------------- | ---- | ---------------------------------- | ------------------------ | -------------------------- |
| `company_name`              | str  | subset 从 question 抽取               | `"中芯国际"`                 | 公司名                        |
| `query`                     | str  | question                           | `"中芯国际2025年年度报告的营收是多少？"` | 检索查询                       |
| `llm_reranking_sample_size` | int  | QuestionsProcessor                 | 30                       | 首轮向量候选数（HybridRetriever 用） |
| `top_n`                     | int  | QuestionsProcessor.top_n_retrieval | 10                       | 最终返回条数                     |
| `return_parent_pages`       | bool | QuestionsProcessor                 | True                     | 是否按页返回（当前按 chunk）          |


### 4.4 HybridRetriever.retrieve_by_company_name 额外/内部参数


| 参数                     | 类型    | 默认值 | 实际取值 | 说明                                                     |
| ---------------------- | ----- | --- | ---- | ------------------------------------------------------ |
| `documents_batch_size` | int   | 10  | 10   | 每批送入 LLM 重排的文档数（调用方未传，用默认）                             |
| `top_n`                | int   | 6   | 10   | 重排后返回条数（由 get_answer_for_company 传 top_n_retrieval=10） |
| `llm_weight`           | float | 0.7 | 0.7  | LLM 分数权重（调用方未传，用默认）                                    |


---

## 5. 重排层：LLMReranker

### 5.1 构造参数


| 参数         | 类型  | 默认值           | 实际取值          | 说明                      |
| ---------- | --- | ------------- | ------------- | ----------------------- |
| `provider` | str | `"dashscope"` | `"dashscope"` | HybridRetriever 未传入，用默认 |


### 5.2 rerank_documents 参数


| 参数                     | 类型    | 默认值 | 实际取值                     | 说明                     |
| ---------------------- | ----- | --- | ------------------------ | ---------------------- |
| `query`                | str   | —   | `"中芯国际2025年年度报告的营收是多少？"` | 查询                     |
| `documents`            | list  | —   | 向量检索返回的 30 条候选           | 待重排文档列表                |
| `documents_batch_size` | int   | 4   | 10                       | HybridRetriever 默认传 10 |
| `llm_weight`           | float | 0.7 | 0.7                      | LLM 分数权重               |


### 5.3 重排模型（写死在 reranking.py）


| 用途             | 模型                       | 实际使用                                              |
| -------------- | ------------------------ | ------------------------------------------------- |
| OpenAI 单块重排    | `gpt-4o-mini-2024-07-18` | provider=openai 时                                 |
| DashScope 单块重排 | `qwen-turbo`              | documents_batch_size=1 时                          |
| DashScope 多块重排 | `qwen-turbo`              | ✓ 当前流程（documents_batch_size=10，30 条分 3 批，每批 10 条） |


---

## 6. 作答层：APIProcessor（get_answer_from_rag_context）

### 6.1 APIProcessor 构造参数


| 参数         | 类型      | 默认值/来源                    | 实际取值          | 说明      |
| ---------- | ------- | ------------------------- | ------------- | ------- |
| `provider` | Literal | `run_config.api_provider` | `"dashscope"` | LLM 提供商 |


### 6.2 get_answer_from_rag_context 参数


| 参数            | 类型  | 来源                         | 实际取值                     | 说明        |
| ------------- | --- | -------------------------- | ------------------------ | --------- |
| `question`    | str | 入口 question                | `"中芯国际2025年年度报告的营收是多少？"` | 问题文本      |
| `rag_context` | str | 检索结果格式化                    | 重排后 10 条 chunk 拼成的上下文字符串 | RAG 上下文   |
| `schema`      | str | 入口 kind                    | `"string"`               | 作答 schema |
| `model`       | str | run_config.answering_model | `"qwen-turbo"`            | 作答模型名     |


---

## 7. 单问流程中 kind（schema）取值与含义


| kind      | 说明       | 对应 Prompt / Schema                            | 本例实际取值 |
| --------- | -------- | --------------------------------------------- | ------ |
| `string`  | 开放性文本答案  | AnswerWithRAGContextStringPrompt              | ✓ 使用   |
| `number`  | 数值类答案    | AnswerWithRAGContextNumberPrompt              | —      |
| `boolean` | 是/否类答案   | AnswerWithRAGContextBooleanPrompt             | —      |
| `names`   | 多选/名单类答案 | AnswerWithRAGContextNamesPrompt               | —      |
| 多公司时内部使用  | 比较类答案    | ComparativeAnswerPrompt（schema="comparative"） | —      |


---

## 8. 环境变量（与单问运行相关）


| 变量                  | 说明                                               | 实际流程是否使用                          |
| ------------------- | ------------------------------------------------ | --------------------------------- |
| `DASHSCOPE_API_KEY` | DashScope（Qwen）API Key                           | ✓ 使用（embedding、重排、作答均为 dashscope） |
| `OPENAI_API_KEY`    | 当 api_provider / embedding_provider 为 openai 时使用 | 未使用（max_config 为 dashscope）       |
| `JINA_API_KEY`      | 仅在使用 JinaReranker 时需要                            | 未使用                               |


---

## 9. 参数传递简图（单问入口 + 实际取值）

```
answer_single_question(question="中芯国际2025年年度报告的营收是多少？", kind="string")
  → RunConfig(max_config): parent_document_retrieval=True, llm_reranking=True,
      llm_reranking_sample_size=30, top_n_retrieval=10, api_provider="dashscope",
      answering_model="qwen-turbo", full_context=False
  → QuestionsProcessor(..., parallel_requests=1)
  → process_single_question(question, kind="string")
       → _extract_companies_from_subset(question)  → company_name="中芯国际"
       → get_answer_for_company("中芯国际", question, schema="string")
            → HybridRetriever(vector_db_dir, documents_dir)
            → retrieve_by_company_name(company_name="中芯国际", query=question,
                llm_reranking_sample_size=30, top_n=10, return_parent_pages=True)
                 → 向量检索 30 条 → rerank_documents(..., documents_batch_size=10, llm_weight=0.7)
                 → 重排后取前 10 条
            → APIProcessor(provider="dashscope").get_answer_from_rag_context(
                  question, rag_context, schema="string", model="qwen-turbo")
```

以上为以 `pipeline.answer_single_question(question="...", kind="string")` 为入口、且 `run_config=max_config` 时的项目运行参数及实际取值整理。