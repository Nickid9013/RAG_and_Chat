# RAG_and_Chat

基于检索增强生成（RAG）的**企业年报 / 研报智能问答系统**：支持 PDF 解析、向量检索、可选 LLM 重排与多题型结构化答案生成，适用于单公司问答与多公司比较场景。

---

## 功能概览

- **单公司问答**：高管姓名、财务指标、是否具备某披露等（支持 string / number / boolean / names 等题型）
- **多公司比较**：自动拆解比较题，按公司分别检索与生成后汇总
- **多轮对话**：支持短期上下文记忆（最近 8 轮）
- **可追溯引用**：答案附带「文件名 + 行号」形式的相关来源

技术栈：MinerU（PDF→Markdown）、LangChain 分块、FAISS 向量检索、多提供商 LLM（DashScope / OpenAI / Gemini / IBM）、Pydantic 结构化输出。

---

## 环境要求

- Python 3.10+
- 依赖见 `requirements.txt`

---

## 安装

```bash
# 进入项目根目录
cd /path/to/your/RAG_and_Chat

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux / macOS

# 安装依赖
pip install -r requirements.txt

# 可选：以可编辑方式安装项目（便于 import src）
pip install -e .
```

---

## 配置

在项目根目录创建 `.env` 文件，按需配置 API 密钥（未用到的可省略）：

```env
# 向量化与作答（二选一或按需）
DASHSCOPE_API_KEY=sk-xxx        # 阿里云 DashScope（默认 embedding + 作答）
OPENAI_API_KEY=sk-xxx            # OpenAI（embedding / 作答 / 重排）

# PDF 解析（MinerU 云服务）
MINERU_API_KEY=xxx               # MinerU API 密钥

# 可选：LLM 重排、其他厂商
JINA_API_KEY=xxx                 # Jina 重排（若用）
GEMINI_API_KEY=xxx               # Google Gemini（若用）
IBM_API_KEY=xxx                  # IBM 模型（若用）
```

数据目录约定：默认使用 `data/stock_data`（可在启动代码中修改 `root_path`），其下需有 `subset.csv`、`questions.json`、`03_reports_markdown`（Markdown 报告）、`databases/vector_dbs`（已建好的向量库）等。首次使用需先完成「PDF → Markdown → 分块 → 建库」流程（见下方 Pipeline 方式）。

---

## 启动方式

### 方式一：Pipeline 命令行交互（推荐本地调试）

在项目根目录执行：

```bash
python src/pipeline.py
```

- 会加载 `data/stock_data` 与默认配置（`max_config`），进入**交互式问答**：在终端输入问题，输入 `q` 退出。
- 数据路径与运行配置在 `src/pipeline.py` 末尾 `if __name__ == "__main__"` 中修改（如 `root_path`、`run_config`）。
- 同一文件中可取消注释以执行：批量 PDF 转 Markdown、分块、建向量库、批量处理问题等。

适用：本地单机、脚本化调用、不依赖浏览器。

---

### 方式二：Streamlit Web 界面

在项目根目录执行：

```bash
# 若未安装 Streamlit，先安装
pip install streamlit

streamlit run app_streamlit.py
```

- 浏览器会自动打开 Streamlit 应用（默认 http://localhost:8501）。
- 使用 `data/stock_data` 与 `max_config`，支持多轮对话、查看推理与引用。
- 数据路径在 `app_streamlit.py` 中通过 `root_path = Path("data/stock_data")` 配置。

适用：需要 Web UI、多轮对话与更好展示效果时使用。

---

## 项目结构（简要）

```
.
├── app_streamlit.py       # Streamlit 入口
├── main.py                # Click CLI 入口（可选）
├── src/
│   ├── pipeline.py        # 主流程与 Pipeline 入口
│   ├── pdf_mineru.py      # PDF → Markdown（MinerU API）
│   ├── text_splitter.py   # 分块
│   ├── ingestion.py       # 向量建库
│   ├── retrieval.py       # 检索（向量 + 可选重排）
│   ├── reranking.py       # LLM 重排
│   ├── questions_processing.py  # 问答与比较题
│   ├── api_requests.py     # 多厂商 LLM 调用
│   └── prompts.py         # 提示词与 Schema
├── data/
│   └── stock_data/        # 默认数据目录（subset、markdown、向量库等）
├── docs/                  # 项目文档与开发记录
├── requirements.txt
└── README.md
```

---

## 文档

- `docs/RAG技术总结.md`：技术架构与实现总结
- `docs/project_logic_summary.md`：代码逻辑与流程概览
- `docs/pipeline_params.md`：运行参数与取值说明
- `docs/项目开发记录.md`：按日期的修改记录

---

## 许可证

请按项目实际情况添加 License 说明。
