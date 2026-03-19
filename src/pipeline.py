from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
from typing import Optional, List, Dict
import logging
import os
import json
import pandas as pd
import shutil
import time

from src import pdf_mineru
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.questions_processing import QuestionsProcessor

@dataclass
class RunConfig:
    """运行流程参数配置，控制数据路径、检索方式、LLM 与并发等。"""
    use_serialized_tables: bool = False   # 是否使用序列化表格（影响 databases 目录名等路径）
    parent_document_retrieval: bool = False  # 检索时是否返回整页文本而非单块
    use_vector_dbs: bool = True   # 是否使用向量检索；检索层未根据此配置切换
    llm_reranking: bool = False   # 是否在向量检索后用 LLM 对候选块重排序
    llm_reranking_sample_size: int = 30   # LLM 重排时参与排序的候选数量
    top_n_retrieval: int = 10     # 检索返回的 top-k 条 chunk 数量
    parallel_requests: int = 1    # 批量问题时并发请求数，过大易触发 Qwen 限流
    pipeline_details: str = ""    #  pipeline 描述，写入答案文件等
    submission_file: bool = True  # 是否按提交格式写答案文件（含 answer_details 等）
    full_context: bool = False    # 为 True 时直接返回该公司全部文本，不做检索
    api_provider: str = "dashscope"   # LLM 提供商：dashscope / openai / gemini / ibm
    answering_model: str = "qwen-turbo"   # 作答使用的模型名
    config_suffix: str = ""       # 配置后缀，用于 answers 文件名等区分不同 run

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"

        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname


class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        # 初始化主流程，加载路径和配置
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """根据配置初始化所有路径"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )
  
    # 这一步的意义存疑？？
    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV。
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

# 以下开始正式的处理流程
# 文本解析
    def export_reports_to_markdown(self, file_name):
        """
        使用 pdf_mineru.py，将指定 PDF 文件转换为 markdown，并放到 reports_markdown_dirname 目录下。
        :param file_name: PDF 文件名（如 '【财报】中芯国际：中芯国际2024年年度报告.pdf'）
        """
        # 调用 pdf_mineru 获取 task_id，zip 保存到 root_path/zips，解压到 root_path/uzips
        print(f"开始处理: {file_name}")
        task_id = pdf_mineru.get_task_id(file_name)
        print(f"task_id: {task_id}")
        extract_dir = pdf_mineru.get_result(task_id, base_path=self.paths.root_path)
        if extract_dir is None:
            print("mineru 未返回解压目录，跳过")
            return

        md_path = Path(extract_dir) / "full.md"
        if not md_path.exists():
            print(f"未找到 markdown 文件: {md_path}")
            return
        # 创建目标目录
        # os.makedirs(self.paths.reports_markdown_path, exist_ok=True)
        path = Path(self.paths.reports_markdown_path)
        path.mkdir(parents=True, exist_ok=True)
        # 目标文件名为原始 file_name，扩展名改为 .md
        base_name = os.path.splitext(file_name)[0]
        target_path = path / f"{base_name}.md"
        shutil.move(md_path, target_path)
        print(f"已将 {md_path} 移动到 {target_path}")

    def export_reports_to_markdown_batch(self, dir_path: Path = None):
        """
        将指定目录下的所有 PDF 文件批量转换为纯 markdown，输出到 reports_markdown_path。
        :param dir_path: PDF 所在目录，默认为配置中的 pdf_reports_dir
        """
        if dir_path is None:
            dir_path = self.paths.pdf_reports_dir
        if not dir_path.exists():
            print(f"目录不存在: {dir_path}")
            return
        pdf_files = [f.name for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
        if not pdf_files:
            print(f"目录下未找到 PDF 文件: {dir_path}")
            return
        print(f"共找到 {len(pdf_files)} 个 PDF，开始批量转换为 markdown ...")
        for file_name in pdf_files:
            try:
                self.export_reports_to_markdown(file_name)
            except Exception as e:
                print(f"处理 {file_name} 时出错: {e}")
        print("批量转换完成")

# 文本分块
    def chunk_reports(self):
        """
        将规整后 markdown 报告分块，便于后续向量化和检索
        """
        # 文档切分
        text_splitter = TextSplitter()
        # 只处理 markdown 文件，输入目录为 reports_markdown_path，输出目录为 documents_dir
        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的 markdown 文件...")
        # 自动传入 subset.csv 路径，便于补充 company_name 字段
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.reports_markdown_path,
            output_dir=self.paths.documents_dir,
            subset_csv=self.paths.subset_path
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

# 创建数据库
    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Faiss向量数据库保存在： {output_dir}目录下")

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库
        """
        print("开始处理报告流程...")
        
        print("步骤1：报告分块...")
        self.chunk_reports()
        
        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()
        
        print("报告处理流程已成功完成！")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀，如不存在直接返回。
        例如：若answers.json已存在，则返回answers_01.json等。
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        # 处理所有问题，生成答案文件
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def answer_single_question(self, question: str, kind: str = "string", history: Optional[List[Dict]] = None):
        """
        单条问题即时推理，支持多轮对话。
        返回 (answer, new_history)。history 为短期对话上下文，长度超过 8 条时自动丢弃最早的消息。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        history = history if history is not None else []
        t0 = time.time()
        print("[计时] 开始初始化 QuestionsProcessor ...")
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,  # 单问无需文件
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=1,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )
        t1 = time.time()
        print(f"[计时] QuestionsProcessor 初始化耗时: {t1-t0:.2f} 秒")
        print("[计时] 开始调用 process_single_question ...")
        answer, new_history = processor.process_single_question(question, kind=kind, history=history)
        t2 = time.time()
        print(f"[计时] process_single_question 推理耗时: {t2-t1:.2f} 秒")
        print(f"[计时] answer_single_question 总耗时: {t2-t0:.2f} 秒")
        return answer, new_history

# preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
#                       "no_ser_tab": RunConfig(use_serialized_tables=False)}

# 预设了一些配置，可以用于不同的场景
base_config = RunConfig(
    parallel_requests=10,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_base"
)

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=20,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_pdr"
)

## 这里是最佳实践
max_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=4,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = qwen-turbo",
    answering_model="qwen-turbo",
    config_suffix="_qwen_plus"
)

configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config}


# 修改 run_config 以尝试不同的配置
if __name__ == "__main__":
    # 设置数据集根目录（此处以 stock_data 为例）
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    # print(type(root_path))
    # 初始化主流程，使用推荐的最佳配置 
    pipeline = Pipeline(root_path, run_config=max_config)
    
    # print('1. 将目录下 PDF 批量转化为纯 markdown 文本')
    # pipeline.export_reports_to_markdown_batch() 

    # # 2. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    # print('2. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    # pipeline.chunk_reports() 
    
    # # 3. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    # print('3. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    # pipeline.create_vector_dbs()     
    
    # 4. 处理问题并生成答案，具体逻辑取决于 run_config
    # 默认questions.json
    print('4. 处理问题并生成答案，具体逻辑取决于 run_config')
    # pipeline.process_questions()
    # 单轮：answer, history = pipeline.answer_single_question(question="...", kind="string")
    # 多轮：先 history = []，每次 answer, history = pipeline.answer_single_question(question="...", kind="string", history=history)
    history = []
    # answer, history = pipeline.answer_single_question(question="半导体行业有哪些关键特性，这些特性如何助力中芯国际发展？", kind="string", history=history)
    # print("答案:", answer.get("final_answer") if isinstance(answer, dict) else answer)
    # answer, history = pipeline.answer_single_question(question="中芯国际的营收和利润情况近期有何变化？影响因素是什么？", kind="string", history=history)
    # print("答案:", answer.get("final_answer") if isinstance(answer, dict) else answer)
    # answer, history = pipeline.answer_single_question(question="我的第一个问题是什么？", kind="string", history=history)
    # print("答案:", answer.get("final_answer") if isinstance(answer, dict) else answer)

    print("欢迎使用RAG知识库助手，按 'q' 退出。")
    while True:
        question = input("用户: ")
        if question.lower() == "q":
            break
        answer, history = pipeline.answer_single_question(question, kind="string", history=history)
        print(f"助手:", answer.get('final_answer') if isinstance(answer, dict) else answer)
        print(history)
    print("完成")
