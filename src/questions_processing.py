import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor
from src import prompts
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures
import time


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,  # 是否启用父文档检索
        llm_reranking: bool = False,              # 是否启用LLM重排
        llm_reranking_sample_size: int = 5,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "dashscope",
        answering_model: str = "qwen-turbo",
        full_context: bool = False
    ):
        # 初始化问题处理器，配置检索、模型、并发等参数
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        
        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.llm_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        # 加载问题文件，返回问题列表
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """将检索结果格式化为RAG上下文字符串（文件名 + 行号范围）"""
        if not retrieval_results:
            return ""
        context_parts = []
        for result in retrieval_results:
            text = result["text"]
            file_name = result.get("file_name", "")
            lines = result.get("lines")
            if lines and len(lines) >= 2:
                line_desc = f"lines {lines[0]}-{lines[1]}"
            else:
                line_desc = f"page {result.get('page', 0)}"
            context_parts.append(f'Text retrieved from file "{file_name}", {line_desc}:\n"""\n{text}\n"""')
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, sources_list: list) -> list:
        """将校验后的来源列表转为引用列表，每项为 file_name + line_start + line_end"""
        refs = []
        for s in sources_list:
            if isinstance(s, dict) and "file_name" in s and "line_start" in s and "line_end" in s:
                refs.append({
                    "file_name": s["file_name"],
                    "line_start": s["line_start"],
                    "line_end": s["line_end"],
                })
        return refs

    def _validate_source_references(self, claimed_sources: list, retrieval_results: list, min_sources: int = 1, max_sources: int = 10) -> list:
        """
        校验 LLM 答案中引用的来源（文件名+行号）是否真实存在于检索结果中。
        若不足 min_sources 则从检索结果中补足，超过 max_sources 则截断。
        """
        if claimed_sources is None:
            claimed_sources = []
        # 从检索结果构建 (file_name, line_start, line_end) 集合
        def _source_key(r):
            lines = r.get("lines")
            if lines and len(lines) >= 2:
                return (r.get("file_name", ""), int(lines[0]), int(lines[1]))
            return None
        valid_keys = set()
        for r in retrieval_results:
            k = _source_key(r)
            if k:
                valid_keys.add(k)
        # 规范化 LLM 返回的项（可能是 dict 或 Pydantic model_dump）
        def _norm(s):
            if isinstance(s, dict):
                fn = s.get("file_name", "")
                ls = s.get("line_start", s.get("line_start"))
                le = s.get("line_end", s.get("line_end"))
                if ls is not None and le is not None:
                    return (fn, int(ls), int(le))
            return None
        validated = []
        for s in claimed_sources:
            k = _norm(s)
            if k and k in valid_keys:
                validated.append({"file_name": k[0], "line_start": k[1], "line_end": k[2]})
        if len(validated) < len(claimed_sources):
            print(f"Warning: Removed {len(claimed_sources) - len(validated)} hallucinated source references")
        if len(validated) < min_sources and retrieval_results:
            seen = set((x["file_name"], x["line_start"], x["line_end"]) for x in validated)
            for r in retrieval_results:
                k = _source_key(r)
                if not k or k in seen:
                    continue
                validated.append({"file_name": k[0], "line_start": k[1], "line_end": k[2]})
                seen.add(k)
                if len(validated) >= min_sources:
                    break
        if len(validated) > max_sources:
            print(f"Trimming references from {len(validated)} to {max_sources} sources")
            validated = validated[:max_sources]
        return validated

    # 针对单个公司，检索上下文并调用LLM生成答案；支持多轮 history，返回 (answer_dict, new_history)
    def get_answer_for_company(self, company_name: str, question: str, schema: str, history: Optional[List[Dict]] = None) -> tuple:
        # 针对单个公司，检索上下文并调用LLM生成答案
        t0 = time.time()
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        t1 = time.time()
        print(f"[计时] [get_answer_for_company] 检索器初始化耗时: {t1-t0:.2f} 秒")
        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
        else:           
            t2 = time.time()
            retrieval_results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages
            )
            t3 = time.time()
            print(f"[计时] [get_answer_for_company] 检索耗时: {t3-t2:.2f} 秒")
            
        if not retrieval_results:
            raise ValueError("未能找到相关上下文")
        t4 = time.time()
        # print(f"检索结果: {retrieval_results}")
        rag_context = self._format_retrieval_results(retrieval_results)
        # print(f"检索结果: {rag_context}")
        t5 = time.time()
        print(f"[计时] [get_answer_for_company] 构建rag_context耗时: {t5-t4:.2f} 秒")
        history = history if history is not None else []
        answer_dict, new_history = self.llm_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model,
            history=history,
        )
        print(answer_dict)
        t6 = time.time()
        print(f"[计时] [get_answer_for_company] LLM调用耗时: {t6-t5:.2f} 秒")
        self.response_data = self.llm_processor.response_data
        if self.new_challenge_pipeline:
            sources = answer_dict.get("relevant_sources", [])
            if not sources and "relevant_pages" in answer_dict:
                sources = []
            # 将 Pydantic 对象转为 dict 便于校验
            sources_list = [s.model_dump() if hasattr(s, "model_dump") else s for s in sources]
            validated_sources = self._validate_source_references(sources_list, retrieval_results)
            answer_dict["relevant_sources"] = [{"file_name": x["file_name"], "line_start": x["line_start"], "line_end": x["line_end"]} for x in validated_sources]
            answer_dict["references"] = self._extract_references(validated_sources)
        print(f"[计时] [get_answer_for_company] 总耗时: {t6-t0:.2f} 秒")
        return answer_dict, new_history

    # 从subset文件中提取公司名，并返回公司名列表
    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """从问题文本中提取公司名，匹配subset文件中的公司"""
        if not hasattr(self, 'companies_df'):
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use subset extraction")
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                self.companies_df = pd.read_csv(self.subset_path, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                self.companies_df = pd.read_csv(self.subset_path, encoding='gbk')
        
        found_companies = []
        # 从长到短进行排序，防止短的公司名被长公司名包含
        company_names = sorted(self.companies_df['company_name'].unique(), key=len, reverse=True)
        
        for company in company_names:
            # 只要公司名在问题文本中出现就算匹配（包含关系）
            if company in question_text:
                found_companies.append(company)
                question_text = question_text.replace(company, '')
        
        return found_companies

    def process_question(self, question: str, schema: str):
        # 处理单个问题，支持多公司比较
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            # UNUSED（当前项目主流程未使用）：
            # pipeline.py 中构造 QuestionsProcessor 时固定 new_challenge_pipeline=True，
            # 因此不会走到旧版“从英文引号里抽公司名”的分支。
            extracted_companies = re.findall(r'"([^"]*)"', question)
        
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict, _ = self.get_answer_for_company(company_name=company_name, question=question, schema=schema, history=[])
            return answer_dict
        else:
            return self.process_comparative_question(question, extracted_companies, schema)
    
    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """创建答案详情的引用ID，并存储详细内容"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_sources": answer_dict.get('relevant_sources', []),
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """统计处理结果，包括总数、错误数、N/A数、成功数"""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, pipeline_details: str = "") -> dict:
        # 批量处理问题列表，支持并行与断点保存，返回处理结果和统计信息
        total_questions = len(questions_list)
        # 给每个问题加索引，便于后续答案详情定位
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # 预分配答案详情列表
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            # 单线程顺序处理
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
        else:
            # 多线程并行处理
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map 保证结果顺序与输入一致
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        
        if self.new_challenge_pipeline:
            question_text = question_data.get("text")
            schema = question_data.get("kind")
        else:
            # UNUSED（当前项目主流程未使用）：
            # 当前数据集/流程使用 new_challenge_pipeline=True 的输入格式（text/kind）。
            # 旧版输入格式（question/schema）保留仅为兼容历史用法。
            question_text = question_data.get("question")
            schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(question_text, schema)
            
            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_sources": []
                }, question_index)
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
            else:
                return {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        处理问题处理过程中的异常。
        记录错误详情并返回包含错误信息的字典。
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            # UNUSED（当前项目主流程未使用）：
            # 当前流程的输出是新挑战提交格式（question_text/kind/value/references）。
            # 这里的旧版输出（question/schema/answer）保留仅为兼容历史用法。
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        提交格式后处理：
        1. 页码从1-based转为0-based
        2. N/A答案清空引用
        3. 格式化为比赛提交schema
        4. 包含step_by_step_analysis
        """
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass
            
            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # 引用格式为 file_name + line_start + line_end（行号保持 1-based 便于阅读）
                references = [
                    {
                        "file_name": ref.get("file_name", ""),
                        "line_start": ref.get("line_start"),
                        "line_end": ref.get("line_end"),
                    }
                    for ref in references
                    if isinstance(ref, dict) and ref.get("file_name") is not None
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        """
        处理多公司比较类问题：
        1. 先将比较问题重写为单公司问题
        2. 并行处理每个公司
        3. 汇总结果并生成最终比较答案
        """
        # 说明：
        # - 当前 pipeline.py 的问题处理主要面向“单公司问题”（subset 提取通常返回 1 个公司）。
        # - 若抽取到多个公司，会进入此函数：先拆解为单公司子问题 -> 分别检索回答 -> 再汇总比较答案。
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.llm_processor.get_rephrased_questions(
            original_question=question,
            companies=companies
        )
        
        individual_answers = {}
        aggregated_references = []
        
        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")
            
            answer_dict, _ = self.get_answer_for_company(
                company_name=company,
                question=sub_question,
                schema=schema,
                history=[],
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company 
                for company in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict
                    
                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise
        
        # Remove duplicate references (by file_name + line range)
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("file_name"), ref.get("line_start"), ref.get("line_end"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.llm_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.llm_processor.response_data
        
        comparative_answer["references"] = aggregated_references
        return comparative_answer

    def process_single_question(self, question: str, kind: str = "string", history: Optional[List[Dict]] = None):
        """
        单条问题推理，返回 (结构化答案, new_history)。
        kind: 支持 'string'、'number'、'boolean'、'names' 等。
        history: 多轮对话上下文，长度超过 8 条时自动丢弃最早的消息。
        """
        history = history if history is not None else []
        t0 = time.time()
        print("[计时] [单问] 开始公司名抽取 ...")
        # 公司名抽取方案：1. 从subset文件中抽取；2. 用正则表达式从问题文本中抽取
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)

        t1 = time.time()
        print(f"[计时] [单问] 公司名抽取耗时: {t1-t0:.2f} 秒")
        if len(extracted_companies) == 0:
            # 未匹配到公司：不经过检索，使用默认系统提示（注明非知识库），JSON 格式与正常流程一致
            print("[计时] [单问] 未匹配到公司，直接调用大模型回答（无文档上下文）...")
            t2 = time.time()
            answer_dict, new_history = self.llm_processor.get_answer_from_rag_context(
                question=question,
                rag_context="",
                schema=kind,
                model=self.answering_model,
                history=history,
                system_prompt_override=prompts.DEFAULT_SYSTEM_PROMPT,
            )
            self.response_data = self.llm_processor.response_data
            t3 = time.time()
            print(f"[计时] [单问] 直接LLM推理耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict, new_history
        elif len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            print("[计时] [单问] 开始检索与LLM推理 ...")
            t2 = time.time()
            answer_dict, new_history = self.get_answer_for_company(company_name=company_name, question=question, schema=kind, history=history)
            t3 = time.time()
            print(f"[计时] [单问] 检索+LLM推理耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict, new_history
        elif len(extracted_companies) > 1:
            print("[计时] [单问] 开始多公司比较 ...")
            t2 = time.time()
            answer_dict = self.process_comparative_question(question, extracted_companies, kind)
            t3 = time.time()
            print(f"[计时] [单问] 多公司比较耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict, history
    