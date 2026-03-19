import json
import logging
from typing import List, Tuple, Dict, Union
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker
import pandas as pd
import time

_log = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, embedding_provider: str = "dashscope"):
        # 初始化向量检索器，加载所有向量库和文档
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        # 最近一次检索的统计信息（用于调试/观测，不影响返回结构）
        self.last_retrieved_chunks: int = 0
        # 默认使用 dashscope 作为 embedding provider
        self.embedding_provider = embedding_provider.lower()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        # 根据 embedding_provider 初始化对应的 LLM 客户端
        load_dotenv()
        if self.embedding_provider == "openai":
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=None,
                max_retries=2
            )
            return llm
        elif self.embedding_provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return None  # dashscope 不需要 client 对象
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    def _get_embedding(self, text: str):
        # 根据 embedding_provider 获取文本的向量表示
        if self.embedding_provider == "openai":
            embedding = self.llm.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return embedding.data[0].embedding
        elif self.embedding_provider == "dashscope":
            import dashscope
            # DashScope 偶发会返回 None（例如鉴权失败/限流/网络异常），这里加健壮性检查与轻量重试
            rsp = None
            last_err = None
            for attempt in range(2):
                try:
                    rsp = dashscope.TextEmbedding.call(
                        model="text-embedding-v1",
                        input=[text]
                    )
                except Exception as e:
                    last_err = e
                    rsp = None
                if rsp is not None:
                    break
                # 短暂退避，避免瞬时限流
                time.sleep(0.5 * (attempt + 1))

            # 健壮性检查，防止 rsp 为 None 或非 dict
            if rsp is None:
                raise RuntimeError(f"DashScope embedding 返回 None（可能是 Key/限流/网络异常）。last_err={last_err}")
            if not isinstance(rsp, dict):
                raise RuntimeError(f"DashScope embedding 返回非 dict: type={type(rsp)} rsp={rsp}")
            # 兼容 dashscope 返回格式，不能用 resp.output，需用 resp['output']
            if 'output' in rsp and 'embeddings' in rsp['output']:
                # 多条输入（本处只有一条）
                emb = rsp['output']['embeddings'][0]
                if emb['embedding'] is None or len(emb['embedding']) == 0:
                    raise RuntimeError(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}")
                return emb['embedding']
            elif 'output' in rsp and 'embedding' in rsp['output']:
                # 兼容单条输入格式
                if rsp['output']['embedding'] is None or len(rsp['output']['embedding']) == 0:
                    raise RuntimeError("DashScope返回的embedding为空")
                return rsp['output']['embedding']
            else:
                raise RuntimeError(f"DashScope embedding API返回格式异常: {rsp}")
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    def _load_dbs(self):
        # 加载所有向量库和对应文档，建立映射
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        for document_path in all_documents_paths:
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"警告：加载json文件失败： {document_path.name}: {e}")
                continue
            # 用 metainfo['sha1'] 拼接 faiss 文件名
            sha1 = document.get('metainfo', {}).get('sha1', None)
            if not sha1:
                _log.warning(f"警告：没有sha1找到在metainfo for document {document_path.name}")
                continue
            faiss_path = self.vector_db_dir / f"{sha1}.faiss"
            if not faiss_path.exists():
                _log.warning(f"警告：没有匹配的向量库找到 for document {document_path.name} (sha1={sha1})")
                continue
            try:
                vector_db = faiss.read_index(str(faiss_path))
            except Exception as e:
                _log.error(f"警告：读取向量库失败 for {document_path.name}: {e}")
                continue
            report = {
                "name": sha1,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    def _get_reports_by_company(self, company_name: str) -> List[Dict]:
        """按公司名收集所有匹配的报告（文件收集模式）。"""
        target_reports = []
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo", {})
            if metainfo.get("company_name") == company_name or company_name in metainfo.get("file_name", ""):
                target_reports.append(report)
        return target_reports

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        # 文件收集模式：收集所有匹配公司名的报告，对每份做向量检索后合并，按相似度取全局 top_n
        target_reports = self._get_reports_by_company(company_name)
        if not target_reports:
            _log.error(f"没有找到报告 with '{company_name}' company name.")
            raise ValueError(f"没有找到报告 with '{company_name}' company name.")
        # 仅对 query 做一次 embedding，在所有报告中复用
        embedding = self._get_embedding(query)
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        # 归一化：IndexFlatIP + L2-normalize 等价于 cosine similarity
        faiss.normalize_L2(embedding_array)
        all_results = []
        total_retrieved_chunks = 0
        for report in target_reports:
            document = report["document"]
            vector_db: faiss.IndexFlatIP = report["vector_db"]
            metainfo = document.get("metainfo", {})
            file_name = metainfo.get("file_name", "")
            chunks = document["content"]["chunks"]
            per_file_k = min(top_n, len(chunks))  # 每份文件先取 top_n，再合并排序
            if per_file_k == 0:
                continue
            total_retrieved_chunks += per_file_k
            distances, indices = vector_db.search(x=embedding_array, k=per_file_k)
            for distance, index in zip(distances[0], indices[0]):
                distance = round(float(distance), 4)
                chunk = chunks[index]
                lines = chunk.get("lines")
                start_line = lines[0] if lines else 0
                all_results.append({
                    "distance": distance,
                    "page": start_line,
                    "text": chunk["text"],
                    "file_name": file_name,
                    "lines": lines,
                })
        self.last_retrieved_chunks = total_retrieved_chunks
        print(
            f"[VectorRetriever] company='{company_name}' 匹配报告数={len(target_reports)}，"
            f"候选chunk总数={total_retrieved_chunks}（每份<=top_n={top_n}），最终返回={min(top_n, len(all_results))}"
        )
        # 按相似度降序排序，取全局 top_n
        all_results.sort(key=lambda x: x["distance"], reverse=True)
        return all_results[:top_n]

    def retrieve_all(self, company_name: str) -> List[Dict]:
        # 文件收集模式：检索该公司名下的所有报告的全部文本块
        target_reports = self._get_reports_by_company(company_name)
        if not target_reports:
            _log.error(f"没有找到报告 with '{company_name}' company name.")
            raise ValueError(f"没有找到报告 with '{company_name}' company name.")
        all_pages = []
        for report in target_reports:
            document = report["document"]
            metainfo = document.get("metainfo", {})
            file_name = metainfo.get("file_name", "")
            chunks = document["content"].get("chunks", [])
            for chunk in chunks:
                lines = chunk.get("lines")
                start_line = lines[0] if lines else 0
                all_pages.append({
                    "distance": 0.5,
                    "page": start_line,
                    "text": chunk["text"],
                    "file_name": file_name,
                    "lines": lines,
                })
        return all_pages


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        使用混合检索方法进行检索和重排。
        
        参数：
            company_name: 需要检索的公司名称
            query: 检索查询语句
            llm_reranking_sample_size: 首轮向量检索返回的候选数量
            documents_batch_size: 每次送入LLM重排的文档数
            top_n: 最终返回的重排结果数量
            llm_weight: LLM分数权重（0-1）
            return_parent_pages: 是否返回完整页面（而非分块）
        
        返回：
            经过重排的文档字典列表，包含分数
        """
        t0 = time.time()
        # 首先用向量检索器获取初步结果
        print("[计时] [HybridRetriever] 开始向量检索 ...")
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        t1 = time.time()
        print(f"[计时] [HybridRetriever] 向量检索耗时: {t1-t0:.2f} 秒")
        # 使用LLM对结果进行重排
        print("[计时] [HybridRetriever] 开始LLM重排 ...")
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        t2 = time.time()
        print(f"[计时] [HybridRetriever] LLM重排耗时: {t2-t1:.2f} 秒")
        print(f"[计时] [HybridRetriever] 总耗时: {t2-t0:.2f} 秒")
        return reranked_results[:top_n]
