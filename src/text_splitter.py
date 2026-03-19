import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os

# 文本分块工具类，支持按页分块、表格插入、token统计等
# 使用langchain.text_splitter.RecursiveCharacterTextSplitter进行分块
class TextSplitter():
    def split_markdown_file(self, md_path: Path, chunk_size: int = 30, chunk_overlap: int = 5):
        """
        按行分割 markdown 文件，每个分块记录起止行号和内容。
        :param md_path: markdown 文件路径
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :return: 分块列表
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        chunks = []
        i = 0
        total_lines = len(lines)
        while i < total_lines:
            start = i
            end = min(i + chunk_size, total_lines)
            chunk_text = ''.join(lines[start:end])
            chunks.append({
                'lines': [start + 1, end],  # 保留行号信息
                'text': chunk_text
            })
            i += chunk_size - chunk_overlap
        return chunks

    def split_markdown_file_langchain(self, md_path: Path, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        使用langchain.text_splitter.RecursiveCharacterTextSplitter进行分块
        :param md_path: markdown 文件路径
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :return: 分块列表
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,           # 目标块大小为500个字符
                        chunk_overlap=chunk_overlap,         # 块之间有50个字符的重叠
                        # 自定义中文友好的分隔符
                        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "] 
                        )
        chunks = text_splitter.split_text(text)

        # 反推行号
        result_chunks = []
        current_pos = 0

        for chunk in chunks:
            # 计算当前块在原始文本中的起始行号
            start_line = text.count('\n', 0, current_pos) + 1
            
            # 计算当前块在原始文本中的结束行号
            end_pos = current_pos + len(chunk)
            end_line = text.count('\n', 0, end_pos) + 1

            # 处理块内容跨越多行的情况，确保行号准确
            # 注意：这里是一个简化的计算，对于复杂的换行符情况可能需要更精细的处理
            result_chunks.append({
                'lines': [start_line, end_line],
                'text': chunk
            })

            # 更新当前位置
            current_pos = end_pos
        return result_chunks

    def split_markdown_reports(self, all_md_dir: Path, output_dir: Path, chunk_size: int = 30, chunk_overlap: int = 5, subset_csv: Path = None):
        """
        批量处理目录下所有 markdown 文件，分块并输出为 json 文件到目标目录。
        :param all_md_dir: 存放 .md 文件的目录
        :param output_dir: 输出 .json 文件的目录
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :param subset_csv: subset.csv 路径，用于建立 file_name 到 company_name 的映射
        """
        # 建立 file_name（去扩展名）到 company_name 的映射
        file2company = {}
        file2sha1 = {}
        if subset_csv is not None and os.path.exists(subset_csv):
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            # 自动识别主键列
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2company[file_no_ext] = row['company_name']
                    if 'sha1' in row:
                        file2sha1[file_no_ext] = row['sha1']
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = str(row['sha1'])
                    file2company[file_no_ext] = row['company_name']
                    file2sha1[file_no_ext] = row['sha1']
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')
        
        all_md_paths = list(all_md_dir.glob("*.md"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for md_path in all_md_paths:
            # chunks = self.split_markdown_file(md_path, chunk_size, chunk_overlap)
            chunks = self.split_markdown_file_langchain(md_path, chunk_size, chunk_overlap)
            output_json_path = output_dir / (md_path.stem + ".json")
            # 查找 company_name 和 sha1
            file_no_ext = md_path.stem
            company_name = file2company.get(file_no_ext, "")
            sha1 = file2sha1.get(file_no_ext, "")
            # metainfo 只保留 sha1、company_name、file_name 字段
            metainfo = {"sha1": sha1, "company_name": company_name, "file_name": md_path.name}
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({"metainfo": metainfo, "content": {"chunks": chunks}}, f, ensure_ascii=False, indent=2)
            print(f"已处理: {md_path.name} -> {output_json_path.name}")
        print(f"共分割 {len(all_md_paths)} 个 markdown 文件")


if __name__ == "__main__":
    text_splitter = TextSplitter()
    # chunks = text_splitter.split_markdown_file_langchain(Path("data/stock_data/debug_data/03_reports_markdown/【财报】中芯国际：中芯国际2024年年度报告.md"), 500, 50)
    # print(len(chunks))
    text_splitter.split_markdown_reports(Path("data/stock_data/debug_data/03_reports_markdown"), Path("data/stock_data/databases/chunked_reports"), 500, 50, Path("data/stock_data/subset.csv"))