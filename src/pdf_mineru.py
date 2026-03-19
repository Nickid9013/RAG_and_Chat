import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MINERU_API_KEY")

# mineru 下载的 zip 保存目录名；解压结果保存目录名
ZIPS_DIR_NAME = "zips"
UZIPS_DIR_NAME = "uzips"

def get_task_id(file_name):
    url='https://mineru.net/api/v4/extract/task'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}"
    }
    pdf_url = 'https://rag-documents.oss-cn-beijing.aliyuncs.com/pdf_reports/' + file_name # 文件的url地址，这里使用的是阿里云服务存储】、
    data = {
        'url':pdf_url,
        'is_ocr':True,
        'enable_formula': False,
    }

    res = requests.post(url,headers=header,json=data)
    # print(res.status_code)
    # print(res.json())
    # print(res.json()["data"])
    task_id = res.json()["data"]['task_id']
    return task_id

def get_result(task_id: str, base_path: Optional[os.PathLike] = None) -> Optional[Path]:
    """
    轮询任务状态，完成后将 zip 保存到 base_path/zips/，解压到 base_path/uzips/{task_id}/。
    :param task_id: mineru 任务 ID
    :param base_path: 根路径，zip 与解压目录在其下的 zips、uzips；None 表示当前工作目录
    :return: 解压后的目录路径 (base_path/uzips/task_id)，失败或未完成返回 None
    """
    base = Path(base_path) if base_path else Path.cwd()
    zip_dir = base / ZIPS_DIR_NAME
    uzip_dir = base / UZIPS_DIR_NAME
    zip_dir.mkdir(parents=True, exist_ok=True)
    uzip_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://mineru.net/api/v4/extract/task/{task_id}"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    while True:
        res = requests.get(url, headers=header)
        result = res.json()["data"]
        print(result)
        state = result.get("state")
        err_msg = result.get("err_msg", "")
        if state in ["pending", "running"]:
            print("任务未完成，等待5秒后重试...")
            time.sleep(5)
            continue
        if err_msg:
            print(f"任务出错: {err_msg}")
            return None
        if state == "done":
            full_zip_url = result.get("full_zip_url")
            if full_zip_url:
                local_zip_path = zip_dir / f"{task_id}.zip"
                print(f"开始下载: {full_zip_url}")
                r = requests.get(full_zip_url, stream=True)
                with open(local_zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"下载完成，已保存到: {local_zip_path}")
                extract_dir = uzip_dir / task_id
                unzip_file(local_zip_path, extract_dir=extract_dir)
                return extract_dir
            print("未找到 full_zip_url，无法下载。")
            return None
        print(f"未知状态: {state}")
        return None


def unzip_file(zip_path: os.PathLike, extract_dir: Optional[os.PathLike] = None) -> None:
    """
    解压指定的 zip 文件到目标文件夹。
    :param zip_path: zip 文件路径（通常位于 zips 目录下）
    :param extract_dir: 解压目标目录（通常为 uzips/{task_id}）；None 时使用 zip 同目录下的同名文件夹
    """
    zip_path = Path(zip_path)
    if extract_dir is None:
        extract_dir = zip_path.parent / zip_path.stem
    else:
        extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")

if __name__ == "__main__":
    file_name = "【财报】中芯国际：中芯国际2024年年度报告.pdf"
    task_id = get_task_id(file_name)
    print("task_id:", task_id)
    # 使用当前目录下的 zips / uzips；可传入 base_path=Path(".") 或项目根路径
    extract_dir = get_result(task_id)
    print("解压目录:", extract_dir)
