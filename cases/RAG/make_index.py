# -*- coding: utf-8 -*-
"""
make_index.py

mtplx 版本：改用地端 mtplx 提供的 OpenAI 相容 embeddings API
（embedding model: bge-m3 = mlx-community/bge-m3-mlx-4bit），
將 news.db 的 [標題 + 文章摘要] 轉換成向量，存成 FAISS 向量索引。

前置作業：

    1. 啟動 mtplx 服務（見 repo 根目錄 README 或 .env）：

        mtplx serve --model /Users/chcben/.mtplx/models/Qwen3.5-9B-MTPLX-Optimized-Speed \
            --model-id mtplx-qwen35-9b \
            --embedding-model mlx-community/bge-m3-mlx-4bit=bge-m3 \
            --profile turbo --reasoning off --port 8000 --no-stats-footer

    2. 確認 .env 中有以下變數：

        MTPLX_BASE_URL=http://127.0.0.1:8000/v1
        MTPLX_API_KEY=mtplx-local
        MTPLX_EMBEDDING_MODEL=bge-m3

執行：

    python make_index.py

[faiss.IndexFlatIP]
IP = Inner Product
同時進行 feature normalization，等同於 cosine similarity
"""
import os
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

# mtplx API 設定（來自 .env）
MTPLX_BASE_URL = os.getenv("MTPLX_BASE_URL", "http://127.0.0.1:8000/v1")
MTPLX_API_KEY = os.getenv("MTPLX_API_KEY", "mtplx-local")
EMBEDDING_MODEL = os.getenv("MTPLX_EMBEDDING_MODEL", "bge-m3")

# 建立 OpenAI 相容客戶端（指向 mtplx）
client = OpenAI(
    base_url=MTPLX_BASE_URL,
    api_key=MTPLX_API_KEY,
)

# 索引存放路徑
BASE_DIR = Path(__file__).resolve().parent
index_path = BASE_DIR / "vector.index"

# 每次 API 最多處理的文字數
BATCH_SIZE = 16


def get_embeddings(texts, batch_size=BATCH_SIZE):
    """
    使用 mtplx 的 embeddings API 將文字轉成向量，
    並回傳 unit normalized 向量（cosine similarity 計算用）。
    """
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        vectors = [item.embedding for item in response.data]
        all_vectors.extend(vectors)
        print(f"embedding batch {i // batch_size + 1}: {len(batch)} 筆")

    embeddings = np.array(all_vectors, dtype="float32")

    # normalize，配合 IndexFlatIP 等同 cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1.0, norms)

    return embeddings


if __name__ == "__main__":

    # 讀取資料庫：文章摘要作為檢索文件（摘要已由 crawl_data.py 用地端模型產生）
    docs = []
    doc_ids = []
    conn = sqlite3.connect(BASE_DIR / "news.db")
    try:
        rows = conn.execute(
            "SELECT id, summary FROM news WHERE summary IS NOT NULL AND summary != ''"
        ).fetchall()
        for news_id, summary in rows:
            docs.append(summary)
            doc_ids.append(news_id)
    finally:
        conn.close()

    print(f"共有 {len(docs)} 篇文章摘要要轉成向量")

    # 將所有摘要轉換成向量
    embeddings = get_embeddings(docs)
    print("向量維度:", embeddings.shape)

    # 重新建立索引（每次都重建，確保 index 內容與 news.db 一致）
    dims = embeddings.shape[1]
    index = faiss.IndexFlatIP(dims)      # 內積索引
    index = faiss.IndexIDMap(index)      # 記錄對應 doc id

    # 加入向量和對應的 doc id
    doc_ids = np.array(doc_ids, dtype="int64")
    index.add_with_ids(embeddings, doc_ids)

    # 儲存索引
    faiss.write_index(index, str(index_path))
    print("索引已儲存到", index_path)
