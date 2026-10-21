# -*- coding: utf-8 -*-
"""
query.py

mtplx 版本：使用地端 mtplx 的 embeddings API（bge-m3），
將查詢句子轉成向量，對 FAISS 向量索引進行語意檢索，
並顯示檢索到的新聞標題、摘要與相似度。

依照要求，透過 Document ID 查詢對應的新聞標題與內容，
組成 user prompt（列出標題與文件摘要）。

執行前，先啟動 mtplx 服務（見 make_index.py 的前置作業）。
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

client = OpenAI(
    base_url=MTPLX_BASE_URL,
    api_key=MTPLX_API_KEY,
)

# 讀取索引
BASE_DIR = Path(__file__).resolve().parent
index_path = BASE_DIR / "vector.index"
index = faiss.read_index(str(index_path))


def get_embedding(text):
    """
    使用 mtplx 的 embeddings API 將文字轉成向量（normalized）。
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    embedding = np.asarray(response.data[0].embedding, dtype="float32")
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


if __name__ == "__main__":

    # 查詢句子
    list_query = ["有人用 AI 來偵測大腸瘜肉嗎？"]

    # 將查詢句子轉換成向量
    embeddings = np.stack([get_embedding(q) for q in list_query])

    # 查詢
    D, I = index.search(embeddings, k=3)

    # 顯示結果
    list_scores = D.tolist()
    list_ids = I.tolist()
    print(f"相似度: {list_scores}")
    print(f"檢索的 Document IDs 為: {list_ids}")

    # 透過 Document ID 查詢對應的新聞標題與內容
    conn = sqlite3.connect(BASE_DIR / "news.db")
    try:
        user_prompt = ""
        for query, ids, scores in zip(list_query, list_ids, list_scores):
            user_prompt = "=" * 80
            user_prompt += f"\n使用者查詢的問題: {query}"

            for doc_id, score in zip(ids, scores):
                if doc_id < 0:
                    continue

                user_prompt += f"\n{'-' * 80}"
                user_prompt += f"\nDocument ID: {doc_id}"
                user_prompt += f"\n相似度: {score:.4f}"

                row = conn.execute(
                    "SELECT title, content, summary, link, date, created_at"
                    "   FROM news WHERE id = ?",
                    (int(doc_id),),
                ).fetchone()

                title, content, summary, link, date, created_at = row
                user_prompt += f"\n新聞標題: {title}"
                user_prompt += f"\n新聞連結: {link}"
                user_prompt += f"\n文章新增時間: {created_at}"
                user_prompt += f"\n文件摘要: {summary}"

        print(user_prompt)
    finally:
        conn.close()
