# -*- coding: utf-8 -*-
"""
app.py

「健康諮詢小幫手」RAG 網頁問答系統（mtplx 版本）。

功能（對應作業要求）：

1. 地端 mtplx（OpenAI 相容 API）作為地端模型與 embedding 取得來源。
2. FAISS 向量索引檢索，對 news.db 中健康檢查位的內容進行語意檢索。
3. 整合地端模型回答，回答的回覆後面以條列方式呈現檢索到的新聞標題、
   新聞連結、文章新增時間。
4.CheckBox 決定「多輪對話時，是否要讓後續的對話進行檢索」。
5. 「重點整理」按鈕：將使用者剛才詢問過的問題整理成重點後回答，
   回答也附上檢索來源。

執行方式：

    1. 啟動 mtplx 服務（見根目錄 README）。
    2. .venv/bin/python app.py
    3. 開啟瀏覽器 http://127.0.0.1:5000
"""
import json
import os
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

# =============================================================================
# 1. mtplx API 設定（來自 .env）
# =============================================================================

MTPLX_BASE_URL = os.getenv("MTPLX_BASE_URL", "http://127.0.0.1:8000/v1")
MTPLX_API_KEY = os.getenv("MTPLX_API_KEY", "mtplx-local")
MTPLX_MODEL = os.getenv("MTPLX_MODEL", "mtplx-qwen35-9b")
EMBEDDING_MODEL = os.getenv("MTPLX_EMBEDDING_MODEL", "bge-m3")

client = OpenAI(
    base_url=MTPLX_BASE_URL,
    api_key=MTPLX_API_KEY,
    timeout=600,
)


# =============================================================================
# 2. 檔案路徑
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "news.db"

INDEX_PATH = BASE_DIR / "vector.index"

TOP_K = 5


# =============================================================================
# 3. Flask app
# =============================================================================

app = Flask(__name__)

# 對話歷史紀錄（每個 session id 一組）
chat_histories = {}


def get_history(session_id):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]


# =============================================================================
# 4. Embedding / 檢索 / 呼叫 mtplx
# =============================================================================

_index_cache = None


def load_index():
    """載入 FAISS 向量索引（只載入一次）。"""
    global _index_cache
    if _index_cache is None:
        _index_cache = faiss.read_index(str(INDEX_PATH))
    return _index_cache


def get_embedding(text):
    """
    使用 mtplx 的 embeddings API 將文字轉成向量（已 normalize）。
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    embedding = np.asarray(response.data[0].embedding, dtype="float32")
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def retrieve_docs(query, session_id):
    """
    根據查詢句做向量檢索：

    1. 查詢句轉向量。
    2. FAISS 檢索 top-k。
    3. 到 news.db 取出新聞標題、連結、摘要、文章新增時間。

    多輪對話檢索時，把 session 之前的問題加到查詢文字中。
    """
    history = get_history(session_id)

    # 多輪檢索：把前面的問題加進查詢文字，增加擴張檢索的語意
    retrieval_query = query
    if history:
        prev_questions = " ".join(
            item["question"] for item in history[-3:]
        )
        retrieval_query = f"{prev_questions} {query}"

    embedding = get_embedding(retrieval_query).reshape(1, -1)

    index = load_index()
    scores, ids = index.search(embedding, TOP_K)

    docs = []
    conn = sqlite3.connect(str(DB_PATH))
    try:
        for doc_id, score in zip(ids[0], scores[0]):
            if doc_id < 0:
                continue
            row = conn.execute(
                "SELECT id, title, link, date, summary, created_at"
                "  FROM news WHERE id = ?",
                (int(doc_id),),
            ).fetchone()
            if row is None:
                continue
            news_id, title, link, date, summary, created_at = row
            docs.append(
                {
                    "id": news_id,
                    "title": title,
                    "link": link,
                    "date": date,
                    "summary": summary,
                    "created_at": created_at,
                    "score": float(score),
                }
            )
    finally:
        conn.close()

    return docs


def format_sources(docs):
    """把檢索到的資料組成「來源列表」文字。"""
    if not docs:
        return "（沒有檢索到相關的健康新聞）"

    lines = []
    for i, doc in enumerate(docs, start=1):
        lines.append(
            f"{i}. {doc['title']}\n"
            f"   連結：{doc['link']}\n"
            f"   文章新增時間：{doc['created_at']}"
        )
    return "\n".join(lines)


def call_mtplx(messages):
    """
    呼叫地端 mtplx 的 chat API。
    """
    response = client.chat.completions.create(
        model=MTPLX_MODEL,
        messages=messages,
        temperature=0.7,
        top_p=0.9,
    )
    text = response.choices[0].message.content
    if text and "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip() if text else ""


SYSTEM_PROMPT = (
    "你是一個健康諮詢小幫手，請使用台灣繁體中文回答使用者的健康相關問題。"
    "回答時，請根據提供給你的健康新聞資料來說明，"
    "儘量提供淺顯易懂的重點，不確定的內容可以鼓勵使用者找醫生諮詢。"
)


# =============================================================================
# 5. Routes
# =============================================================================

@app.route("/")
def index():
    """首頁：健康諮詢小幫手的問答網頁。"""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    session_id = request.headers.get("X-Session-Id") or data.get("session_id", "")
    question = (data.get("question") or "").strip()
    use_retrieve = bool(data.get("retrieve", False))

    if not question:
        return jsonify({"error": "請輸入問題！"}), 400

    if not session_id:
        import uuid
        session_id = uuid.uuid4().hex

    history = get_history(session_id)

    # -------------------------------------------------------------
    # 檢索（含 checkbox 決定的多輪檢索行為）
    # -------------------------------------------------------------
    if use_retrieve or not history:
        docs = retrieve_docs(question, session_id)
    else:
        docs = []

    # -------------------------------------------------------------
    # 組 prompt
    # -------------------------------------------------------------
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 多輪對話時，提供一點過往對讓模型理解前後文
    if not use_retrieve:
        for item in history[-2:]:
            messages.append({"role": "user", "content": item["question"]})
            messages.append({"role": "assistant", "content": item["answer"]})

    user_content = question
    if docs:
        sources_text = "\n\n".join(
            f"【{d['title']}】{d['summary']}" for d in docs
        )
        user_content = (
            f"以下是相關的健康新聞資料：\n\n{sources_text}\n\n"
            f"請根據以上資料回答使用者的問題：\n{question}"
        )

    messages.append({"role": "user", "content": user_content})

    # -------------------------------------------------------------
    # 呼叫地端模型
    # -------------------------------------------------------------
    answer = call_mtplx(messages)

    # 回覆後面附上來源資訊
    sources_list = [
        {
            "title": d["title"],
            "link": d["link"],
            "created_at": d["created_at"],
        }
        for d in docs
    ]

    sources_text = format_sources(docs)
    full_answer = f"{answer}\n\n━━ 參考健康新聞 ━━\n{sources_text}"

    # 紀錄歷史
    history.append(
        {
            "question": question,
            "answer": answer,
            "full_answer": full_answer,
            "sources": sources_list,
            "retrieved": bool(docs),
            "doc_ids": [d["id"] for d in docs],
        }
    )

    return jsonify(
        {
            "session_id": session_id,
            "answer": answer,
            "sources": sources_list,
            "full_answer": full_answer,
            "retrieved": bool(docs),
        }
    )


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True)
    session_id = request.headers.get("X-Session-Id") or data.get("session_id", "")

    history = get_history(session_id)

    if not history:
        return jsonify({"error": "請先詢問一些問題後再按重點整理。"}), 400

    # -------------------------------------------------------------
    # 把剛才詢問過的問題，整理成重點
    # -------------------------------------------------------------
    questions = [item["question"] for i, item in enumerate(history, 1)]
    qa_text = "\n".join(f"問題 {i}：{q}" for i, q in enumerate(questions, 1))

    # 收集前面所有回答中參考的來源（去重）
    all_sources = {}
    for item in history:
        for doc in item["sources"]:
            all_sources[doc["link"]] = doc

    sources_text = format_sources(
        [
            {
                "title": doc["title"],
                "link": doc["link"],
                "score": 0,
                "id": 0,
                "date": "",
                "summary": "",
                "created_at": doc["created_at"],
            }
            for doc in all_sources.values()
        ]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一個健康諮詢小幫手，請使用台灣繁體中文回答。"
                "請把使用者詢問過的問題與回答，整理成清楚的重點，"
                "重點整理要包含每個問題的核心結論，並條列呈現。"
                "不要使用 markdown 語法。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"我剛才詢問過以下的健康問題：\n\n{qa_text}\n\n"
                f"請幫我把以上的問題整理成重點，作為一個完整的重點總結。"
            ),
        },
    ]

    summary = call_mtplx(messages)

    full_answer = (
        f"{summary}\n\n━━ 重點整理時依據的參考健康新聞 ━━\n{sources_text}"
    )

    history.append(
        {
            "question": "（重點整理）",
            "answer": summary,
            "full_answer": full_answer,
            "sources": list(all_sources.values()),
            "retrieved": False,
            "doc_ids": [],
        }
    )

    return jsonify(
        {
            "session_id": session_id,
            "answer": summary,
            "sources": list(all_sources.values()),
            "full_answer": full_answer,
        }
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5060,
        debug=False,
        threaded=True,
    )
