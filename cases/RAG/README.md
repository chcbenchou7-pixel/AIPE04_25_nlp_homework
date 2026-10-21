# 健康諮詢小幫手（RAG 網頁問答系統）

## 目標
透過 **RAG（Retrieval-Augmented Generation）** 建立一個健康諮詢小幫手：

1. 爬取 [健康醫療網](https://www.healthnews.com.tw/) 的「大腸癌」分類新聞。
2. 把新聞存進 SQLite（含流水號、標題、連結、全文、摘要、文章新增時間）。
3. 用**地端模型**（mtplx + Qwen3.5-9B）產生文章摘要。
4. 把摘要轉向量（mtplx + Qwen3-Embedding），存成 FAISS 向量索引。
5. 提供**網頁問答系統**，回答時依向量檢索帶入相關新聞，
   並在回覆後以條列方式附上**新聞標題、連結、文章新增時間**。
6. 有 checkbox 可決定**多輪對話時，後續的問題是否進行檢索**。
7. 「重點整理」按鈕：把剛才詢問過的問題整理成重點後回答。

## 架構
```text
            ┌───────────────────────────────┐
            │      mtplx serve（地端）      │
            │  ・mtplx-qwen35-9b（chat）    │  ← http://127.0.0.1:8000/v1
            │  ・qwen3-embed（embeddings）  │     OpenAI 相容 API
            └──────────────┬────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   crawl_data.py     make_index.py       app.py (Flask)
   爬新聞＋摘要      摘要→向量→FAISS     網頁問答（RAG）
        │                  │              query.py(檢索 demo)
        └──── news.db ─────┘
              (SQLite)
```

- 全程**地端運行**，不需要任何雲端 API。
- embedding 模型更換說明：原本想用 bge-m3（xlm-roberta），
  但 mtplx 的 retrieval runtime 只支援 MLX LM 的 decoder 型模型，不接受 encoder 型模型，
  所以改用 [mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ)
  （1024 維、多語系含繁中、last-token pooling）。

## 資料表（news.db）
| 欄位 | 說明 |
| ---- | ---- |
| id | 流水號（PRIMARY KEY） |
| link | 新聞連結 |
| reporter | 記者名稱 |
| date | 新聞發文日期 |
| title | 新聞標題 |
| content | 文章全文 |
| img_path | 新聞圖片 |
| summary | 文章摘要（地端模型產生） |
| created_at | 文章新增時間 |

- 共 **248** 筆大腸癌相關新聞（`crawl_data.py` 抓自健康醫療網的大腸癌分類頁）。
- 摘要產生方式：把**新聞標題 + 文章全文**合併後，
  透過**地端模型**（原本 Ollama gemma4:e4b，本專案改 mtplx Qwen3.5-9B）產生
  190~200 字的摘要（詳見 `crawl_data.py` 的 `summarize_text()`）。

## 建立向量索引（make_index.py）
- 讀取 `news.db` 的每篇新聞摘要。
- 呼叫 mtplx `/v1/embeddings` 把摘要轉成向量（每批 16 筆、normalize）。
- 建 `faiss.IndexFlatIP` + `IndexIDMap`（以「流水號」作為向量ID，搜尋可還原新聞）。
- 存到 `vector.index`。
- 實際執行過程：

```text
共有 248 篇文章摘要要轉成向量
embedding batch 1: 16 筆
...
embedding batch 16: 8 筆
向量維度: (248, 1024)
索引已儲存到 vector.index
```

## 語意檢索示範（query.py）
```text
相似度: [[0.6946, 0.6759, 0.6144]]
檢索的 Document IDs 為: [[102, 137, 5]]

使用者查詢的問題: 有人用 AI 來偵測大腸瘜肉嗎？
Document ID: 102   相似度: 0.6946
新聞標題: 國內首度「以AI偵測大腸瘜肉」準確率高達95%　有效降低大腸癌發生率
新聞連結: https://www.healthnews.com.tw/article/55249
文章新增時間: 2026-09-20 23:58:51
文件摘要: 臺大醫院、國泰綜合醫院與科技公司合作，開發出「大腸鏡即時AI瘜肉偵測」...
...
```

## 網頁系統（app.py + templates/index.html）
- 執行：`python app.py`，瀏覽器打 http://127.0.0.1:5060
- 功能：
  1. 輸入問題 → 向量檢索（top-5）→ 檢索內容加進 prompt → 地端模型回答。
  2. 回覆後方會以條列顯示**參考健康新聞**：新聞標題、連結、文章新增時間。
  3. checkbox：「多輪對話時，後續的問題也進行檢索」
     - 打勾：每次對話都會用前面問題＋這次的問題做擴張檢索。
     - 不打勾：第一個問題檢索一次，後續就不再檢索（純多輪對話）。
  4. 「重點整理」按鈕：把剛才問過的問題整理成重點後回答，
     也會同時附上前面所有參考過的新聞（去重後）。

### 實際測試（按作業要求詢問 2 個不同的問題）
問題一：「**有人用 AI 來偵測大腸瘜肉嗎？**」
回答（節錄）：
```text
是的，根據提供的健康新聞資料，目前台灣確實已經有使用人工智慧（AI）來
輔助偵測大腸瘜肉，並且取得了相當好的成效：
1. 準確率顯著提升（臺大醫院...高達 95%）
2. 解決漏診問題（26% 息肉難被發現...AI 輔助可增加約 50%）
...
━━ 參考健康新聞 ━━
1. 國內首度「以AI偵測大腸瘜肉」準確率高達95%　有效降低大腸癌發生率
   連結：https://www.healthnews.com.tw/article/55249
   文章新增時間：2026-09-20 23:58:51
2. 換水大腸鏡搭配AI輔助　增大腸息肉發現率
   連結：https://www.healthnews.com.tw/article/49424
   文章新增時間：2026-09-20 23:58:51
...
```

問題二：「**大腸癌篩檢應該多久做一次？**」
回答（節錄）：
```text
1. 篩檢頻率：每兩年一次
2. 適合篩檢的年齡層：50 歲至未滿 75 歲
3. 篩檢方式：糞便潛血檢查；陽性後 6 個月內做大腸鏡
...
━━ 參考健康新聞 ━━
（附上新聞標題 / 連結 / 文章新增時間的清單）
```

按「重點整理」的回覆（節錄）：
```text
問題一：有人用 AI 來偵測大腸瘜肉嗎？
核心結論：AI 技術已應用於大腸內視鏡檢查，能協助醫師更精準地偵測...

問題二：大腸癌篩檢應該多久做一次？
核心結論：建議定期篩檢（糞便潛血 2 年一次）...

━━ 重點整理時依據的參考健康新聞 ━━
1. 國內首度「以AI偵測大腸瘜肉」...
   連結：...
   文章新增時間：...
```

## 安裝套件（python 3.12 / pip list 節錄）
- openai `3.35.1`（透過 OpenAI 相容 API 打 mtplx）
- faiss-cpu `1.15.1`
- Flask `3.1.3`
- python-dotenv `1.2.4`
- numpy `2.5.2`
- beautifulsoup4 `4.15.0`、requests `2.34.2`、lxml `6.1.3`（爬蟲）

## 環境變數（.env）
```bash
MTPLX_BASE_URL=http://127.0.0.1:8000/v1
MTPLX_API_KEY=mtplx-local
MTPLX_MODEL=mtplx-qwen35-9b
MTPLX_EMBEDDING_MODEL=qwen3-embed
```

## 執行方法
```bash
# 0. 啟動地端 mtplx（chat + embeddings）
mtplx serve \
  --model /Users/chcben/.mtplx/models/Qwen3.5-9B-MTPLX-Optimized-Speed \
  --model-id mtplx-qwen35-9b \
  --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ=qwen3-embed \
  --profile turbo --reasoning off --port 8000 --no-stats-footer

# 1. （可以跳過）爬新聞＋用地端模型產生摘要 → news.db 已建好
python crawl_data.py

# 2. 把摘要轉向量、建 FAISS 索引
python make_index.py

# 3. 檢索示範（terminal 版）
python query.py

# 4. 網頁問答系統
python app.py      # http://127.0.0.1:5060
```

## 檔案說明
| 檔案 | 說明 |
| ---- | ---- |
| crawl_data.py | 爬健康醫療網大腸癌分類新聞，並用地端模型產生文章摘要 |
| init_db.py | 建立 news.db 資料表與 insert |
| make_index.py | 摘要 → 向量（mtplx embeddings）→ FAISS 索引 |
| query.py | terminal 版檢索示範 |
| app.py | Flask 網頁問答系統（RAG） |
| templates/index.html | 網頁前端（checkbox、重點整理按鈕、來源列表） |
| news.db | 已有的 248 筆新聞資料（SQLite） |
| vector.index | 已建好的 FAISS 向量索引 |
