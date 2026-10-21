# NLP 自然語言處理作業（AIPE04）

三個作業皆在 **Apple Silicon（Apple M5）本機**完成，全程地端運行，未使用雲端 API。

| # | 作業 | 目錄 | 主要成果 |
| - | ---- | ---- | ---- |
| 一 | 微調 BERT，建立多元分類模型 | [cases/bert_finetune](cases/bert_finetune/README.md) | 中文 8 類情緒分類，驗證集 macro F1 **0.8637** |
| 二 | 微調語言模型，建立多輪對話的「土味情話」專家 | [cases/llm_finetune_mlx](cases/llm_finetune_mlx/README.md) | Qwen3.5-4B + MLX LoRA（300 iters），多輪推論 |
| 三 | 透過 RAG 建立健康諮詢小幫手 | [cases/RAG](cases/RAG/README.md) | mtplx（地端 OpenAI 相容 API）+ FAISS + Flask 網頁問答 |

## 環境
- macOS（Apple M5, arm64）、Python 3.12（`.venv` 虛擬環境）
- 作業一：PyTorch **MPS** 後端（FP32）
- 作業二：MLX（mlx 0.32.4 / mlx-lm 0.32.2，需 README 內說明的 `qwen3_5_text` remapping 補丁）
- 作業三：mtplx 2.11.3 提供地端 chat（Qwen3.5-9B）與 embeddings（Qwen3-Embedding-0.6B），透過 OpenAI 相容 API 呼叫

## 安裝
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 執行方法
```bash
# ── 作業一：BERT 8 類情緒分類 ──
cd cases/bert_finetune
python prepare_emotion_data.py     # 下載資料集、stratified split
python train_bert.py               # fine-tuning（MPS，約 4 分鐘）
python infer_bert.py               # 對指定 5 句預測
python plot_bert_metrics.py        # 畫 loss / macro F1 曲線

# ── 作業二：土味情話專家（多輪對話）──
cd cases/llm_finetune_mlx
python persona_seed.py             # 產生 55 組多輪語料
python prepare_data.py             # 切 train/valid
python -m mlx_lm.lora --model models/Qwen3.5-4B-4bit-text-mlx --train \
  --data data --adapter-path outputs/adapters --num-layers 16 \
  --batch-size 1 --iters 300 --learning-rate 1e-5 --mask-prompt \
  --max-seq-length 1024 --grad-checkpoint --steps-per-eval 50 --save-every 100
python infer_chat.py demo          # 多輪推論

# ── 作業三：健康諮詢小幫手（RAG）──
mtplx serve --model ~/.mtplx/models/Qwen3.5-9B-MTPLX-Optimized-Speed \
  --model-id mtplx-qwen35-9b \
  --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ=qwen3-embed \
  --profile turbo --reasoning off --port 8000 --no-stats-footer
cd cases/RAG
python make_index.py               # 摘要 → 向量 → FAISS 索引
python query.py                    # 檢索示範
python app.py                      # 網頁問答（http://127.0.0.1:5060）
```

## 作業要求之輸出
- 作業一 5 句預測結果（含 confidence 四位小數）見
  [cases/bert_finetune/README.md](cases/bert_finetune/README.md#作業要求的-5-個-texts-預測結果實際執行輸出)
- 作業二多輪對話示範見
  [cases/llm_finetune_mlx/README.md](cases/llm_finetune_mlx/README.md#示範輸出demo-模式實際執行結果)
- 作業三兩個測試問題與重點整理見
  [cases/RAG/README.md](cases/RAG/README.md#實際測試按作業要求詢問-2-個不同的問題)

## 大檔案說明
- `cases/bert_finetune/output/bert_emotion_clf/model.safetensors`（390MB）以 **Git LFS** 提交。
- 作業二基礎模型（2.1GB）不入庫，下載方式見其 README；
  LoRA 成果 `adapters.safetensors`（31MB）已提交。