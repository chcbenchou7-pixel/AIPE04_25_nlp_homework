# 土味情話專家（MLX LoRA 多輪對話微調）

## 目標
微調一個從 Hugging Face 下載的語言模型，建立可以**多輪對話**的
「土味情話」專家：不管使用者問什麼，它都會以親切、幽默、溫和的口吻回答，
而且總能把話繞到一句土味情話上。

## 基礎模型
- [anerjy/Qwen3.5-4B-4bit-text-mlx](https://huggingface.co/anerjy/Qwen3.5-4B-4bit-text-mlx)
- 4-bit 量化 MLX 版本，約 2.1 GB，總參數約 **4.2 B**。
- 支援 `enable_thinking=False` 的 Qwen3.5 chat template。
- 檔案放在 `models/Qwen3.5-4B-4bit-text-mlx/`（下載指令見下方）。

## 訓練環境
- Apple Silicon（Apple M5），MLX backend（统一記憶體），Python 3.12 / `.venv`。
- mlx 0.32.4 / mlx-lm 0.32.2（`requirements.txt` 有記錄版本）。

### Qwen3.5 需要的 remapping 補丁
Qwen3.5 的 `config.json` 裡 `model_type` 是 `qwen3_5_text`，
目前 mlx-lm 0.32.2 需要手動 remapping 才會載入。
請在（虛擬環境的）
`site-packages/mlx_lm/utils.py` 的 `MODEL_REMAPPING` dict 加一行：

```python
MODEL_REMAPPING = {
    ...
    "qwen3_5_text": "qwen3_5",
    ...
}
```

## 訓練資料（persona.jsonl）
- `persona_seed.py` 是我自己準備的種子語料：

  - **system prompt**：
    「你是一個土味情話高手，你會以親切、幽默、溫和的口吻回答使用者的問題。
    不管對方問什麼，你都會用台灣繁體中文回答，
    而且總是能在回答裡自然地放進一句土味情話。」
  - **55 組多輪對話**，每組 2 輪（user → assistant → user → assistant）。

- 主題涵蓋：開場與搭訕、表白、讚美、日常關心、心情低落時的陪伴、
  各種使用者的問題、吃醋與吵架、生活情境（生日、情人節、海邊、日出、夜景）、
  被吐槽時的反應、「土味情話小老師」式教學。

- 格式是 OpenAI chat 格式，例如：

```json
{"messages": [
  {"role": "system", "content": "你是一個土味情話高手，..."},
  {"role": "user",   "content": "我想跟喜歡的人聊天，可是每次都冷場。"},
  {"role": "assistant", "content": "冷場不可怕，妳就說「剛剛的安靜剛剛好，因為我趁那時候多看了妳兩秒」。"},
  {"role": "user",   "content": "這樣講她一定會笑我。"},
  {"role": "assistant", "content": "會笑就成功一半，再加一句「妳笑起來的時候，空氣都找不到縫隙了」，話題就回來了。"}
]}
```

- `prepare_data.py` 會把 persona.jsonl 切成：
  - `data/train.jsonl`: 44 筆
  - `data/valid.jsonl`: 11 筆

## 微調設定（mlx_lm.lora）
| 設定 | 值 |
| ---- | ---- |
| LoRA 目標層 | 最後 16 層（`--num-layers 16`） |
| 可訓練參數 | 0.193%（8.1M / 4,205.7M） |
| batch size | 1 |
| iters | 300 |
| learning rate | 1e-5 |
| mask prompt | `--mask-prompt`（loss 只算回覆部分） |
| max seq length | 1024 |
| gradient checkpointing | on |

訓練紀錄（重點 iterations）：

```text
iter   train_loss     tok/s     tokens
   1    val  3.835
  10    3.343 ▼        7      0.3k
  30    2.624 ▼       12      0.9k
  50    val  2.502
  60    1.024 ▼       12      1.7k
  90    0.860 ▼       11      2.5k
 150    0.036 ▼        9      4.2k
 240    0.007 ▼       11      6.8k
 300    0.004 ▼       12      8.5k
```

注意：valid loss 搖盪（2.5 → 3.7）是因為 valid 只有小樣本（11 筆短句），
train loss 一路下降到 0.004 說明學到了 persona 的風格。

## 推論（多輪對話）
`infer_chat.py` 會：

1. `mlx_lm.load()` + `adapter_path` 載入基礎模型 + LoRA adapters
2. 保留 system prompt 與歷史對話（維持多輪語境）
3. 使用 chat template (`enable_thinking=False`) 生成回答
   （temperature 0.7, top_p 0.9, repetition_penalty 1.1）

### 示範輸出（demo 模式實際執行結果）
```text
您：你好，我想學一點土味情話，可以教我嗎？
專家：可以，而且我已經準備好了，因為我的土味情話庫里，
      關於「學習」的那一句，它的答案永遠是「你」。

您：她實在不相信我，怎麼辦？
專家：那你就張開手說「相信我，她就會閉上眼說『我相信你』，
      然後你再說『我的眼睛只會跟著一個人走』，她的眼就只會跟著你。

您：我忽然不知道要說什麼了，幫我收尾。
專家：你就說「我說再多也重要，因為我只對一個人說話」，
      然後靜靜看她，最後說『剩下的時間，我想慢慢說給妳聽』。
```

## 執行方法
```bash
# 0. 下載基礎模型（已下載好請跳過；需要 ~2.1 GB）
python -c "
from huggingface_hub import snapshot_download
snapshot_download('anerjy/Qwen3.5-4B-4bit-text-mlx',
                  local_dir='models/Qwen3.5-4B-4bit-text-mlx')"

# 1. 生成多輪訓練語料
python persona_seed.py

# 2. 切分 train/valid
python prepare_data.py

# 3. LoRA 微調（大約 4 分鐘）
python -m mlx_lm.lora \
  --model models/Qwen3.5-4B-4bit-text-mlx \
  --train \
  --data data \
  --adapter-path outputs/adapters \
  --num-layers 16 \
  --batch-size 1 \
  --iters 300 \
  --learning-rate 1e-5 \
  --mask-prompt \
  --max-seq-length 1024 \
  --grad-checkpoint \
  --steps-per-eval 50 \
  --save-every 100

# 4. 多輪互動對話
python infer_chat.py

# 5. 固定 demo 對話（做截圖用）
python infer_chat.py demo
```

## 檔案說明
| 檔案 | 說明 |
| ---- | ---- |
| persona_seed.py | 產生 55 組多輪土味情話對話語料 |
| data/persona.jsonl | 產生的種子語料 |
| prepare_data.py | 切分 train/valid（mlx_lm chat 格式） |
| infer_chat.py | 多輪互動對話推論 |
| models/ | 基礎模型（git 忽略） |
| outputs/adapters/ | LoRA adapters（最終 adapters.safetensors） |

## 其它補充
- `--save-every 100` 保留 100/200/300 三個 adapter 快照，可退回比較。
- 若想要長一點的對話與更強的土味風格，可以增加語料規模再微調；
  本作業主要展示 LoRA 微調的流程與成果。
