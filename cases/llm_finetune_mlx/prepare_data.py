# -*- coding: utf-8 -*-
"""
prepare_data.py

把 persona.jsonl（OpenAI chat 格式）轉成 mlx_lm.lora 訓練需要的
multi-turn chat 格式資料集，並切成 train / valid 兩份。

作業說明：微調從 Hugging Face 下載的語言模型，
建立可以多輪對話的「土味情話」專家。

persona.jsonl 每一行：

    {"messages": [{"role": ..., "content": ...}, ...]}

輸出：

    data/train.jsonl
    data/valid.jsonl

mlx_lm 0.31 的 ChatDataset 會直接取 "messages" 欄位，
用模型自帶的 chat template 套用成訓練文字。

使用方式：

    python prepare_data.py           # 80/20 切分
    python prepare_data.py 0.1      # 第一個參數 = valid 比例
"""

import json
import random
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split


SOURCE_FILE = Path(__file__).resolve().parent / "data" / "persona.jsonl"

OUTPUT_DIR = Path(__file__).resolve().parent / "data"

VALID_RATIO = 0.2

SEED = 42


def main():

    valid_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else VALID_RATIO

    rows = []
    with SOURCE_FILE.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            # 基本檢查
            if "messages" not in row:
                raise ValueError(f"第 {line_num} 行缺少 messages 欄位")

            messages = row["messages"]
            if not (isinstance(messages, list) and len(messages) >= 3):
                raise ValueError(
                    f"第 {line_num} 行：messages 需至少含 system + 一輪對話（{len(messages)} 個訊息）"
                )

            if messages[0].get("role") != "system":
                raise ValueError(f"第 {line_num} 行：第一個訊息必須是 system")

            rows.append(messages)

    if not rows:
        raise RuntimeError(f"{SOURCE_FILE} 是空的")

    n_valid = max(1, int(len(rows) * valid_ratio))
    train_rows, valid_rows = train_test_split(
        rows,
        test_size=n_valid,
        random_state=SEED,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "train.jsonl").open("w", encoding="utf-8") as f:
        for msgs in train_rows:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    with (OUTPUT_DIR / "valid.jsonl").open("w", encoding="utf-8") as f:
        for msgs in valid_rows:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    print(f"來源      : {SOURCE_FILE}（{len(rows)} 筆）")
    print(f"train.jsonl: {len(train_rows)} 筆")
    print(f"valid.jsonl: {len(valid_rows)} 筆")
    print(f"輸出目錄  : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
