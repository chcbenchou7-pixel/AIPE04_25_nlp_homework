"""
prepare_emotion_data.py

從 Hugging Face 下載多元情緒對話資料集，並切成 BERT 多元分類所需的
train / validation JSONL。

作業說明：

    微調 BERT，建立多元分類模型
    https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset

資料內容：

    text     : 中文對話句子
    emotion  : 8 種情緒標籤（文字）

標籤（文字）轉整數的對應：

    0 平淡語氣
    1 開心語調
    2 關切語調
    3 憤怒語調
    4 驚奇語調
    5 悲傷語調
    6 厭惡語調
    7 疑問語調

輸出：

    data/bert_emotion/train.jsonl
    data/bert_emotion/valid.jsonl

每一行的格式：

    {"sentences": "我媽終於要讓我養狗了！", "labels": 1}
"""

import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from datasets import load_dataset


# 順序固定的標籤清單，確保每次執行都得到相同的整數標籤對應。
EMOTION_LABELS = [
    "平淡語氣",
    "開心語調",
    "關切語調",
    "憤怒語調",
    "驚奇語調",
    "悲傷語調",
    "厭惡語調",
    "疑問語調",
]

LABEL2ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

# Hugging Face 資料集 ID。
DATASET_ID = "Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset"

# 輸出位置。
OUTPUT_DIR = Path("data/bert_emotion")

VALID_RATIO = 0.2

SEED = 42


def main():

    print(f"下載資料集：{DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")
    print(dataset)

    texts = []
    labels = []

    # 把文字標籤轉成整數，並檢查是否有未知的標籤。
    for row in dataset:
        emotion = row["emotion"].strip()
        if emotion not in LABEL2ID:
            raise ValueError(f"未知標籤：{emotion!r}（合法標籤：{EMOTION_LABELS}）")

        text = row["text"].strip()
        if not text:
            continue

        texts.append(text)
        labels.append(LABEL2ID[emotion])

    if not texts:
        raise RuntimeError("資料集沒有讀到任何有效資料。")

    print("=" * 72)
    print(f"全部資料：{len(texts)} 筆")
    for label, count in Counter(labels).most_common():
        print(f"  label={label} {EMOTION_LABELS[label]}: {count} 筆 ({count/len(labels):.2%})")

    # 以整數標籤做 stratified split。
    (
        train_texts,
        valid_texts,
        train_labels,
        valid_labels,
    ) = train_test_split(
        texts,
        labels,
        test_size=VALID_RATIO,
        random_state=SEED,
        stratify=labels,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save_jsonl(subset_texts, subset_labels, path: Path):
        with path.open("w", encoding="utf-8") as file:
            for text, label in zip(subset_texts, subset_labels):
                row = {"sentences": text, "labels": label}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}：{len(subset_labels)} 筆")

    print("=" * 72)
    save_jsonl(train_texts, train_labels, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid_texts, valid_labels, OUTPUT_DIR / "valid.jsonl")

    print("=" * 72)
    print("Training distribution：")
    for label, count in Counter(train_labels).most_common():
        print(f"  {EMOTION_LABELS[label]}: {count} 筆 ({count/len(train_labels):.2%})")

    print("Validation distribution：")
    for label, count in Counter(valid_labels).most_common():
        print(f"  {EMOTION_LABELS[label]}: {count} 筆 ({count/len(valid_labels):.2%})")


if __name__ == "__main__":
    main()
