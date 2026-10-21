"""
infer_bert.py

載入 train_bert.py 儲存的最佳 BERT 多元情緒分類模型，
對作業指定的繁體中文句子進行 8 類情緒分類，並輸出：

    文字 => 情緒標籤 (信心分數，取小數點後 4 位)

使用 Apple MPS（本機）或 CPU 進行推論。

作業說明：

    微調 BERT，建立多元分類模型
    資料集：Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset
"""

from pathlib import Path
from pprint import pprint

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


# =============================================================================
# 1. 推論設定
# =============================================================================

MODEL_DIR = Path(
    "output/bert_emotion_clf"
)

# 作業指定的預測文字。
TEST_TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。",
]


# =============================================================================
# 2. 主流程
# =============================================================================

def main():

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 fine-tuned model：{MODEL_DIR.resolve()}\n"
            "請先執行 train_bert.py。"
        )

    # -------------------------------------------------------------------------
    # Step 1：載入 tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    # -------------------------------------------------------------------------
    # Step 2：載入 sequence classifier
    # -------------------------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_DIR
        )
    )

    # -------------------------------------------------------------------------
    # Step 3：決定 inference device
    # -------------------------------------------------------------------------

    mps_available = torch.backends.mps.is_available()

    device = (
        0
        if torch.cuda.is_available()
        else ("mps" if mps_available else -1)
    )

    print(
        "Inference device："
        + (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else ("MPS" if mps_available else "CPU")
        )
    )

    # -------------------------------------------------------------------------
    # Step 4：建立 Hugging Face pipeline
    # -------------------------------------------------------------------------

    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=False,
    )

    # -------------------------------------------------------------------------
    # Step 5：執行分類
    #
    # 輸出格式：
    #     文字 => 情緒標籤 (信心分數，小數點後 4 位)
    # -------------------------------------------------------------------------

    results = classifier(
        TEST_TEXTS,
        truncation=True,
        max_length=512,
    )

    for text, result in zip(
        TEST_TEXTS,
        results,
    ):
        label = result["label"]
        score = result["score"]

        print(f"{text} => {label} ({score:.4f})")

    print("=" * 72)

    print("Raw pipeline output：")
    pprint(results)


if __name__ == "__main__":
    main()
