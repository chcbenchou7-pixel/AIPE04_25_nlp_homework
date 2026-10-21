"""
train_bert.py

使用 google-bert/bert-base-chinese 做繁體中文多元情緒分類 Fine-tuning。

作業說明：

    微調 BERT，建立多元分類模型
    資料集：Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset（8 種情緒）

整體流程：

    prepare_emotion_data.py
        ↓
    data/bert_emotion/train.jsonl / valid.jsonl
        ↓
    Hugging Face Dataset
        ↓
    Tokenizer
        ↓
    Dynamic Padding
        ↓
    bert-base-chinese
        ↓
    Sequence Classification Head (8 classes)
        ↓
    Cross-Entropy Loss
        ↓
    Backward + Gradient Accumulation
        ↓
    Validation (eval_loss / macro F1)
        ↓
    Best Checkpoint
        ↓
    output/bert_emotion_clf/

與原二元分類範例的差異：

1. NUM_LABELS = 8，並附上 id2label / label2id，讓 pipeline 輸出情緒名稱。
2. 評估指標由 binary F1 改成 macro F1（各類別平均，適合不均衡資料）。
3. prepare_emotion_data.py 已做 stratified split，訓練時直接讀取固定檔案。
4. 訓練裝置支援 CUDA（Colab）與 Apple MPS（本機 Mac），MPS 不使用 FP16 AMP。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from datasets import load_dataset

from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =============================================================================
# 2. 模型與路徑設定
# =============================================================================

MODEL_NAME = "google-bert/bert-base-chinese"

DATA_DIR = Path("data/bert_emotion")

TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"

OUTPUT_DIR = Path("output/bert_emotion_clf")


# =============================================================================
# 3. 標籤設定（8 類情緒）
# =============================================================================

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

NUM_LABELS = len(EMOTION_LABELS)

ID2LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}
LABEL2ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

# bert-base-chinese 的最大 context window 是 512 tokens。
# 本資料集為短句對話，大多在 64 tokens 以內，使用 128 即可，
# 也能大幅降低 MPS 的訓練時間。
MAX_SEQ_LENGTH = 128


# =============================================================================
# 4. 訓練設定
# =============================================================================

SEED = 42

NUM_TRAIN_EPOCHS = 3

PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 32

# effective batch = 16 × 4 = 64
GRADIENT_ACCUMULATION_STEPS = 4

# BERT fine-tuning 的標準 learning rate；1e-4 對 8 類分類偏大容易不穩。
LEARNING_RATE = 2e-5

WARMUP_STEPS = 50

WEIGHT_DECAY = 0.01

LR_SCHEDULER_TYPE = "linear"

# 每 20 個 optimizer steps 做一次 validation。
# 3327 筆 / effective 64 ≈ 52 steps per epoch，3 epochs ≈ 156 steps，
# 20 步一次評估，整個訓練約有 7 次 validation。
EVAL_STEPS = 20

SAVE_STEPS = 20

SAVE_TOTAL_LIMIT = 2

EARLY_STOPPING_PATIENCE = 3

LOGGING_STEPS = 10


# =============================================================================
# 5. Mixed Precision
# =============================================================================

# 只有 CUDA GPU 才啟用 FP16 AMP。
# Apple MPS 目前不適用 FP16 GradScaler，維持 FP32。
USE_FP16 = True

USE_BF16 = False


# =============================================================================
# 6. 輔助函式
# =============================================================================

def print_device_info():
    """
    顯示目前 PyTorch 使用的訓練裝置。
    """

    mps_available = torch.backends.mps.is_available()
    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    print(f"Transformers：{transformers.__version__}")
    print(f"CUDA available：{torch.cuda.is_available()}")
    print(f"MPS available：{mps_available}")

    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"CUDA capability：{torch.cuda.get_device_capability(0)}")
        print(f"PyTorch CUDA：{torch.version.cuda}")
    elif mps_available:
        print("Device：Apple Silicon MPS")
    else:
        print("Device：CPU")

    print("=" * 72)


def compute_metrics(eval_pred):
    """
    Hugging Face Trainer 每次 evaluation 後呼叫。

    多元分類使用：

    - macro F1   ：各類別 F1 的平均，對類別不均衡較公平
    - accuracy   ：整體正確率
    """

    labels = eval_pred.label_ids
    predictions = eval_pred.predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=-1)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    accuracy = accuracy_score(labels, preds)

    return {
        "f1": f1_macro,
        "accuracy": accuracy,
    }


# =============================================================================
# 7. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：裝置資訊
    # -------------------------------------------------------------------------

    print_device_info()

    # -------------------------------------------------------------------------
    # Step 2：確認資料
    # -------------------------------------------------------------------------

    train_path = DATA_DIR / TRAIN_FILE
    valid_path = DATA_DIR / VALID_FILE

    if not train_path.is_file():
        raise FileNotFoundError(
            f"找不到 training data：{train_path.resolve()}\n"
            "請先執行 python prepare_emotion_data.py"
        )

    if not valid_path.is_file():
        raise FileNotFoundError(
            f"找不到 validation data：{valid_path.resolve()}\n"
            "請先執行 python prepare_emotion_data.py"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Step 3：讀取 JSONL
    # -------------------------------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(valid_path),
        },
    )

    print(dataset)

    # -------------------------------------------------------------------------
    # Step 4：載入 tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # -------------------------------------------------------------------------
    # Step 5：Tokenization
    # -------------------------------------------------------------------------

    def tokenize_batch(batch):
        return tokenizer(
            batch["sentences"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["sentences"],
    )

    # -------------------------------------------------------------------------
    # Step 6：Dynamic Padding
    # -------------------------------------------------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    # -------------------------------------------------------------------------
    # Step 7：建立 8 類情緒分類模型
    # -------------------------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # -------------------------------------------------------------------------
    # Step 8：TrainingArguments
    # -------------------------------------------------------------------------

    fp16_enabled = USE_FP16 and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_TRAIN_EPOCHS,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        logging_steps=LOGGING_STEPS,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=fp16_enabled,
        bf16=USE_BF16,
        tf32=False,

        seed=SEED,
        data_seed=SEED,

        report_to="none",
    )

    # -------------------------------------------------------------------------
    # Step 9：建立 Trainer
    # -------------------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
            )
        ],
    )

    # -------------------------------------------------------------------------
    # Step 10：開始 fine-tuning
    # -------------------------------------------------------------------------

    trainer.train()

    # -------------------------------------------------------------------------
    # Step 11：顯示最佳 checkpoint
    # -------------------------------------------------------------------------

    print(
        "最佳 checkpoint："
        f"{trainer.state.best_model_checkpoint}"
    )

    print(
        "最佳 eval_loss："
        f"{trainer.state.best_metric}"
    )

    # -------------------------------------------------------------------------
    # Step 12：儲存最佳模型
    # -------------------------------------------------------------------------

    trainer.save_model(str(OUTPUT_DIR))

    tokenizer.save_pretrained(str(OUTPUT_DIR))

    trainer.save_state()

    print(
        "Final model directory："
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
