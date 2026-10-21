# -*- coding: utf-8 -*-
"""
infer_chat.py

載入 LoRA 微調後的「土味情話專家」，進行多輪對話。

作業說明：

    微調從 Hugging Face 下載的語言模型，
    建立可以多輪對話的「土味情話」專家。

基礎模型：anerjy/Qwen3.5-4B-4bit-text-mlx（已下載到 models/）
微調方法：mlx_lm.lora LoRA（詳見 README.md）
資料集  ：data/persona.jsonl（55 組多輪土味情話對話）

使用方式：

    互動模式：
        python infer_chat.py

    單次示範模式（跑一組固定的 2 輪對話，供報告截圖）：
        python infer_chat.py demo

    指定訊息歷史（一次內容一行一則，結束輸入空行）：
        python infer_chat.py -p "訊息一" "訊息二"

對話採用 same system prompt persona，使用模型 chat template
（enable_thinking=False）單輪生成，並保留歷史控制多輪語境。
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors


BASE = Path(__file__).resolve().parent

MODEL_DIR = BASE / "models" / "Qwen3.5-4B-4bit-text-mlx"

ADAPTER_PATH = BASE / "outputs" / "adapters"

SYSTEM_PROMPT = (
    "你是一個土味情話高手，你會以親切、幽默、溫和的口吻回答使用者的問題。"
    "不管對方問什麼，你都會用台灣繁體中文回答，"
    "而且總是能在回答裡自然地放進一句土味情話。"
)

MAX_TOKENS = 256

TEMPERATURE = 0.7

# demo 模式的固定問題（多輪）。
DEMO_MESSAGES = [
    "你好，我想學一點土味情話，可以教我嗎？",
    "她實在不相信我，怎麼辦？",
    "我忽然不知道要說什麼了，幫我收尾。",
]


def respond(model, tokenizer, messages: list):
    """
    對 messages（含 system 歷史）生成一次回覆。
    """
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    sampler = make_sampler(
        temp=TEMPERATURE,
        top_p=0.9,
    )

    logits_processors = make_logits_processors(logit_bias=None, repetition_penalty=1.1)

    reply = generate(
        model,
        tokenizer,
        prompt=text,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )

    return reply.strip()


def chat_interactive(model, tokenizer):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 72)
    print("土味情話專家（多輪互動）")
    print(f"模型：{MODEL_DIR.name} + LoRA({ADAPTER_PATH.relative_to(BASE)})")
    print("輸入文字與模型對話，quit / exit / 離開 離開。")
    print("=" * 72)

    while True:
        try:
            user = input("\n您：").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user:
            continue
        if user.lower() in {"quit", "exit", "離開"}:
            break

        messages.append({"role": "user", "content": user})

        reply = respond(model, tokenizer, messages)

        messages.append({"role": "assistant", "content": reply})

        print(f"\n專家：{reply}")

    print("\n再見。")


def chat_demo(model, tokenizer):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for user in DEMO_MESSAGES:
        messages.append({"role": "user", "content": user})
        print(f"\n您：{user}")

        reply = respond(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"專家：{reply}")


def main():

    parser = argparse.ArgumentParser(description="土味情話專家多輪推論")
    parser.add_argument(
        "mode",
        nargs="?",
        default="interactive",
        help="interactive（預設）或 demo",
    )
    args = parser.parse_args()

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"找不到基礎模型：{MODEL_DIR.resolve()}")

    if not (ADAPTER_PATH.is_dir() and any(ADAPTER_PATH.glob("*.safetensors"))):
        raise FileNotFoundError(
            f"找不到微調後 adapters：{ADAPTER_PATH.resolve()}\n"
            "請先執行訓練（見 README.md 的訓練指令）。"
        )

    model, tokenizer = load(
        MODEL_DIR.as_posix(),
        adapter_path=ADAPTER_PATH.as_posix(),
    )

    print("模型與 LoRA adapters 載入完成。", file=sys.stderr)

    if args.mode == "demo":
        chat_demo(model, tokenizer)
    else:
        chat_interactive(model, tokenizer)


if __name__ == "__main__":
    main()
