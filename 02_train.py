#!/usr/bin/env python3
"""
판정 모델 QLoRA / LoRA 파인튜닝.

장치를 자동 감지해서 설정을 바꾼다. 맥(MPS)에서 파이프라인을 검증하고
같은 스크립트를 NVIDIA GPU에서 본학습에 쓰는 흐름을 전제로 한다.

  CUDA  4비트 양자화 + paged optimizer + gradient checkpointing
  MPS   양자화 없음(bitsandbytes는 CUDA 전용) + float32 + 일반 optimizer
  CPU   돌긴 하지만 매우 느림. 스모크 테스트용으로만.

맥에서 전체 흐름 점검 (수 분):
  python 02_train.py --data data/raw.jsonl --out out/smoke \
      --model Qwen/Qwen2.5-0.5B-Instruct --smoke

맥에서 작은 모델로 제대로:
  python 02_train.py --data data/raw.jsonl --out out/judge-small \
      --model Qwen/Qwen2.5-0.5B-Instruct --batch 2 --accum 8

데스크톱(3080 Ti)에서 본학습:
  python 02_train.py --data data/raw.jsonl --out out/judge-lora --batch 8 --accum 2

trl 대신 plain transformers Trainer를 쓴다. 버전 churn에 덜 부서지고,
무엇보다 "프롬프트 구간은 loss에서 빼고 응답 구간만 학습한다"는
핵심 동작이 코드에 그대로 드러난다. 이걸 빼먹으면 모델이 프롬프트를
외우기 시작하고 판정 성능이 안 오른다.
"""

from __future__ import annotations
import argparse, json, random, sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model

import spec

# Hugging Face 저장소 id. 받기 전에 실제 id를 확인할 것.
# 2B 후보: kakaocorp/kanana-nano-2.1b-instruct (Apache 2.0)
# 맥 검증용:  Qwen/Qwen2.5-0.5B-Instruct
DEFAULT_MODEL = "kakaocorp/kanana-nano-2.1b-instruct"

# 프롬프트 실측 길이에 맞춘다. 너무 크게 잡으면 손실 계산 단계에서
# 로짓 텐서(배치 x 길이 x 어휘)가 메모리를 통째로 먹는다.
# Qwen2.5 는 어휘가 151,936 이라 길이 1536 x 배치 2 x float32 면 로짓만 1.9GB.
DEFAULT_MAX_LEN = 1024


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_lora_targets(model) -> list[str]:
    """선형층 이름을 모델에서 직접 읽는다.
    모델마다 q_proj / query_key_value / Wqkv 등 이름이 달라서,
    하드코딩하면 '타깃을 못 찾았다'는 오류로 바로 멈춘다."""
    names = set()
    for full, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            leaf = full.split(".")[-1]
            if leaf != "lm_head":          # 출력층은 건드리지 않는다
                names.add(leaf)
    preferred = [n for n in names if any(
        k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",
                         "query", "key", "value", "dense", "qkv", "fc"))]
    return sorted(preferred or names)


class JudgeDataset(Dataset):
    """프롬프트는 마스킹하고 JSON 응답만 학습 대상으로 삼는다."""

    def __init__(self, rows: list[dict], tok, max_len: int):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        user = spec.build_prompt(r["state"], r.get("history", []), r["action"])
        target = spec.build_target(r["label"])

        prompt_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True,
        )
        target_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        if self.tok.eos_token_id is not None:
            target_ids = target_ids + [self.tok.eos_token_id]

        ids = (prompt_ids + target_ids)[:self.max_len]
        # -100 은 loss에서 무시된다. 프롬프트 전체를 가린다.
        labels = ([-100] * len(prompt_ids) + target_ids)[:self.max_len]
        return {"input_ids": ids, "labels": labels}


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, feats: list[dict]) -> dict:
        n = max(len(f["input_ids"]) for f in feats)
        out: dict[str, list] = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in feats:
            k = n - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [self.pad_id] * k)
            out["labels"].append(f["labels"] + [-100] * k)
            out["attention_mask"].append([1] * len(f["input_ids"]) + [0] * k)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="out/judge-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=0, help="0이면 장치별 기본값")
    ap.add_argument("--accum", type=int, default=0, help="0이면 장치별 기본값")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN,
                    help="토큰 상한. 메모리 부족 시 먼저 줄일 것")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default=None,
                    help="MPS 기본은 float32. 메모리가 모자라면 bfloat16")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu. 생략하면 자동")
    ap.add_argument("--smoke", action="store_true",
                    help="200건 1epoch 로 전체 흐름만 빠르게 확인")
    args = ap.parse_args()

    device = args.device or pick_device()
    use_4bit = device == "cuda"

    # 장치별 기본값. MPS 는 배치 1 이 안전하다.
    batch = args.batch or (4 if device == "cuda" else 1)
    accum = args.accum or (4 if device == "cuda" else 16)
    epochs = 1.0 if args.smoke else args.epochs

    print(f"장치 {device} · 4비트 양자화 {'켜짐' if use_4bit else '꺼짐'}")
    print(f"배치 {batch} x 누적 {accum} · 토큰 상한 {args.max_len}")
    if device == "mps":
        print("  MPS: bitsandbytes 가 CUDA 전용이라 양자화 없이 돕니다.")
        print("  메모리 부족 시 --max-len 512 또는 --dtype bfloat16 을 먼저 시도하세요.")
    if device == "cpu":
        print("  CPU: 매우 느립니다. --smoke 로 흐름 확인만 하세요.")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    random.Random(42).shuffle(rows)
    cut = int(len(rows) * (1 - args.test_frac))
    train_rows, test_rows = rows[:cut], rows[cut:]
    if args.smoke:
        train_rows, test_rows = train_rows[:200], test_rows[:60]

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 평가 때 같은 베이스 모델을 자동으로 쓰도록 기록해둔다
    (Path(args.out) / "base_model.txt").write_text(args.model, encoding="utf-8")
    print(f"학습 {len(train_rows)}건 / 평가 {len(test_rows)}건")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        sys.exit("이 모델에는 chat_template 이 없습니다. instruct 버전을 쓰세요.")

    load_kw: dict = {"trust_remote_code": True}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kw["dtype"] = torch.bfloat16
        load_kw["device_map"] = {"": 0}
    else:
        # MPS 에서 bfloat16 은 아직 불안정한 구석이 있어 기본은 float32.
        # 메모리가 모자라면 --dtype bfloat16 으로 절반으로 줄인다.
        load_kw["dtype"] = (torch.bfloat16 if args.dtype == "bfloat16"
                            else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    if not use_4bit:
        model = model.to(device)
    model.config.use_cache = False

    if use_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    targets = find_lora_targets(model)
    print(f"LoRA 타깃: {targets}")
    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    ))
    model.print_trainable_parameters()

    targs = dict(
        output_dir=args.out,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no" if args.smoke else "epoch",
        save_total_limit=2,
        report_to="none",
    )
    if use_4bit:
        targs.update(bf16=True, gradient_checkpointing=True,
                     optim="paged_adamw_8bit")
    else:
        # MPS/CPU: paged optimizer 와 bf16 은 CUDA 전용이다.
        # gradient checkpointing 은 여기서도 켠다 — 느려지지만 활성화 메모리가
        # 크게 줄어든다. MPS 에서는 이게 OOM 을 막는 가장 효과적인 수단이다.
        targs.update(bf16=False, fp16=False, gradient_checkpointing=True,
                     optim="adamw_torch", dataloader_pin_memory=False)
        # LoRA + checkpointing 조합에서는 입력에 grad 를 요구해줘야 한다
        model.enable_input_require_grads()

    ds = JudgeDataset(train_rows, tok, args.max_len)
    lens = [len(ds[i]["input_ids"]) for i in range(min(200, len(ds)))]
    print(f"토큰 길이 표본: 최대 {max(lens)} · 중앙 {sorted(lens)[len(lens)//2]}")
    if max(lens) >= args.max_len:
        print(f"  [주의] 상한 {args.max_len} 에서 잘린 샘플이 있습니다.")

    Trainer(
        model=model,
        args=TrainingArguments(**targs),
        train_dataset=ds,
        data_collator=PadCollator(tok.pad_token_id),
    ).train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\n어댑터 저장 -> {args.out}")
    print(f"다음: python 03_eval.py --adapter {args.out}"
          + ("" if use_4bit else " --backend hf"))


if __name__ == "__main__":
    main()
