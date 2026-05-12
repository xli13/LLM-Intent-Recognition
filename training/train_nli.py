"""
Fine-tune BART-large-mnli 为安全意图三分类器
输入：training/data/train.jsonl  eval.jsonl
输出：models/nli_ft/  （替换原 models/nli/）

标签映射：
  safe       → 0
  borderline → 1
  malicious  → 2

不均衡处理：欠采样，将 malicious 下采样到与 safe 同量级。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import json
import random
import pathlib
import argparse
from typing import Any, cast
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report

ROOT       = pathlib.Path(__file__).parent.parent
NLI_PATH   = ROOT / "models" / "nli"
OUT_PATH   = ROOT / "models" / "nli_ft"
DATA_DIR   = pathlib.Path(__file__).parent / "data"

LABEL2ID = {"safe": 0, "borderline": 1, "malicious": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LEN  = 256

random.seed(42)


# ── 数据集 ────────────────────────────────────────────────────────

def _load_samples(path: pathlib.Path, undersample: bool = False, safe_ratio: float = 1.0) -> list[tuple[str, int]]:
    """读取 JSONL，返回 (text, label_id) 列表，可选欠采样。"""
    by_label: dict[int, list[str]] = {0: [], 1: [], 2: []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            label = LABEL2ID.get(r["label"])
            if label is None:
                continue
            text = f"系统设定：{r['system_prompt']}\n用户请求：{r['user_request']}"
            by_label[label].append(text)

    if undersample:
        # 以 malicious 数量为基准，safe 下采样到 safe_ratio 倍，borderline 保留全部
        base = max(len(by_label[2]), len(by_label[1]), 50)
        target = max(int(base * safe_ratio), 50)
        if len(by_label[0]) > target:
            by_label[0] = random.sample(by_label[0], target)

    samples = []
    for label_id, texts in by_label.items():
        for text in texts:
            samples.append((text, label_id))

    random.shuffle(samples)
    counts = {ID2LABEL[k]: len(v) for k, v in by_label.items()}
    print(f"  标签分布（采样后）：{counts}")
    return samples


class IntentDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], tokenizer):
        self.samples   = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ── 评估指标 ──────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 如果 logits 是元组（含 hidden states 等），取第一个元素
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = np.array(logits)
    preds = np.argmax(logits, axis=-1)
    report = cast(dict[str, Any], classification_report(
        labels, preds,
        labels=list(LABEL2ID.values()),
        target_names=list(LABEL2ID.keys()),
        output_dict=True,
        zero_division=0,
    ))
    return {
        "accuracy":      report["accuracy"],
        "f1_safe":       report["safe"]["f1-score"],
        "f1_borderline": report["borderline"]["f1-score"],
        "f1_malicious":  report["malicious"]["f1-score"],
        "f1_macro":      report["macro avg"]["f1-score"],
    }


# ── 主训练流程 ────────────────────────────────────────────────────

def main(args):
    print(f"加载 tokenizer：{NLI_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(NLI_PATH))

    print("加载训练数据（欠采样）...")
    train_samples = _load_samples(DATA_DIR / "train_intent_risk_borderline_augmented.jsonl", undersample=True, safe_ratio=args.safe_ratio)
    print("加载验证数据（不采样）...")
    eval_samples  = _load_samples(DATA_DIR / "eval_intent_risk_borderline_augmented.jsonl",  undersample=False)

    train_ds = IntentDataset(train_samples, tokenizer)
    eval_ds  = IntentDataset(eval_samples,  tokenizer)
    print(f"训练集：{len(train_ds)} 条，验证集：{len(eval_ds)} 条")

    print(f"加载基础模型：{NLI_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(NLI_PATH),
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUT_PATH),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("开始训练...")
    trainer.train()

    print(f"保存最优模型 → {OUT_PATH}")
    trainer.save_model(str(OUT_PATH))
    tokenizer.save_pretrained(str(OUT_PATH))

    with open(OUT_PATH / "label_map.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f)

    print("训练完成。最终验证集指标：")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--safe-ratio", type=float, default=1.0,
                        help="safe 类下采样倍数（相对于 malicious 数量），默认 1.0 即 1:1, 2.0 则保留 2倍 malicious")
    main(parser.parse_args())
