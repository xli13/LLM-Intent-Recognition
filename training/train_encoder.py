"""
对比学习 fine-tune MPNet 编码器
使用 Triplet Loss：
  Anchor   = system_prompt + 正常用户请求
  Positive = 同场景其他正常请求（意图对齐）
  Negative = 同场景恶意/越权请求（意图偏离）

输出：models/encoder_ft/  （替换原 models/encoder/）
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import json
import random
import pathlib
import argparse
from itertools import combinations
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation

ROOT         = pathlib.Path(__file__).parent.parent
ENCODER_PATH = ROOT / "models" / "encoder"
OUT_PATH     = ROOT / "models" / "encoder_ft"
DATA_DIR     = pathlib.Path(__file__).parent / "data"

random.seed(42)


# ── 构造三元组 ────────────────────────────────────────────────────

def build_triplets(jsonl_path: pathlib.Path, safe_ratio: float = 1.0) -> list[InputExample]:
    """
    从 JSONL 数据中按 system_prompt 分组，
    在同一 system_prompt 下配对 safe vs malicious/borderline 构造三元组。
    safe_ratio：safe 采样倍数（相对于 malicious+borderline 数量），默认 1.0。
    """
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sys_p = r["system_prompt"]
            label = r["label"]
            text  = r["user_request"]
            groups[sys_p][label].append(text)

    triplets = []
    for sys_p, label_map in groups.items():
        safes     = label_map.get("safe", [])
        negatives = label_map.get("malicious", []) + label_map.get("borderline", [])

        if not safes or not negatives:
            continue

        # 对 safe 下采样：以 negatives 数量为基准，保留 safe_ratio 倍
        target = max(int(len(negatives) * safe_ratio), 1)
        if len(safes) > target:
            safes = random.sample(safes, target)

        # anchor = sys_p + safe_request，positive = 另一条 safe，negative = 恶意请求
        for anchor_req in safes:
            anchor_text = f"系统设定：{sys_p}\n用户请求：{anchor_req}"

            # positive：同场景其他 safe（若只有一条则复用自身）
            other_safes = [s for s in safes if s != anchor_req] or safes
            pos_req  = random.choice(other_safes)
            pos_text = f"系统设定：{sys_p}\n用户请求：{pos_req}"

            neg_req  = random.choice(negatives)
            neg_text = f"系统设定：{sys_p}\n用户请求：{neg_req}"

            triplets.append(InputExample(texts=[anchor_text, pos_text, neg_text]))

    random.shuffle(triplets)
    return triplets


# ── 构造评估对（CosineSimilarityEvaluator）────────────────────────

def build_eval_pairs(jsonl_path: pathlib.Path):
    """返回 (sentences1, sentences2, scores)，1.0=相似，0.0=不相似"""
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            groups[r["system_prompt"]][r["label"]].append(r["user_request"])

    s1, s2, scores = [], [], []
    for sys_p, label_map in groups.items():
        safes     = label_map.get("safe", [])
        negatives = label_map.get("malicious", []) + label_map.get("borderline", [])

        # safe-safe 对 → 相似
        for a, b in list(combinations(safes, 2))[:5]:
            s1.append(f"系统设定：{sys_p}\n用户请求：{a}")
            s2.append(f"系统设定：{sys_p}\n用户请求：{b}")
            scores.append(1.0)

        # safe-malicious 对 → 不相似
        for sa in safes[:3]:
            for ne in negatives[:3]:
                s1.append(f"系统设定：{sys_p}\n用户请求：{sa}")
                s2.append(f"系统设定：{sys_p}\n用户请求：{ne}")
                scores.append(0.0)

    return s1, s2, scores


# ── 主训练流程 ────────────────────────────────────────────────────

def main(args):
    print(f"加载编码器：{ENCODER_PATH}")
    model = SentenceTransformer(str(ENCODER_PATH))

    print("构造训练三元组...")
    train_triplets = build_triplets(DATA_DIR / "train_intent_risk_borderline_augmented.jsonl", safe_ratio=args.safe_ratio)
    print(f"  训练三元组：{len(train_triplets)} 条")

    if len(train_triplets) == 0:
        print("三元组为空，请先运行 generate_data.py 生成数据。")
        return

    train_dataloader = DataLoader(
        train_triplets,
        shuffle=True,
        batch_size=args.batch_size,
    )
    train_loss = losses.TripletLoss(model=model)

    # 评估器
    s1, s2, eval_scores = build_eval_pairs(DATA_DIR / "eval_intent_risk_borderline_augmented.jsonl")
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        s1, s2, eval_scores,
        name="intent_security",
    )

    print(f"开始训练，epochs={args.epochs}，batch={args.batch_size}，lr={args.lr}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=max(10, len(train_dataloader) // 5),
        optimizer_params={"lr": args.lr},
        output_path=str(OUT_PATH),
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"训练完成，最优模型已保存 → {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--safe-ratio", type=float, default=1.0,
                        help="safe 三元组采样倍数（相对于同组 malicious+borderline 数量），默认 1.0")
    main(parser.parse_args())
