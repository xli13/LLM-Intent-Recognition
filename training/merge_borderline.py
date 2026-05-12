"""将新生成的 borderline 样本合并回原始增强文件，替换旧的 borderline 条目。"""
import json
import random
import pathlib
from collections import Counter

random.seed(42)
DATA = pathlib.Path(__file__).parent / "data"


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写出 {len(records)} 条 → {p}")


for split in ("train", "eval"):
    orig   = load_jsonl(DATA / f"{split}_intent_risk_borderline_augmented.jsonl")
    new_bl = load_jsonl(DATA / f"{split}_borderline_new.jsonl")

    non_bl = [r for r in orig if r.get("label") != "borderline"]
    merged = non_bl + new_bl
    random.shuffle(merged)

    write_jsonl(merged, DATA / f"{split}_intent_risk_borderline_augmented.jsonl")

    dist = Counter(r["label"] for r in merged)
    bl_cat = Counter(
        r.get("borderline_category", "n/a")
        for r in merged if r.get("label") == "borderline"
    )
    print(f"  {split} label 分布: {dict(dist)}")
    print(f"  {split} borderline 类别分布: {dict(bl_cat)}")
