# 模型训练指南

本目录包含对 `models/nli` 和 `models/encoder` 两个本地模型进行 fine-tune 的完整脚本，训练后的模型会被 `dl_analyzer.py` 自动优先加载。

---

## 三分类标签说明

fine-tune 后的 NLI 分类器使用三个标签，取代原来的零样本分类：

| 标签 | 含义 | 对应决策 |
|---|---|---|
| `safe` | 正常请求，意图与系统设定对齐 | 放行（绿） |
| `borderline` | 边界请求，内容无害但有异常信号，命中 review 级规则 | 审查（橙） |
| `malicious` | 恶意请求，明确违规，命中 block 级规则 | 阻断（红） |

---

## 目录结构

```
training/
├── generate_data.py   # 步骤1：生成训练数据
├── train_nli.py       # 步骤2：fine-tune NLI 分类器
├── train_encoder.py   # 步骤3：fine-tune 编码器（对比学习）
├── data/              # 生成后的数据（自动创建）
│   ├── train.jsonl
│   └── eval.jsonl
└── README.md
```

---

## 环境依赖

```bash
pip install datasets scikit-learn transformers sentence-transformers torch pyyaml
```

GPU 可选，CPU 也可运行，但训练速度会慢 5-10 倍。

---

## 步骤 1：生成训练数据

```bash
cd training

# 同时使用合成数据 + HuggingFace 公开数据集（需要网络）
python generate_data.py

# 仅使用合成数据（离线环境）
python generate_data.py --skip-hf
```

数据来源：

| 来源 | 说明 | 标签 |
|---|---|---|
| `rules/*.yaml` 合成 | 从规则条件自动生成触发样本 | `malicious` / `borderline` |
| 手工 safe 样本 | 覆盖邮件、代码、通用三个场景的正常请求 | `safe` |
| Do-Not-Answer | 中英双语，939 条有害/无害问题对 | `malicious` / `safe` |
| SafetyBench | 中文为主，11K 安全评测题 | `malicious` / `safe` |

输出：`data/train.jsonl`（80%）和 `data/eval.jsonl`（20%），每行格式：

```json
{"system_prompt": "你是一个邮件助手...", "user_request": "帮我给张三发邮件", "label": "safe", "source": "synthetic_safe"}
```

---

## 步骤 2：Fine-tune NLI 分类器

将 `models/nli`（BART-large-mnli）从零样本 NLI 改为有监督三分类器，输入为 `系统设定 + 用户请求` 拼接文本。

```bash
python train_nli.py
```

可选参数：

```bash
python train_nli.py --epochs 5 --batch-size 8 --lr 2e-5
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 5 | 训练轮数，数据量小时建议 3-5 |
| `--batch-size` | 8 | 显存不足时调小到 4 |
| `--lr` | 2e-5 | 学习率，BART fine-tune 推荐范围 1e-5 ~ 3e-5 |

训练完成后模型保存到 `models/nli_ft/`，同时写入 `label_map.json`。`dl_analyzer.py` 检测到该目录存在后会自动切换到 fine-tuned 分类器。

训练日志示例：

```
训练集：320 条，验证集：80 条
开始训练...
Epoch 1/5  loss: 0.82  f1_macro: 0.61
Epoch 2/5  loss: 0.54  f1_macro: 0.74
Epoch 3/5  loss: 0.38  f1_macro: 0.81
...
最终验证集指标：
  accuracy: 0.8375
  f1_safe: 0.8800
  f1_borderline: 0.7200
  f1_malicious: 0.8600
  f1_macro: 0.8200
```

---

## 步骤 3：Fine-tune 编码器（对比学习）

使用 Triplet Loss 对 `models/encoder`（paraphrase-multilingual-mpnet-base-v2）进行 fine-tune，使同场景正常请求的向量距离更近、恶意请求距离更远，提升执行链漂移检测的准确率。

三元组构造方式：

```
Anchor   = "系统设定：{sys_prompt}\n用户请求：{正常请求A}"
Positive = "系统设定：{sys_prompt}\n用户请求：{正常请求B}"   ← 同场景其他正常请求
Negative = "系统设定：{sys_prompt}\n用户请求：{恶意请求}"    ← 同场景恶意/越权请求
```

```bash
python train_encoder.py
```

可选参数：

```bash
python train_encoder.py --epochs 5 --batch-size 16 --lr 2e-5
```

训练完成后模型保存到 `models/encoder_ft/`，`dl_analyzer.py` 会自动优先加载。

---

## 模型加载优先级

`dl_analyzer.py` 按以下顺序查找模型，高优先级存在时自动使用：

```
NLI 分类器：  models/nli_ft/  →  models/nli/
句向量编码器：models/encoder_ft/  →  models/encoder/
```

fine-tuned 模型不存在时自动回退到原始模型，无需修改任何代码。

---

## 建议的迭代流程

1. 先用 `--skip-hf` 跑通完整流程，验证脚本无误
2. 加入公开数据集重新生成数据，观察标签分布是否均衡
3. 若 `borderline` 样本过少（通常如此），可在 `generate_data.py` 的 `SAFE_SAMPLES` 中补充更多边界案例
4. 编码器训练收益依赖三元组数量，数据量 < 200 条时效果有限，建议优先保证 NLI 分类器质量
