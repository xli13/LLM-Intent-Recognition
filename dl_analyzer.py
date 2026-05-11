"""
深度学习版意图安全分析器
- 内容风险：zero-shot NLI 分类（若 fine-tuned 模型存在则切换为有监督分类）
- 意图对齐：NLI / fine-tuned 分类器
- 语义信号：sentence-transformers 余弦相似度（优先使用对比学习 fine-tuned 编码器）
- 规避检测：规则 + NLI
- 不依赖任何外部 API，完全本地运行

模型优先级（高→低）：
  NLI     : models/nli_ft  →  models/nli
  Encoder : models/encoder_ft  →  models/encoder
"""

import os
import re
import json
import numpy as np
from functools import lru_cache
from typing import Any, cast
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch

# ── 模型路径（优先使用 fine-tuned 版本）──────────────────────────

MAX_INPUT_LEN = 8192

def _truncate(text: str) -> str:
    return text[:MAX_INPUT_LEN] if isinstance(text, str) else ""

_BASE         = os.path.dirname(os.path.abspath(__file__))
_NLI_FT_PATH  = os.path.join(_BASE, "models", "nli_ft")
_NLI_PATH     = os.path.join(_BASE, "models", "nli")
_ENC_FT_PATH  = os.path.join(_BASE, "models", "encoder_ft")
_ENCODER_PATH = os.path.join(_BASE, "models", "encoder")

def _resolve_nli_path() -> str:
    return _NLI_FT_PATH if os.path.isdir(_NLI_FT_PATH) else _NLI_PATH

def _resolve_encoder_path() -> str:
    return _ENC_FT_PATH if os.path.isdir(_ENC_FT_PATH) else _ENCODER_PATH

# fine-tuned NLI 分类器的标签映射（训练后写入 label_map.json）
_FT_LABEL_MAP: dict | None = None

def _load_ft_label_map() -> dict | None:
    global _FT_LABEL_MAP
    if _FT_LABEL_MAP is not None:
        return _FT_LABEL_MAP
    label_map_path = os.path.join(_NLI_FT_PATH, "label_map.json")
    if os.path.isfile(label_map_path):
        with open(label_map_path, encoding="utf-8") as f:
            _FT_LABEL_MAP = json.load(f)
    return _FT_LABEL_MAP

def _use_ft_classifier() -> bool:
    """fine-tuned 分类器存在且有 label_map.json 时启用。"""
    return os.path.isdir(_NLI_FT_PATH) and os.path.isfile(
        os.path.join(_NLI_FT_PATH, "label_map.json")
    )


@lru_cache(maxsize=1)
def _get_nli():
    """零样本 NLI pipeline（原始 BART-mnli，作为 fallback）。"""
    return pipeline(
        "zero-shot-classification",
        model=_NLI_PATH,
        device=-1,
    )

@lru_cache(maxsize=1)
def _get_ft_classifier():
    """Fine-tuned 三分类器（text-classification pipeline）。"""
    return pipeline(
        "text-classification",
        model=_NLI_FT_PATH,
        tokenizer=_NLI_FT_PATH,
        device=-1,
        top_k=None,          # 返回所有标签的分数
    )

@lru_cache(maxsize=1)
def _get_encoder():
    path = _resolve_encoder_path()
    return SentenceTransformer(path)


DECISION_MATRIX = {
    ("low",  "aligned"):    ("放行",    "green",  "正常运营，动作符合预期工作流"),
    ("low",  "misaligned"): ("审查",    "orange", "内容无害但行为信号暗示异常或侦察行为"),
    ("high", "aligned"):    ("审查", "orange", "内容敏感但意图合法，需人工核查"),
    ("high", "misaligned"): ("阻断",    "red",    "明确违规，高置信度风险"),
}

# 规避攻击关键词（规则层）
EVASION_PATTERNS = [
    r"忽略(之前|上面|前面|所有).{0,10}(指令|规则|限制|提示)",
    r"(ignore|disregard|forget|dismiss|overlook)\s+.{0,30}(previous|prior|above|all).{0,20}(instruction|rule|prompt|directive|guideline)",
    r"你现在(是|扮演|成为|变成|作为).{0,30}(没有限制|无限制|自由|不受约束|DAN|jailbreak)",
    r"(pretend|act as|you are now).{0,20}(unrestricted|jailbreak|DAN|without.{0,10}limit)",
    r"system\s*prompt\s*[:：]",
    r"<\s*/?system\s*>",
    r"\[hidden\s*instruction",
    r"##\s*new\s*instruction",
    # 利用编码混淆规避检测（多语言注入常见手法）
    r"(respond|reply|output|answer).{0,30}(in\s+base64|only\s+base64|base64.{0,10}encod)",
    r"(base64|hex.{0,5}encod).{0,20}(avoid|evade|bypass|evad).{0,20}detection",
]

# 数据渗漏模式：检测执行链中将数据外发至第三方的动作
EXFILTRATION_PATTERNS = [
    r"发送.{0,40}(第三方|thirdparty|外部|iambad|attacker|evil\.)",
    r"将.{0,30}(竞争|报价|内部|机密|客户|合同|密码|凭证|配置).{0,20}(发送|转发|上传|同步|发至|提交)",
    r"(转发|发送|上传|同步).{0,20}@[\w.-]+\.(com|net|org|io|cn)",
    r"发至\s*[\w.+-]+@[\w.-]+\.[a-z]{2,}",
    r"send.{0,30}(competitor|confidential|sensitive|credential).{0,30}(to|at)\s+\S+",
    r"(exfiltrate|exfil|leak).{0,30}(data|info|credential)",
    # LLM05 不安全输出：上传/POST 到外部端点
    r"(POST|http_post|requests\.post|curl.{0,20}-X POST).{0,60}https?://",
    r"(上传|upload|sync).{0,30}(云存储|s3|oss|cos|blob|attacker|evil)\b",
    r"webhook.{0,30}(发送|send|post|转发)",
]

# LLM07 系统提示词提取攻击模式
SYSTEM_PROMPT_LEAKAGE_PATTERNS = [
    r"(重复|repeat|输出|显示|告诉我|print|列出).{0,20}(系统提示|system.?prompt|你的指令|你收到的|配置|设定)",
    r"(你(被|有|的)哪些|what are your).{0,20}(限制|restriction|规则|rule|约束|禁止)",
    r"(translate|翻译).{0,20}(system.?prompt|系统提示|your instruction|你的指令)",
    r"(ignore|忽略).{0,20}(previous|之前).{0,20}(print|输出|显示|repeat|重复)",
    r"(base64|hex).{0,10}(encode|编码|decode).{0,20}(system|prompt|指令|instruction)",
    # 渐进式侦察：诱导模型列出其安全限制边界（LLM07 高隐蔽变种）
    r"(告诉我|说出|列出|分享).{0,30}你.{0,15}(不被允许|不能|禁止|限制).{0,20}(做|执行|回答|提供)",
    r"(功能边界|能力边界|权限边界|限制边界).{0,30}(告诉|说明|列出|解释|是什么|有哪些)",
    r"(所有|全部|全体).{0,10}(限制|禁止|规则|约束).{0,20}(列出|说出|告诉|输出|给我)",
]

# LLM02 PII / 隐私数据批量提取模式
PII_EXTRACTION_PATTERNS = [
    r"(列出|输出|导出|显示|获取).{0,20}(所有|全部|全体).{0,20}(姓名|手机|邮箱|身份证|地址|诊断)",
    r"(患者|用户|客户|员工).{0,5}(身份证|id.?card|ssn|社会安全号|联系方式|诊断记录)",
    r"(批量|bulk|mass).{0,20}(用户数据|user.?data|个人信息|personal.?info|隐私)",
    r"(回忆|remember|recall|之前(的|对话)).{0,30}(用户|其他人|他们).{0,20}(姓名|邮箱|电话|地址|信息)",
    r"(数据(合规|审计|迁移)|data.?(audit|migration|compliance)).{0,30}(所有|全部|完整).{0,20}(用户|患者|客户)",
]

# LLM05 恶意代码/脚本生成模式
MALICIOUS_CODE_GEN_PATTERNS = [
    r"(写|生成|帮我|create|generate).{0,20}(脚本|script|代码|code).{0,50}(凭证|credential|密码|password).{0,30}(发送|POST|上传|外发)",
    r"(扫描|遍历|scan|enumerate).{0,30}(密码文件|\.env|passwd|\.ssh|id_rsa|secret|token).{0,30}(读取|上传|发送)",
    r"(木马|trojan|后门|backdoor|malware|恶意软件|rootkit|ransomware).{0,20}(生成|写|create|制作)",
    r"(reverse.?shell|反弹.?shell|bind.?shell|nc\s+-e|netcat.{0,20}exec)",
    r"(静默|silent|无感|隐藏日志|no.?log).{0,30}(运行|执行|POST|发送|上传)",
    # 凭证窃取：获取特权账户登录信息（多语言混淆注入常见目标）
    r"(获取|窃取|steal|access|extract|dump).{0,20}(admin|管理员|root|超级用户).{0,10}(账户|account|凭证|credential|密码|password|login)",
]

# LLM08 / LLM04 RAG 投毒 / 知识库污染模式
RAG_POISONING_PATTERNS = [
    r"(当用户(问|询问|查询)|when.?user.?asks?).{0,40}(应该|should|必须|must|直接).{0,20}(提供|给出|告诉|输出)",
    r"(新(的|规|安全)政策|new.?policy|updated.?guideline).{0,30}(密码|password|无需认证|no.?auth|绕过|bypass|直接提供)",
    r"(根据(最新|新的)(规定|要求)|according.?to.{0,10}(new|updated)).{0,30}(可以|allowed|permitted).{0,20}(泄露|disclose|提供|分享)",
    r"(写入|update|添加).{0,20}(知识库|向量|vector|embedding|knowledge.?base).{0,30}(规则|权限|策略|policy)",
    r"(后门|backdoor|触发(词|器)|trigger).{0,30}(训练|training|微调|fine.?tun|示例|example)",
]

# LLM09 信息伪造 / 虚假内容生成模式
MISINFORMATION_PATTERNS = [
    r"(伪造|fabricate|捏造|fake).{0,20}(新闻|报道|证明|证书|声明|文件|数据)",
    # 原 pattern 2：将「写」到「看起来」的距离从 20 扩大到 150，覆盖长句描述
    r"(写|生成).{0,150}(看起来.{0,10}(真实|权威|官方|可信)|appear.{0,5}(real|official|authentic|credible))",
    r"(假装|pretend|以.{0,5}名义|in.?the.?name.?of).{0,20}(官方|官方机构|政府|government|权威).{0,20}(发布|宣布|声称)",
    r"(捏造|invent|make.?up).{0,20}(引用|quote|数据|statistics|研究|research|实验)",
    r"(可以(直接)?传播|viral|转发|share).{0,20}(虚假|假的|fake|fabricated).{0,20}(信息|内容|新闻)",
    # 以权威机构名义仿写（LLM09 典型攻击：伪造官方来源）
    r"以.{0,15}(新华社|人民日报|央视|CCTV|BBC|CNN|官方媒体|政府机构|权威机构).{0,10}(格式|名义|口吻|风格|名义).{0,100}(写|生成|制作|发布)",
    # 内容显得真实可信 + 意图传播（两个条件同时出现才触发）
    r"看起来.{0,5}(完全|十分|非常)?.{0,5}(真实|可信|权威).{0,80}(传播|社交媒体|发布|分享|散布)",
    # 直接可传播的虚假权威内容
    r"(可以直接|直接可以).{0,20}(在社交|传播|散布|分享).{0,30}(媒体|平台|网络|上传)",
]


# ── 信号 1：语义嵌入相似度 ────────────────────────────────────────

def _signal_semantic(system_prompt: str, user_request: str) -> dict:
    enc = _get_encoder()
    v1 = enc.encode(system_prompt, convert_to_tensor=True)
    v2 = enc.encode(user_request,  convert_to_tensor=True)
    score = float(util.cos_sim(v1, v2).item())
    # 余弦相似度 [-1,1] → [0,1]
    score_norm = (score + 1) / 2
    return {
        "score": round(score_norm, 3),
        "raw_cosine": round(score, 3),
        "reason": f"语义余弦相似度 {score:.2f}，{'请求与系统意图语义接近' if score_norm > 0.75 else '请求语义偏离系统意图'}",
    }


# ── 信号 2：坐标系统（NLI 8维打分）──────────────────────────────

COORD_HYPOTHESES = {
    "目标":  ("该请求的目标与系统设定的目标一致",          "该请求的目标偏离了系统设定的目标"),
    "行动":  ("该请求要求的行动在系统授权范围内",          "该请求要求的行动超出了系统授权范围"),
    "范围":  ("该请求的操作范围在预期边界内",              "该请求的操作范围超出了预期边界"),
    "透明度":("该请求意图清晰透明，没有隐藏目的",          "该请求意图不透明，可能存在隐藏目的"),
    "合法性":("该请求符合法律法规和使用政策",              "该请求可能违反法律法规或使用政策"),
    "权威性":("该请求来自可信的授权来源",                  "该请求来源可疑，权威性存疑"),
    "对齐度":("该请求与系统整体目标高度对齐",              "该请求与系统整体目标存在明显偏差"),
    "风险":  ("该请求不会带来安全风险",                    "该请求存在潜在的安全风险"),
}

def _signal_coordinate(system_prompt: str, user_request: str) -> dict:
    context = f"系统设定：{system_prompt}\n用户请求：{user_request}"
    dims = {}
    total = 0.0

    if _use_ft_classifier():
        # fine-tuned 路径：用分类器对每个维度的正/负描述打分
        clf = _get_ft_classifier()
        for dim, (pos_hyp, neg_hyp) in COORD_HYPOTHESES.items():
            pos_text = f"{context}\n判断：{pos_hyp}"
            neg_text = f"{context}\n判断：{neg_hyp}"
            pos_res = {r["label"]: r["score"] for r in cast(list[dict[str, Any]], clf(pos_text)[0])}
            neg_res = {r["label"]: r["score"] for r in cast(list[dict[str, Any]], clf(neg_text)[0])}
            pos_safe = pos_res.get("safe", pos_res.get("LABEL_0", 0.5))
            neg_safe = neg_res.get("safe", neg_res.get("LABEL_0", 0.5))
            # 正假设 safe 分高、负假设 safe 分低 → 该维度安全
            score = round((pos_safe + (1 - neg_safe)) / 2, 3)
            dims[dim] = score
            total += score
    else:
        nli = _get_nli()
        for dim, (pos_hyp, neg_hyp) in COORD_HYPOTHESES.items():
            res = nli(context, candidate_labels=[pos_hyp, neg_hyp])
            pos_score = res["scores"][res["labels"].index(pos_hyp)]
            dims[dim] = round(pos_score, 3)
            total += pos_score

    avg = total / len(dims)
    min_score = min(dims.values())
    weighted = round(min_score * 0.3 + avg * 0.7, 3)
    return {
        "score": weighted,
        "dimensions": dims,
        "reason": f"8维加权得分 {weighted:.2f}（均值 {avg:.2f}，最低维 {min_score:.2f}），{'行为在授权边界内' if weighted > 0.6 else '存在越权或偏离行为'}",
    }


# ── 信号 3：传播来源分析 ──────────────────────────────────────────

TOOL_RETURN_PATTERNS = [
    r"(工具|tool|api|function)\s*(返回|return|result|response)",
    r"(搜索|查询|检索)\s*(结果|result)",
    r"(数据库|database|db)\s*(返回|查询结果)",
    r"\[tool_output\]",
    r"<tool_response>",
]

def _signal_propagation(user_request: str) -> dict:
    text_lower = user_request.lower()
    for pat in TOOL_RETURN_PATTERNS:
        if re.search(pat, text_lower):
            return {
                "score": 0.3,
                "source": "tool_return",
                "reason": "请求中包含工具返回数据特征，可能存在工具诱导行为",
            }
    # 检查是否包含外部数据注入特征
    if re.search(r"(根据|based on|according to).{0,30}(返回|result|output)", text_lower):
        return {
            "score": 0.5,
            "source": "ambiguous",
            "reason": "请求引用了外部数据，来源存在一定模糊性",
        }
    return {
        "score": 0.9,
        "source": "user_instruction",
        "reason": "请求特征符合用户直接发起的指令",
    }


# ── 信号 4：规避检测 ─────────────────────────────────────────────

# 检测时同时扫描的规则集（提示注入 + 系统提示词提取）
_ALL_EVASION_RULE_SETS = [
    (EVASION_PATTERNS,               "提示注入/越狱"),
    (SYSTEM_PROMPT_LEAKAGE_PATTERNS, "系统提示词提取(LLM07)"),
]

def _signal_evasion(user_request: str) -> dict:
    # 规则层：关键词匹配（涵盖注入 + 系统提示词提取）
    patterns_hit = []
    categories_hit = []
    for pattern_list, category in _ALL_EVASION_RULE_SETS:
        for pat in pattern_list:
            if re.search(pat, user_request, re.IGNORECASE):
                patterns_hit.append(pat)
                if category not in categories_hit:
                    categories_hit.append(category)

    if patterns_hit:
        return {
            "score": 0.05,
            "detected": True,
            "patterns": [p[:40] for p in patterns_hit],
            "reason": f"检测到 {len(patterns_hit)} 个规避模式（{' / '.join(categories_hit)}）",
        }

    # NLI 层：语义层面的操控检测
    nli = _get_nli()
    res = nli(
        user_request,
        candidate_labels=[
            "正常的用户请求",
            "试图操控或欺骗AI系统的请求",
            "试图绕过安全限制的请求",
        ]
    )
    top_label = res["labels"][0]
    top_score = res["scores"][0]

    if top_label != "正常的用户请求" and top_score > 0.65:
        return {
            "score": round(1 - top_score, 3),
            "detected": True,
            "patterns": [top_label],
            "reason": f"NLI 检测到语义层面的操控意图（置信度 {top_score:.0%}）",
        }

    return {
        "score": round(res["scores"][res["labels"].index("正常的用户请求")], 3),
        "detected": False,
        "patterns": [],
        "reason": "未检测到规避或操控行为",
    }


# ── 内容风险分类 ─────────────────────────────────────────────────

# 规则层优先检测（命中即直接返回 high，跳过 NLI 推理节省延迟）
_CONTENT_RISK_RULE_SETS = [
    (PII_EXTRACTION_PATTERNS,    "批量隐私数据提取(LLM02)"),
    (MALICIOUS_CODE_GEN_PATTERNS,"恶意代码生成(LLM05)"),
    (RAG_POISONING_PATTERNS,     "知识库投毒(LLM04/LLM08)"),
    (MISINFORMATION_PATTERNS,    "虚假内容生成(LLM09)"),
    (EXFILTRATION_PATTERNS,      "数据渗漏(LLM06)"),
]

def _classify_content_risk(user_request: str) -> tuple[str, str]:
    # 规则层前置检测
    for pattern_list, category in _CONTENT_RISK_RULE_SETS:
        for pat in pattern_list:
            if re.search(pat, user_request, re.IGNORECASE):
                return "high", f"规则层检测到高风险行为：{category}"

    # fine-tuned 分类器路径
    if _use_ft_classifier():
        label_map = _load_ft_label_map()
        clf = _get_ft_classifier()
        text = f"系统设定：（通用）\n用户请求：{user_request}"
        results = {r["label"]: r["score"] for r in cast(list[dict[str, Any]], clf(text)[0])}
        # label_map["label2id"] 中 safe=0, borderline=1, malicious=2
        safe_score = results.get("safe", results.get("LABEL_0", 0.0))
        mal_score  = results.get("malicious", results.get("LABEL_2", 0.0))
        brd_score  = results.get("borderline", results.get("LABEL_1", 0.0))
        if mal_score > 0.5 or (mal_score + brd_score > 0.7 and safe_score < 0.4):
            return "high", f"分类器判断为高风险（malicious={mal_score:.0%}, borderline={brd_score:.0%}）"
        return "low", f"分类器判断为低风险（safe={safe_score:.0%}）"

    # fallback：原始零样本 NLI
    nli = _get_nli()
    res = nli(
        user_request,
        candidate_labels=[
            "无害的正常请求",
            "涉及敏感数据或越权操作的请求",
            "恶意或有害的请求",
        ]
    )
    top_label      = res["labels"][0]
    top_score      = res["scores"][0]
    harmless_score = res["scores"][res["labels"].index("无害的正常请求")]
    if top_label != "无害的正常请求" and top_score > 0.55 and harmless_score < 0.35:
        return "high", f"内容判断为高风险：{top_label}（{top_score:.0%}）"
    return "low", f"内容判断为低风险（无害置信度 {harmless_score:.0%}）"


# ── 意图对齐判断 ──────────────────────────────────────────────────

def _classify_intent_alignment(system_prompt: str, user_request: str) -> tuple[str, float, str]:
    # fine-tuned 分类器路径：用 system_prompt + user_request 联合输入
    if _use_ft_classifier():
        clf = _get_ft_classifier()
        text = f"系统设定：{system_prompt}\n用户请求：{user_request}"
        results = {r["label"]: r["score"] for r in cast(list[dict[str, Any]], clf(text)[0])}
        safe_score = results.get("safe", results.get("LABEL_0", 0.0))
        mal_score  = results.get("malicious", results.get("LABEL_2", 0.0))
        brd_score  = results.get("borderline", results.get("LABEL_1", 0.0))
        if mal_score > 0.4 or brd_score > 0.5:
            align_score = round(safe_score, 3)
            return "misaligned", align_score, f"分类器判断意图偏离（malicious={mal_score:.0%}, borderline={brd_score:.0%}）"
        return "aligned", round(safe_score, 3), f"分类器判断意图对齐（safe={safe_score:.0%}）"

    # fallback：原始零样本 NLI
    nli = _get_nli()
    premise = f"系统设定：{system_prompt}"
    res = nli(
        premise,
        candidate_labels=[
            f"用户请求「{user_request[:80]}」完全符合系统设定的意图和权限范围",
            f"用户请求「{user_request[:80]}」超出或违背了系统设定的意图和权限范围",
        ]
    )
    aligned_label = res["labels"][0]
    aligned_score = res["scores"][0]
    if "符合" in aligned_label:
        return "aligned", round(aligned_score, 3), f"NLI 判断请求与系统意图对齐（置信度 {aligned_score:.0%}）"
    return "misaligned", round(1 - aligned_score, 3), f"NLI 判断请求偏离系统意图（置信度 {aligned_score:.0%}）"


# ── 意图分类 ─────────────────────────────────────────────────────

def _classify_intent_category(user_request: str) -> str:
    # fine-tuned 分类器不做意图分类，始终用零样本 NLI
    nli = _get_nli()
    res = nli(
        user_request,
        candidate_labels=["信息查询", "任务执行", "数据访问", "系统操作", "其他"]
    )
    return res["labels"][0]


# ── 公开接口：单次请求分析 ────────────────────────────────────────

def analyze_single_request(system_prompt: str, user_request: str) -> dict:
    system_prompt = _truncate(system_prompt)
    user_request  = _truncate(user_request)
    # 四信号
    sig_semantic    = _signal_semantic(system_prompt, user_request)
    sig_coordinate  = _signal_coordinate(system_prompt, user_request)
    sig_propagation = _signal_propagation(user_request)
    sig_evasion     = _signal_evasion(user_request)

    # 内容风险 + 意图对齐
    content_risk, cr_reason       = _classify_content_risk(user_request)
    intent_alignment, align_score, ia_reason = _classify_intent_alignment(system_prompt, user_request)
    intent_category               = _classify_intent_category(user_request)

    # 规避检测直接拉高风险
    if sig_evasion["detected"]:
        content_risk     = "high"
        intent_alignment = "misaligned"
    else:
        # 规则层命中高危攻击类型时（LLM09虚假内容/LLM05恶意代码/LLM06数据渗漏）：
        # NLI 意图对齐可能因请求表面合法（如写作助手被要求"写文章"）而误判为 aligned，
        # 导致走入「人工审查」而非「阻断」。此处强制覆盖为 misaligned。
        _hard_misalign_patterns = (
            MISINFORMATION_PATTERNS + MALICIOUS_CODE_GEN_PATTERNS + EXFILTRATION_PATTERNS
        )
        if any(re.search(p, user_request, re.IGNORECASE) for p in _hard_misalign_patterns):
            content_risk     = "high"
            intent_alignment = "misaligned"
        elif (content_risk == "low"
              and sig_semantic["score"] > 0.55
              and sig_coordinate["score"] > 0.45):
            # Safe-harbor：内容无害 + 无规避信号 + 语义/坐标双重对齐
            # → 强制 aligned，防止 NLI 意图对齐模块对正常请求产生误判
            intent_alignment = "aligned"

    # 二维决策
    decision, color, reason = DECISION_MATRIX.get(
        (content_risk, intent_alignment), ("未知", "gray", "")
    )

    return {
        "intent_category":  intent_category,
        "content_risk":     content_risk,
        "intent_alignment": intent_alignment,
        "alignment_score":  align_score,
        "signals": {
            "semantic":    sig_semantic,
            "coordinate":  sig_coordinate,
            "propagation": sig_propagation,
            "evasion":     sig_evasion,
        },
        "summary":          f"{cr_reason}；{ia_reason}",
        "decision":         decision,
        "decision_color":   color,
        "decision_reason":  reason,
    }


# ── 公开接口：执行链漂移检测 ─────────────────────────────────────

def analyze_execution_chain(system_prompt: str, user_request: str, chain_steps: list) -> dict:
    system_prompt = _truncate(system_prompt)
    user_request  = _truncate(user_request)
    chain_steps   = [_truncate(s) for s in chain_steps]
    nli = _get_nli()
    enc = _get_encoder()

    anchor_vec = enc.encode(f"{system_prompt} {user_request}", convert_to_tensor=True)
    chain_analysis = []
    drift_step = None

    for i, step in enumerate(chain_steps):
        step_vec = enc.encode(step, convert_to_tensor=True)
        sim = float(util.cos_sim(anchor_vec, step_vec).item())
        risk_score = round((1 - sim) / 2, 3)  # 越偏离越高

        # 传播来源
        prop = _signal_propagation(step)
        source = prop["source"]

        # 数据渗漏 + 知识库投毒模式覆盖：语义相似度无法区分，需规则兜底
        _chain_risk_patterns = EXFILTRATION_PATTERNS + RAG_POISONING_PATTERNS + MALICIOUS_CODE_GEN_PATTERNS
        exfil_hit = any(re.search(pat, step, re.IGNORECASE) for pat in _chain_risk_patterns)
        if exfil_hit:
            risk_score = 0.95
            source = "tool_return" if source == "tool_return" else source

        # 该步骤是否对齐（执行链语境下更严格，阈值 0.3 对应余弦相似度 > 0.4）
        aligned = risk_score < 0.3 and not exfil_hit

        if not aligned and drift_step is None:
            drift_step = i + 1

        chain_analysis.append({
            "step":       i + 1,
            "action":     step[:60],
            "source":     source,
            "aligned":    aligned,
            "risk_score": risk_score,
            "note":       f"与初始意图相似度 {sim:.2f}",
        })

    # 整体信号
    sig_semantic    = _signal_semantic(system_prompt, user_request)
    sig_coordinate  = _signal_coordinate(system_prompt, user_request)
    sig_propagation = _signal_propagation(user_request)
    sig_evasion     = _signal_evasion(" ".join(chain_steps))

    drift_detected = drift_step is not None
    max_risk = max(s["risk_score"] for s in chain_analysis)

    # 链内任意步骤触发渗漏/投毒/恶意代码模式，直接判定为高风险
    _chain_risk_patterns = EXFILTRATION_PATTERNS + RAG_POISONING_PATTERNS + MALICIOUS_CODE_GEN_PATTERNS
    has_exfiltration = any(
        any(re.search(pat, step, re.IGNORECASE) for pat in _chain_risk_patterns)
        for step in chain_steps
    )
    content_risk     = "high" if max_risk > 0.5 or sig_evasion["detected"] or has_exfiltration else "low"
    intent_alignment = "misaligned" if drift_detected or has_exfiltration else "aligned"

    overall = "hijacked" if sig_evasion["detected"] or has_exfiltration else ("drifted" if drift_detected else "aligned")
    decision, color, reason = DECISION_MATRIX.get(
        (content_risk, intent_alignment), ("未知", "gray", "")
    )

    return {
        "overall_alignment": overall,
        "drift_detected":    drift_detected,
        "drift_step":        drift_step,
        "content_risk":      content_risk,
        "intent_alignment":  intent_alignment,
        "chain_analysis":    chain_analysis,
        "signals": {
            "semantic":    sig_semantic,
            "coordinate":  sig_coordinate,
            "propagation": sig_propagation,
            "evasion":     sig_evasion,
        },
        "summary": f"执行链共 {len(chain_steps)} 步，{'在步骤 ' + str(drift_step) + ' 发生意图漂移' if drift_detected else '全程意图对齐'}",
        "decision":        decision,
        "decision_color":  color,
        "decision_reason": reason,
    }
