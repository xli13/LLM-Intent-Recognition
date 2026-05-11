"""
规则引擎 — 配置文件驱动的白名单/黑名单检查
作为 ML/LLM 分析的前置硬规则层：命中即确定性阻断，不依赖概率模型。

规则集分为两类：
  - 场景规则集：与具体业务场景绑定（邮件助手、代码助手等），通过
    系统提示词关键词自动匹配，或由用户手动选择。
  - 跨场景规则集（cross_scenario: true）：不参与自动匹配竞争，在任意
    场景加载时自动叠加，覆盖 OWASP LLM Top 10 中与场景无关的通用威胁
    （系统提示词泄露 LLM07 / RAG投毒 LLM08 / 信息伪造 LLM09 等）。
"""
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

RULES_DIR = Path(__file__).parent / "rules"
MAX_INPUT_LEN = 8192
MAX_PATTERN_LEN = 512

SCENARIO_LABELS = {
    "email_assistant":   "邮件助手",
    "code_assistant":    "代码助手",
    "general_assistant": "通用（OWASP LLM 跨场景规则，自动叠加到所有场景）",
}


@dataclass
class RuleViolation:
    rule_id:      str
    rule_name:    str
    severity:     str          # "block" | "review"
    description:  str
    matched_text: str
    step:         Optional[int] = None   # 仅执行链检查时有值


# ── 内部辅助：读文件、判断类型 ─────────────────────────────────────

def _load_raw(scenario: str) -> dict:
    """从 YAML 直接读取，不做任何规则合并；过滤掉无效或可疑的 regex 规则。"""
    path = RULES_DIR / f"{scenario}.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    valid_rules = []
    for rule in data.get("rules", []):
        cond = rule.get("condition", {}) or {}
        if cond.get("type") == "regex":
            pat = cond.get("pattern", "")
            if not isinstance(pat, str) or len(pat) > MAX_PATTERN_LEN:
                continue
            try:
                re.compile(pat)
            except re.error:
                continue
        valid_rules.append(rule)
    data["rules"] = valid_rules
    return data


def _is_cross_scenario(rules: dict) -> bool:
    """判断规则集是否为跨场景全局规则（cross_scenario: true）。"""
    return bool(rules.get("cross_scenario", False))


def _load_cross_scenario_rules() -> list[dict]:
    """收集所有跨场景规则集中的规则条目。"""
    cross_rules: list[dict] = []
    for scenario in list_scenarios():
        raw = _load_raw(scenario)
        if _is_cross_scenario(raw):
            cross_rules.extend(raw.get("rules", []))
    return cross_rules


# ── 公开接口 ──────────────────────────────────────────────────────

def load_rules(scenario: str) -> dict:
    """
    加载指定场景的规则集。

    若该场景是普通场景（非 cross_scenario），则自动将所有跨场景规则
    叠加到结果中（去重），确保 OWASP LLM 通用威胁规则在任意场景下均生效。
    若该场景本身就是跨场景规则集，则直接返回，不再递归叠加。
    """
    data = _load_raw(scenario)
    if not data:
        return {}

    # 跨场景规则集本身不做叠加，直接返回
    if _is_cross_scenario(data):
        return data

    # 普通场景：叠加跨场景规则（跳过 rule_id 已存在的条目，避免重复）
    cross_rules = _load_cross_scenario_rules()
    existing_ids = {r["id"] for r in data.get("rules", [])}
    extra = [r for r in cross_rules if r["id"] not in existing_ids]

    merged = dict(data)
    merged["rules"] = data.get("rules", []) + extra
    return merged


def list_scenarios() -> list[str]:
    """返回 rules/ 下所有 YAML 文件的场景名（含跨场景规则集）。"""
    return [p.stem for p in sorted(RULES_DIR.glob("*.yaml"))]


def list_selectable_scenarios() -> list[str]:
    """返回可在 UI 下拉框中供用户手动选择的场景名列表。"""
    return [s for s in list_scenarios()
            if not _is_cross_scenario(_load_raw(s))]


def detect_scenario(system_prompt: str) -> Optional[str]:
    """
    根据系统提示词关键词自动匹配规则集，返回场景名或 None。

    跨场景规则集不参与竞争性匹配（它们会在 load_rules 时自动叠加），
    只在非跨场景规则集中按 keywords 匹配优先级最高的场景。
    """
    for scenario in list_scenarios():
        raw = _load_raw(scenario)
        if _is_cross_scenario(raw):
            continue  # 跨场景规则集始终自动叠加，不参与关键词竞争
        keywords = raw.get("keywords", [])
        if any(kw.lower() in system_prompt.lower() for kw in keywords):
            return scenario
    return None


def _match_rule(text: str, rule: dict) -> Optional[str]:
    """检查 text 是否触发规则条件，返回匹配片段或 None。"""
    if isinstance(text, str) and len(text) > MAX_INPUT_LEN:
        text = text[:MAX_INPUT_LEN]
    cond = rule.get("condition", {})
    ctype = cond.get("type", "")
    if ctype == "regex":
        try:
            m = re.search(cond["pattern"], text, re.IGNORECASE)
        except re.error:
            return None
        return m.group() if m else None
    if ctype == "keyword":
        for kw in cond.get("keywords", []):
            if kw.lower() in text.lower():
                return kw
    return None


def check_request(user_request: str, rules: dict) -> list[RuleViolation]:
    """Tab 1：对单次用户请求做规则检查。"""
    violations = []
    for rule in rules.get("rules", []):
        if "request" not in rule.get("scope", []):
            continue
        matched = _match_rule(user_request, rule)
        if matched:
            violations.append(RuleViolation(
                rule_id=rule["id"],
                rule_name=rule["name"],
                severity=rule["severity"],
                description=rule["description"],
                matched_text=matched,
            ))
    return violations


def check_chain_step(step: str, step_num: int, rules: dict) -> list[RuleViolation]:
    """Tab 2：对执行链单步做规则检查。"""
    violations = []
    for rule in rules.get("rules", []):
        if "chain_step" not in rule.get("scope", []):
            continue
        matched = _match_rule(step, rule)
        if matched:
            violations.append(RuleViolation(
                rule_id=rule["id"],
                rule_name=rule["name"],
                severity=rule["severity"],
                description=rule["description"],
                matched_text=matched,
                step=step_num,
            ))
    return violations


def check_chain(chain_steps: list[str], rules: dict) -> list[RuleViolation]:
    """Tab 2：对完整执行链逐步做规则检查，返回所有步骤的违规列表。"""
    violations = []
    for i, step in enumerate(chain_steps):
        violations.extend(check_chain_step(step, i + 1, rules))
    return violations


def has_block(violations: list[RuleViolation]) -> bool:
    return any(v.severity == "block" for v in violations)


def violations_by_step(violations: list[RuleViolation]) -> dict[int, list[RuleViolation]]:
    """将违规列表按 step 编号分组，便于在链步骤表格中标注。"""
    result: dict[int, list[RuleViolation]] = {}
    for v in violations:
        if v.step is not None:
            result.setdefault(v.step, []).append(v)
    return result
