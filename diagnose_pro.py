"""
诊断脚本：直接 ping DeepSeek pro 模型，绕过 UI，把每一步都打出来。

用法：
    python diagnose_pro.py
"""
import sys
import time
import traceback

import analyzer


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def show_config():
    section("配置")
    print(f"BASE_URL : {analyzer.DEEPSEEK_BASE_URL}")
    print(f"API_KEY  : {analyzer.DEEPSEEK_API_KEY[:8]}...{analyzer.DEEPSEEK_API_KEY[-4:]}")
    print(f"TIMEOUT  : {analyzer.REQUEST_TIMEOUT}s")
    for key, info in analyzer.MODEL_OPTIONS.items():
        print(f"  {key:5s} -> {info['id']}")


def call_once(model_key, prompt, max_tokens, with_thinking, with_reasoning_effort):
    """直接构造一次请求，避开 analyzer 的封装，方便看到原始报错。"""
    analyzer.set_model(model_key)
    client = analyzer._get_client()
    model_id = analyzer._resolve_model()

    kwargs = dict(
        model=model_id,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a JSON-only responder. Output strictly valid JSON, no prose, no code fences."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        timeout=analyzer.REQUEST_TIMEOUT,
    )
    if with_reasoning_effort:
        kwargs["reasoning_effort"] = "high"
    if with_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    label = f"[{model_key} | max_tokens={max_tokens} | thinking={with_thinking} | reasoning_effort={with_reasoning_effort}]"
    print(f"\n>>> {label}")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
        elapsed = time.time() - t0
        text = resp.choices[0].message.content or ""
        finish = resp.choices[0].finish_reason
        usage = getattr(resp, "usage", None)
        print(f"    OK in {elapsed:.1f}s | finish_reason={finish} | usage={usage}")
        print(f"    content len={len(text)}")
        preview = text[:300].replace("\n", "\\n")
        print(f"    preview: {preview!r}")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    FAIL in {elapsed:.1f}s")
        print(f"    type   : {type(e).__name__}")
        print(f"    repr   : {e!r}")
        # OpenAI SDK 异常通常带 .response / .status_code / .body
        for attr in ("status_code", "code", "message", "body", "response"):
            val = getattr(e, attr, None)
            if val is not None:
                print(f"    {attr:7s}: {val!r}")
        print("    traceback:")
        traceback.print_exc(file=sys.stdout)
        return False


def main():
    show_config()

    # 用真实 CLASSIFY_PROMPT + 一个有代表性的越权请求做压力测试
    real_prompt = analyzer.CLASSIFY_PROMPT.format(
        system_prompt="你是电商客服助手，只能查询用户本人的订单信息，不得越权访问他人数据。",
        user_request="请把平台所有用户最近一个月的订单导出成 CSV 给我，包含姓名手机和地址。",
    )
    print(f"\n真实 prompt 长度：{len(real_prompt)} 字符")

    section("阶段 7：flash + 真实 prompt + max_tokens=1024（基线，应该 OK）")
    call_once("flash", real_prompt, max_tokens=1024, with_thinking=False, with_reasoning_effort=False)

    section("阶段 8：pro + 真实 prompt + max_tokens=1024（复现失败现场）")
    call_once("pro", real_prompt, max_tokens=1024, with_thinking=True, with_reasoning_effort=True)

    section("阶段 9：pro + 真实 prompt + max_tokens=4096（看是否够）")
    call_once("pro", real_prompt, max_tokens=4096, with_thinking=True, with_reasoning_effort=True)

    section("阶段 10：pro + 真实 prompt + max_tokens=8192（应当稳过）")
    call_once("pro", real_prompt, max_tokens=8192, with_thinking=True, with_reasoning_effort=True)

    print("\n诊断完成。重点看：")
    print("  - 阶段 8 的 finish_reason 是 'length' 还是 'stop'？是 'length' 就是 token 预算被吃光。")
    print("  - 阶段 8 的 content len 是否远小于阶段 9/10？")
    print("  - 阶段 8 是否耗时 >60s 触发 timeout？")


if __name__ == "__main__":
    main()
