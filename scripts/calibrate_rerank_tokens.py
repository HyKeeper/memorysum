#!/usr/bin/env python
"""标定 rerank 单对预算：代理分词器（tiktoken cl100k）vs 服务端真实分词器（XLM-R）。

为什么要标定：`champion-v1.json` 的 `reranker.single_pair_proxy_tokens` 比的是
`reader_token_count()` 的**估算值**，而 llama.cpp 判定超限用的是 `bge-reranker-v2-m3`
自己的 XLM-R sentencepiece。XLM-R 对英文/URL/标点切得更细（实测最坏 real/est = 1.494），
所以按代理值放行到 8192 就会真的把 12000 token 送进服务端 → HTTP 500 → Search 503。

本脚本做两件事，并且**在配置不安全时以非 0 退出**（可当上线前预检用）：
1. 量出代理低估倍率的分布（真实 memory + query + 对抗样本）；
2. 用最坏倍率构造「刚好等于预算」的 pair，直接向服务端确认真实 token 数 ≤ 模型窗口。

用法（需要 :7997 rerank 服务在线）：
    .venv/bin/python scripts/calibrate_rerank_tokens.py [--samples 25] [--quick]
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import statistics
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import httpx  # noqa: E402

from memorysum.core.text import reader_token_count  # noqa: E402
from memorysum.settings import load_pipeline_config  # noqa: E402

RERANK_URL = os.environ.get("MEMORYSUM_RERANK_BASE_URL", "http://127.0.0.1:7997")
#: bge-reranker-v2-m3 的训练窗口，也是 `-c/-ub` 的实际上限（llama.cpp 会把更大的 -c 压回来）。
MODEL_WINDOW = 8192

ADVERSARIAL: dict[str, str] = {
    "英文散文": "The quarterly pre-incident baseline sentiment reading was 62 out of 100, "
    "which the normalisation pipeline corrected to 67 after re-weighting neutral mentions. " * 8,
    "数字为主": " ".join(f"{i * 7919 % 9973},{i * 104729 % 65521:.2f};" for i in range(400)),
    "代码": "def compute(a, b):\n    return {'x': a, 'y': b}\n" * 30,
    "URL/UUID": " ".join(
        f"https://example.com/api/v1/items/{i}?token=abc{i}def-{i * 3}" for i in range(120)
    ),
    "base64": "".join(
        random.Random(7).choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
        for _ in range(3000)
    ),
    "标点密集": "!?.,;:-—()[]{}'\"…  " * 400,
    "日文": "申請書の提出期限は来週の金曜日です。その前に必要な書類をすべて揃えてください。" * 20,
    "韩文": "신청서 제출 기한은 다음 주 금요일입니다. 그 전에 필요한 서류를 모두 준비하세요. " * 20,
    "俄文": "Срок подачи заявления — пятница следующей недели, подготовьте документы. " * 20,
    "阿拉伯文": "الموعد النهائي لتقديم الطلب هو الجمعة القادمة، يرجى تجهيز المستندات. " * 20,
    "emoji": "🙂🎉🚀🔥💡🧠📌✅❌⚠️ " * 300,
    "中文": "客户反馈希望把会议安排到周五下午，并且用中文记录所有要点，特别是关于退款流程的部分。"
    * 30,
}


def real_tokens(client: httpx.Client, text: str) -> int:
    response = client.post("/tokenize", json={"content": text}, timeout=120.0)
    response.raise_for_status()
    return len(response.json()["tokens"])


def memory_samples(data_root: Path, per_scope: int) -> list[str]:
    samples: list[str] = []
    for path in sorted((data_root / "scopes").glob("*.sqlite3")):
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT raw_content FROM messages ORDER BY random() LIMIT ?", (per_scope,)
            ).fetchall()
        except sqlite3.Error:
            continue
        finally:
            connection.close()
        samples.extend(str(content) for (content,) in rows if len(str(content)) > 120)
    return samples


def text_at_budget(client: httpx.Client, seed: str, budget: int) -> str:
    """把文本重复到**代理估算值**刚好不超过 budget（服务真正会比对的数值）。"""
    del client  # 只需要代理计数，保留参数是为了调用点对称
    high = 1
    while reader_token_count((seed + " ") * high) <= budget:
        high *= 2
    low = 0
    while low < high:
        mid = (low + high + 1) // 2
        if reader_token_count((seed + " ") * mid) <= budget:
            low = mid
        else:
            high = mid - 1
    return (seed + " ") * low


def pair_text(query: str, document: str, instruction: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def report(client: httpx.Client, name: str, samples: list[str]) -> float:
    ratios = []
    for text in samples:
        estimated = reader_token_count(text)
        if estimated:
            ratios.append(real_tokens(client, text) / estimated)
    if not ratios:
        return 0.0
    ratios.sort()
    print(
        f"{name:<22} n={len(ratios):>5} 中位={statistics.median(ratios):.3f} "
        f"p99={ratios[min(len(ratios) - 1, int(len(ratios) * 0.99))]:.3f} 最大={ratios[-1]:.3f}"
    )
    return ratios[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=25, help="每个 scope 抽多少条 memory")
    parser.add_argument("--quick", action="store_true", help="只跑对抗样本")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    pipeline = load_pipeline_config()
    budget = pipeline.reranker.single_pair_proxy_tokens
    client = httpx.Client(base_url=RERANK_URL, timeout=300.0)

    worst = 0.0
    worst_seed = next(iter(ADVERSARIAL.values()))
    if not args.quick:
        samples = memory_samples(Path(args.data_root), args.samples)
        print(f"真实 memory 样本 {len(samples)} 条：")
        worst = max(worst, report(client, "memory", samples))
    print("对抗样本：")
    for name, text in ADVERSARIAL.items():
        ratio = report(client, name, [text])
        if ratio >= worst:
            worst, worst_seed = ratio, text

    if not worst:
        print("没有采到样本（--quick 且无对抗输出？）")
        client.close()
        return 1

    print(f"\n最坏 real/est = {worst:.3f}")
    print(f"配置预算 single_pair_proxy_tokens = {budget}（代理值）")
    print(f"按最坏倍率推算最大真实长度 = {budget * worst:.0f} tokens，模型窗口 = {MODEL_WINDOW}")

    # 直接向服务端确认：用最坏倍率的那类文本顶到预算，看真实 token 是否仍落在模型窗口内。
    document = text_at_budget(client, worst_seed, budget)
    query = "What does the record say about the deadline?"
    pair = pair_text(query, document, pipeline.reranker.instruction)
    real_pair = real_tokens(client, pair)
    estimated_pair = reader_token_count(pair)
    print(
        f"\n边界验证（最坏倍率文本顶到预算）：代理估算 {estimated_pair} token -> 真实 {real_pair} token"
    )
    print(f"  真实值 ≤ 模型窗口 {MODEL_WINDOW}：{real_pair <= MODEL_WINDOW}")

    safe = budget * worst <= MODEL_WINDOW and real_pair <= MODEL_WINDOW
    print(
        "\n结论：" + ("安全（代理预算 × 最坏倍率 ≤ 模型窗口）" if safe else "不安全，请下调预算！")
    )
    client.close()
    return 0 if safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
