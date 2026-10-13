#!/usr/bin/env python
"""检索质量评测器：对已灌入的 BEAM scope 跑 60 条英文事实查询。

用法（在仓库根目录）：
    .venv/bin/python scripts/eval_retrieval.py                     # 全量与 split-half
    .venv/bin/python scripts/eval_retrieval.py --user-id <uid>     # 换一个 scope
    .venv/bin/python scripts/eval_retrieval.py --dump-rank         # 打印每条查询的排名

设计要点：
- **离线运行**：直接实例化 SearchService（读 DB + 调本地模型服务），不经过 HTTP，
  因此不会重启或干扰正在运行的 memorysum。
- **gold 判定**：返回的原文片段（空白规范化后取前 100 字符）必须是 gold 消息原文的
  子串——Search 返回的是逐字原文切片，所以这是精确判定，不依赖模糊匹配。
- **必须看 split-half**：只报全样本净值会掩盖"两半相抵"的假象（实测纯 rerank 排序
  全样本净 +1，实际两半为 +4 / −3）。

依赖：BEAM exp0 数据已灌入（见 README 的 Add 示例），且本地 embedding(:7999) /
rerank(:7997) 服务可达。缺 rerank 服务时把 config 的 reranker.enabled 置 false 仍可评测。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"), override=False)

from memorysum.api.schemas import SearchRequest  # noqa: E402
from memorysum.providers.factory import build_provider, embedding_endpoint  # noqa: E402
from memorysum.services.search import SearchService  # noqa: E402
from memorysum.settings import RuntimeSettings, load_pipeline_config  # noqa: E402
from memorysum.storage.database import DatabaseManager  # noqa: E402

DEFAULT_USER = (
    "agentic_memory_http_memory_beam_128k_chunk4096_runcb1793d1-3c47-43e5-b319-1a5c68e2087f_exp0"
)

# (问题, gold 消息的 message_id, 人工核对用的独特标记)
CASES: list[tuple[str, int, str]] = [
    ("How fast is the dashboard API after the caching improvements?", 97, "250ms"),
    ("What gunicorn command is used for deployment?", 159, "gunicorn -w 4"),
    ("Which pre-commit hooks are configured for the project?", 135, "psf/black"),
    ("How are edge cases and security vulnerabilities tested?", 117, "Test Suite Enhancements"),
    ("Where are the SSL certificates obtained for deployment?", 121, "Let's Encrypt"),
    ("When was the user finalizing the deployment?", 112, "April-25-2024"),
    (
        "How is the dark mode toggle styled in the navbar?",
        144,
        "navbar-dark .navbar-nav .nav-link:hover",
    ),
    ("What is the maximum password length validation?", 149, "Password cannot exceed"),
    (
        "How to fix a memory leak related to session management?",
        154,
        "memory leak in your Flask application",
    ),
    ("What does the least privilege principle require?", 173, "Least Privilege Principle"),
    ("How is the microservices architecture class structured?", 168, "MicroservicesArchitecture"),
    ("What does the API endpoints documentation page include?", 165, "Table for API Endpoints"),
    (
        "Which packages are installed when initializing the project?",
        101,
        "pip install flask flask-login",
    ),
    ("What database URI is configured in the Flask app?", 104, "SQLALCHEMY_DATABASE_URI"),
    ("What test checks the fetched transaction description?", 108, "fetched_transaction"),
    ("Which logging levels are demonstrated in the index route?", 86, "logging.critical"),
    ("How is the chart tooltip configured to show on data points?", 81, "tooltip.intersect"),
    ("Which decorator restricts the admin route to users?", 77, "@role_required"),
    (
        "How does selective column retrieval improve query efficiency?",
        73,
        "Selective Column Retrieval",
    ),
    ("What colors are defined for the pie chart?", 69, "rgba(255, 99, 132"),
    ("What options does the transaction type dropdown offer?", 64, "Select Type"),
    ("How does the register endpoint handle JSON input validation?", 60, "request.get_json()"),
    (
        "What is reviewed for handling the transaction creation process?",
        56,
        "transaction creation process correctly",
    ),
    ("Which Flask version is pinned in requirements.txt?", 49, "Flask==2.0.1 Flask-SQLAlchemy"),
    ("What does run.py contain in the blueprint structure?", 45, "This file runs the Flask app"),
    (
        "How is user registration implemented with hashed passwords?",
        42,
        "password_hash = db.Column(db.String(120)",
    ),
    (
        "What tasks are listed for a secure registration system?",
        38,
        "secure and functional user registration",
    ),
    ("Which commands initialize database migrations?", 32, "flask db init"),
    (
        "How to reduce Flask application response times?",
        25,
        "reduce response times and improve overall performance",
    ),
    ("How are transactions inserted into the SQLite database?", 16, "INSERT INTO transactions"),
    ("Which SQLite file name is used in the dev database config?", 2, "sqlite:///budget.db"),
    (
        "What error is returned when username or password is missing?",
        6,
        "Username and password are required",
    ),
    ("What does the base template use for styling?", 10, "use Bootstrap for styling"),
    ("How does the income form submit its amount?", 13, "url_for('add_income')"),
    ("What input type is used for the amount field?", 18, 'id="amount"'),
    ("How to improve Flask session handling security?", 23, "handle user sessions more securely"),
    ("How to test the JSON response for a 500 internal server error?", 26, "Internal server error"),
    ("How to define and migrate a new Category model?", 30, "class Category(db.Model)"),
    (
        "What is being optimized for the dev environment transaction endpoints?",
        35,
        "transaction CRUD endpoints in the dev environment",
    ),
    ("Which fields should be indexed to speed up queries?", 39, "'date' and 'user_id' fields"),
    ("What does the login endpoint return on success?", 43, "some_token"),
    ("What should be documented during integration testing?", 47, "Document Issues"),
    ("What does the login form template contain?", 52, "imple login form"),
    ("Which Flask-Login version is integrated?", 58, "Flask-Login v0.6.2"),
    ("Why are wildcard LIKE searches slow?", 62, "Avoid Wildcard Searches"),
    ("What query should return the expected transaction results?", 66, "Transaction.query.all()"),
    ("How are transactions filtered for the current month?", 71, "current_month"),
    ("How is the transaction date parsed when updating?", 75, "datetime.strptime"),
    ("How is OperationalError handled in the DB call wrapper?", 79, "except OperationalError as e"),
    (
        "What tests are written for password hashing and login validation?",
        84,
        "pytest tests for my auth.py",
    ),
    ("What columns does the transaction table define?", 89, 'ForeignKey("user.id")'),
    (
        "How can async views improve I/O-bound task handling?",
        94,
        "Flask supports asynchronous views",
    ),
    ("Which library provides datetime support for the form dropdowns?", 98, "Flask-Datetime"),
    ("Where are the route unit tests written?", 102, "tests/test_routes.py"),
    ("What does the logout test assert in the response?", 106, "Logout successful!"),
    ("How is a transaction created with a note field?", 110, "transaction = Transaction("),
    (
        "What is reviewed along with the Gunicorn configuration?",
        115,
        "review both your Gunicorn configuration",
    ),
    ("What is the Gunicorn worker timeout setting?", 119, "timeout = 120"),
    ("What hashing implementation is used in the login endpoint?", 124, "bcrypt"),
    ("Where is Chart.js loaded from?", 129, "Include Chart.js from CDN"),
]


def normalise(text: str) -> str:
    return " ".join(text.split())


def build_service(user_id: str):
    runtime = RuntimeSettings.from_environment()
    pipeline = load_pipeline_config()
    space = hashlib.sha256(
        (pipeline.embedding_space_id + "\n" + embedding_endpoint(runtime, pipeline)).encode()
    ).hexdigest()
    database = DatabaseManager(runtime.data_root, pipeline, space)
    provider = build_provider(runtime, pipeline)
    return SearchService(database, provider, pipeline), provider, database


def load_gold(database, user_id: str) -> dict[int, str]:
    reference = database.resolve_scope(user_id, create=False)
    if reference is None:
        raise SystemExit(
            f"找不到 user_id={user_id} 的 scope；先按 README 的 Add 示例把数据灌进去。"
        )
    connection = database.connect(reference, user_id)
    rows = connection.execute("select message_id, raw_content from messages").fetchall()
    connection.close()
    return {int(r["message_id"]): normalise(str(r["raw_content"])) for r in rows}


def evaluate(service, user_id: str, gold: dict[int, str], dump: bool) -> tuple[list, list]:
    ranks, latencies = [], []
    for query, gold_id, _marker in CASES:
        started = time.perf_counter()
        response = service.search(SearchRequest(query=query, user_id=user_id, top_k=10))
        latencies.append(time.perf_counter() - started)
        target = gold.get(gold_id, "")
        rank = None
        for index, item in enumerate(response.data, start=1):
            head = normalise(item.content)[:100]
            if head and head in target:
                rank = index
                break
        ranks.append(rank)
        if dump:
            print(
                f"    {'OK  ' if rank == 1 else (f'@{rank:<3d}' if rank else 'MISS ')} {query[:66]}"
            )
    return ranks, latencies


def metrics(ranks: list) -> tuple[int, int, float]:
    top1 = sum(1 for r in ranks if r == 1)
    top3 = sum(1 for r in ranks if r and r <= 3)
    mrr = sum(1.0 / r for r in ranks if r) / len(ranks)
    return top1, top3, mrr


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval quality evaluation")
    parser.add_argument("--user-id", default=DEFAULT_USER)
    parser.add_argument("--dump-rank", action="store_true", help="打印每条查询的排名")
    parser.add_argument("--no-split-half", action="store_true", help="跳过 split-half")
    args = parser.parse_args()

    pipeline = load_pipeline_config()
    reranker = pipeline.reranker
    print(f"pipeline   : {pipeline.pipeline_id}")
    print(f"embedding  : {pipeline.embedding.provider} / {pipeline.embedding.model}")
    print(
        f"reranker   : {reranker.provider} / {reranker.model} / enabled={reranker.enabled} "
        f"/ fusion_weight={reranker.fusion_weight}"
    )
    print(f"user_id    : {args.user_id}")
    print(f"queries    : {len(CASES)}\n")

    service, provider, database = build_service(args.user_id)
    try:
        gold = load_gold(database, args.user_id)
        started = time.perf_counter()
        ranks, latencies = evaluate(service, args.user_id, gold, args.dump_rank)
        elapsed = time.perf_counter() - started
    finally:
        provider.close()

    top1, top3, mrr = metrics(ranks)
    print(
        f"\ntop1={top1}/{len(CASES)}  top3={top3}/{len(CASES)}  MRR={mrr:.4f}  "
        f"({elapsed:.0f}s, p50={sorted(latencies)[len(latencies) // 2] * 1000:.0f}ms)"
    )

    if args.no_split_half:
        return
    half = len(ranks) // 2
    print("\nsplit-half（只报全样本净值会被两半相抵的假象骗到）:")
    for label, part in (("前半", ranks[:half]), ("后半", ranks[half:])):
        a, b, c = metrics(part)
        print(f"  {label}  top1={a}/{len(part)}  top3={b}/{len(part)}  MRR={c:.4f}")


if __name__ == "__main__":
    main()
