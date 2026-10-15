# 本地 200 题基准测试

**目的**：替代平台 smoke 赛道的 ~30 题噪声评分，用自己控制的 200 题数据集衡量检索质量。
平台每轮 G4/C1/C2 的二项噪声 ±3.4 分，本地 200 题可将标准误降到 ±1 以内。

## 文件

- `generate_questions.py` — 从 scope 消息中提取事实，生成 QA 对
- `run_benchmark.py` — 跑基准、自动评分、输出逐叶子指标
- `benchmark_v1.jsonl` — 生成的题库（每行一个 QA 对）
- `results/` — 历次评测结果

## 题库格式

```json
{
  "id": "A1-001",
  "leaf": "A1",
  "scope_hash": "abc123...",
  "question": "What is the speaker's name?",
  "answer_substrings": ["John Smith"],
  "score_type": "substring",
  "evidence_msg_ids": [3]
}
```

## 运行

```bash
# 1. 生成题库（从现有 scope 提取）
.venv/bin/python bench/generate_questions.py --target 200

# 2. 跑基准
.venv/bin/python bench/run_benchmark.py

# 3. 对比两次运行
.venv/bin/python bench/run_benchmark.py --compare results/previous.json
```

## 评分口径

- `substring`: 所有 answer_substrings 都出现在搜索结果的正文中
- `temporal`: 时间锚点 + 答案都要出现
- `coverage`: 关键事实覆盖率（F1 摘要类）
- `multi_evidence`: 多条证据都要出现（B1/B2 多跳类）
