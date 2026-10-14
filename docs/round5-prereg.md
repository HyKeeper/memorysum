# 第 5 轮预注册：锚点全量覆盖（`time_anchor_scope=all`）

> 平台上一轮 71.27（2026-10-14T16:35）。本轮**单变量**：只把时间锚点的覆盖面从
> 「时间/顺序类问题」扩到「所有问题」。别的什么都不动（`direct_max_tokens` 保持 384、
> 两段式输出保持关闭、ids/预算/rerank 全部不变）。

## 为什么赌这一件事

- **算术**：总分 ≈ 20 片叶子均值。71.27 → 73 需要 +1.73 分 ≈ 叶子总和 +35 点 ≈
  **2-3 个实例翻转**（叶子是 2-5 个实例的聚合，单实例值 20-33 点）。
- **唯一被实例级证明的机制就是锚点**：`locomo1:q0000:step01` 连续三轮 0.0
  （答题模型答 "Yesterday."），16:35 轮锚点开启后判 1.0，答案
  "Jon lost his job as a banker on 19 Jan, 2023" = 我们锚点 `(20 Jan, 2023)` 减一天；
  对应叶子 **I1 46.43 → 66.67（+20.24）**。
- **但覆盖面太小**：`is_temporal_query` 在 52 条观测 query 里只命中 14 条。吃锚点的
  C1(60)/C2(80)/I1(66.67) 三片叶子合计 100+ 叶子分点，**全是时间/顺序家族**，而我们在
  41 条可观测 query 里只看到其中 7 条实例 —— 实例集在观测之外。
- **扩词表是猜测，全量不是**：新增了 `groups.time_anchor_scope` 取值
  `"temporal"`（旧行为，默认）/ `"all"`。本轮用 `"all"`。

## 本地投影（46 道题，同一批数据重跑两种配置）

| 指标 | temporal | all |
|---|---|---|
| 有锚点行的返回记录 | 445/1162（38%） | **1135/1162（98%）** |
| 响应 token 总量 | 136 406 | 142 991（**+4.8%**） |
| 返回记录条数 | — | **逐题相同（差 0）** |
| 响应 > 4096 cl100k token | 4 题 | 8 题 |
| `eval_retrieval.py` ids_all | 39/60、49/60、MRR 0.746 | **逐字相同** |
| `eval_retrieval.py` txt_all | 47/60、MRR 0.855 | 46/60、MRR 0.8465（同配置重跑的噪声） |

- 顺序错位 6/46 题 —— **固定配置重跑也有 4-6/46 题错位**，即运行间非确定性，与锚点无关。
- 回归锚点：`locomo1` 必须继续判 1.0（若它回落，说明锚点机制本身坏了）。

## 预注册的判据

**成功信号（预注册"该动的叶子"）**：
1. **C1 / C2 / I1 三片里至少一片的叶子分变化**（这是本轮的靶子）；
2. 实例级证据优先：新增/变化的 QA feedback（`extract_qa_feedback.py`）；
3. `locomo1` 保持 1.0。

**失败信号**：
- C1/C2/I1 全部逐字不动，且实例级证据没有任何新翻转 → 覆盖面不是瓶颈，转向
  `direct_max_tokens` 640 的单变量实验。

**已知成本（接受）**：
- 响应 token +4.8%（每条多一行锚点）。锚点行在**每条记录的前面**，在任何截断假设下都
  不会被切掉；而**平台会不会截断我们没有口径**（详见 AGENTS.md 第 37 条，那条「4096
  限制」是错误前提，已删）。
- 若平台把这一行当噪声（理论上不会：它保留同形态的 `Session N (…)`），最坏回到 16:35
  轮的状态，不亏内容。

**噪声警告**：平台自身两轮差 0.97（C1 53.33↔60、C2 60↔80 是唯二会跳的叶子）。
所以**单看总分不足以判成败**，必须落到实例级证据与预注册的叶子。

## 回滚

一行配置：`champion-v1.json` 的 `groups.time_anchor_scope` 改回 `"temporal"`，重启即回到
16:35 轮的行为（该字段默认值就是 `"temporal"`，`test_time_anchor_scope_stays_declared_and_typed`
守护它在 config 里显式存在、非法值报错）。

## 跑分后处置（跑分前不要再重启/改配置，否则指纹不干净）

```bash
# 切日志（锚点行在 backups/marker-*.json）→ 抽流量与平台信号 → 快照 scope
.venv/bin/python scripts/extract_loadtest_requests.py <切片> backups/round5/requests.jsonl
.venv/bin/python scripts/extract_qa_feedback.py <切片> backups/round5/platform_signals.jsonl
.venv/bin/python scripts/backup_scopes.py backups/round5
# 本地判定（先机械规则，人/agent 补余下）→ 双口径 + 逐叶子对齐
.venv/bin/python scripts/dump_question_evidence.py backups/round5/requests.jsonl
.venv/bin/python scripts/eval_local_score.py <verdicts> --tree <树>
.venv/bin/python scripts/eval_alignment.py <verdicts> <树> --record
```

**先看两件事，再谈总分**：
1. **实例级**：`platform_signals.jsonl` 里 4 条 QA feedback 的 `score` 变化。
   `locomo1` 必须仍是 1.0（回归）；任何**新翻正的实例**都是本轮的功劳，
   任何原为 1.0 却翻负的实例优先归因为噪声（先与上一轮逐题对齐）。
2. **预注册叶子**：C1/C2/I1 是否至少一片变化。注意叶子是 ~5 个实例的均值，
   单实例翻转撼不动它 —— **叶子不动而实例变了也算成功**（16:35 轮的教训）。

**本轮需要额外记的一笔**：`<切片>` 里所有 /search 的 `returned_chars`（对比 17:42 轮的同一批题）。

> **2026-10-14 17:52 更正**：本轮之后把 `direct_max_tokens` 从 384 改回了 **640**（依据：「平台响应受 4096 cl100k token 限制」是错误前提，**我们对其没有任何口径**）。所以如果下一轮要跑，与第 5 轮的**唯一差异是 `direct_max_tokens`**（锚点保持 all）——靶子/判据不变，但要把「平台截断」从任何推理里拿掉：**我们交出了多少内容**是事实，**平台看到多少**我们没有口径。
