# exp/ — 专家评测流量的可重放导出（2026-10-14 16:35 轮）

用途：在自有评测平台上**原样重放专家的 add + search 流量**，并复用本仓库的评分口径。
数据来源：`backups/accuracy-2026-10-14T1635/`（该轮 08:31–08:35Z = 本地 16:31–16:35），
已剔除本机自己在 16:03 的验证请求。

## 文件

| 文件 | 记录数 | 内容 |
|---|---|---|
| `adds.csv` | 1495 行（来自 134 次 /add） | 一行一条 message：`timestamp, user_id, request_id, session_id, ordinal, role, message_timestamp_ms, content`。`content` 是**逐字原文**（含平台自己的 `[<suite>][session_N][D…]` 头、`Streaming online QA feedback` 回灌、`intra-transfer train trajectory` 载荷）。按 `(user_id, request_id, ordinal)` 排序即可还原一次 Add 的 messages 数组。 |
| `searches.csv` | 48 行（41 个唯一题面） | 一行一次 /search：`seq, timestamp, user_id, query, top_k, options_json, qa_id, platform_score, platform_answer`。后三列是平台官方判定（**只有 5 行有**，其余题没有回灌）。 |
| `questions.csv` | 41 行 | 唯一题面 + 本仓库的本地判定：`capability_local`（推断的子能力）、`verdict_local`（1.0/0.5/0.0）、`blame_local`（失分归因）、`note_local`、以及该题的平台判定（若有）。 |
| `platform_leaf_scores.csv` | 21 行（20 叶子 + TOTAL） | 平台四次跑分的子能力分值（A 13:25 / B 14:27 / C 15:28 / D 16:35），可直接复算聚合口径。 |
| `scoring_rubric.csv` | 9 行 | 本地判据规则 + 两个聚合口径的定义（见下）。 |

## 重放步骤

1. **灌库**：按 `adds.csv` 分组（`user_id` + `request_id`），把该组的 messages 按 `ordinal` 拼成
   `{"request_id", "user_id", "session_id", "messages":[{"role","timestamp","content"}]}` 调 `/add`。
   `content` 必须逐字发送（含 `\n`、`\t`、反斜杠转义）—— 那是平台的原始输入格式。
2. **提问**：按 `searches.csv` 的 `seq` 顺序调 `/search`（`{"query","user_id","top_k"}`，`options` 若
   非空需带上）。
3. **对分**：`questions.csv.platform_score` 有值的 5 题可做端到端对照；其余题只能比"证据是否命中"
   （本仓库的口径 ①/②）。
4. **注意**：题面顺序每轮会被平台打乱，跨轮对比一律**按题面文本**连接，不要按序号。

## 本地评分口径（`scoring_rubric.csv` 的展开）

**逐题判定**（读我们返回的证据即可判，不需要金答案）：

| 规则 | 适用 | 判定 |
|---|---|---|
| 1 | 精确值题（年份/日期/姓名/数值） | 答案原样出现在正文且与 as-of 条件吻合 → 1.0，否则 0.0 |
| 2 | 时间题（when/what time/多久/顺序） | 只有相对时间（yesterday/半个月前）而无可换算的绝对锚点 → 0.0；有锚点按能否读出答案判 1.0/0.5 |
| 3 | 选择题 | 需决定性证据 → 1.0；仅相关片段 → 0.5；无关 → 0.0 |
| 4 | 不依赖记忆的题（建议/推理/长历史综合） | 短建议题 1.0；长历史综合 0.0（用官方 E1=100 / F1=0 校准） |

**两种聚合口径**（都要报）：

- **① 证据充分性** = 41 题按题等权。衡量"我们交出的证据够不够"，是平台端到端分的**上界**
  （实测偏高 5~8 分）。
- **② 能力树口径** = 按子能力等权，观测不到的 5 片（C3/D2/E2/G2/I1，实例不经过 /search）按平台
  最近一次的值代入（跨轮极稳，四轮里 18/20 逐字相同）。**平台总分 ≈ 其 20 片叶子均值 ±1.2**（实测），
  所以这个口径才能对齐。实测精度 **±2 分**（本轮 72.23 vs 平台 71.27）。

**失分归因 `blame`**：`ok` 证据够 / `anchor` 缺时间锚点 / `retrieval` 碎片化漏召回 / `partial` 需推理 /
`memory` 库里没有或灌错 / `judge` 平台误判 / `n/a` 不依赖记忆。

**方法提醒**：A/B 要看**实例级证据**（平台回灌的 QA feedback），不要只看叶子或总分——平台自身两轮
就差 ~1 分；而叶子是 ~5 个实例的均值，单个实例翻转撼不动它（2026-10-14 实测：我们把 5 道题从
"读不出"改成"读得出"，C1/C2 叶子一片没动，总分却 +2.22，收益落在 I1）。

## 本次导出的题面分布（`capability_local`）

C1×5、B4×6、A1×4、A2×3、B1×2、B2×2、C2×2、D1×2、E1×2、E3、F1、G1×2、G4、H2、I2×4 等，共 41 题；
平台侧的 20 个子能力里有 5 片（C3/D2/E2/G2/I1）在这 41 条检索里没有对应实例。
