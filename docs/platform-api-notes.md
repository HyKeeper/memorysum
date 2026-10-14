# 平台 API 与官方文档口径（2026-10-14 建立）

**这是本仓库第一次能主动读到平台侧的数据**，而不是从 `/add` 回灌里反推。本文分两部分：
**① 我们从 API 取到的事实**（可复现，脚本在仓库里）与 **② 平台公开文档的口径**（原文摘录，
附抓取方式）。两部分的性质不同，不要混用：① 是我们量出来的，② 是平台"说"的 —— 文档与实际
行为可能不一致，凡引用 ② 必须带出处，凡引用 ① 必须带命令行。

---

## ① API：能取到什么（已实测）

取数：`.venv/bin/python scripts/fetch_platform_results.py [--leaves|--diff|--save DIR]`
（key 在 `.env` 的 `AML_LEADERBOARD_API_KEY`；认证头 `X-Leaderboard-Api-Key`）

| 端点 | 作用 | 实测 |
|---|---|---|
| `POST /access/validate` | 会话信息 | 返回 `system_name=memorysum`、`leaderboard_track=academic`、`allowed_modes=["full"]`、`cooldown_exempt=false` |
| `GET /access/versions` | 版本与版本绑定 | `version_c75181fa1e95` / `v1.0` / status `active` |
| `GET /access/version-endpoints` | **我们注册的接口地址** | `add/search/health` 都在 `http://rainhan.dns.army:10011/...`（对外端口，不是本机 8001） |
| `GET /api/v1/evaluations?limit=100` | **全部评测轮次 + 逐能力分数** | 6 轮，全部 succeeded |
| `GET /api/v1/evaluations/{id}` | 单轮详情 | 与列表项同构（无 artifacts 字段） |
| `GET /api/v1/evaluation-tracks` | 三赛道清单 | textual/coding/multimodal，contract `textual-v1` |
| `GET /api/v1/evaluation-tracks/textual/contract` | 赛道契约 | adapter 协议、options 范围、result schema |
| `GET /datasets` | 各模式的成员数据集 | smoke 10 个 / full 12 个（见下） |
| `GET /static/cycle1-{industry,academic}-textual.json` | **Cycle 1 公开榜原文** | industry 15 条、academic 50 条（full 模式历史提交） |

**取不到的**（别浪费时间试）：
- **逐题 gold / 平台答案 / 逐题判定** —— 文档明说"我们不会提供金标答案、评分依据或完整数据集下载"；
  逐题信息仍**只能**从 `/add` 回灌的 `qa_feedback` 抽（`scripts/extract_qa_feedback.py`）。
- **artifact**：smoke 轮次的 `result` 里没有 `artifacts` 字段，`/api/v1/evaluations/{id}/artifacts/*` 一律 404。
- `/eval-jobs/{id}`（legacy 引擎）对我们的 `teval_*` 一律 404；`/eval-jobs/{id}/attribution` 需要
  admin key（403）；`/eval-results/private` 返回空。
- **Cloudflare 会按指纹拦**：urllib 默认 UA（`Python-urllib/3.x`）直接 403 `error code: 1010`，
  脚本里带了正常 UA 才通（curl 默认 UA 也能过）。

### 六轮的权威分数与逐叶子（**这是以前只能猜的东西**）

北京时间（UTC+8）对齐，脚本 `--leaves` 直接输出：

```
leaf                      13:25  14:27  15:28  16:31  17:38  18:38
A1                        66.67  66.67  66.67  66.67  66.67  66.67
A2/B1/B2/B4/C3/D1/E1/E2/E3/G1  100.00（六轮逐字不动）
C1                        60.00  53.33  60.00  60.00  53.33  53.33
C2                        80.00  60.00  80.00  80.00  60.00  60.00
D2                        40.00（不动）  F1 0.00（不动）
G2                        20.00（不动）  G4 50.00（不动）  H2 50.00（不动）
streaming.online          46.43  46.43  46.43  66.67  64.29  66.67
streaming.intra_transfer  50.00（不动）
总分                      69.05  68.08  69.05  71.27  70.03  70.30
```

三条新事实：

1. **`I1` 就是平台的 `streaming.online`**（本地脚本给的别名），`I2` = `streaming.intra_transfer`。
2. **20 个叶子里只有 3 个在六轮中动过**：`C1`、`C2`、`streaming.online`。其余 17 片逐字不变 ——
   包括我们做过的所有改动（锚点、`direct_max_tokens` 640/384）**都没有触及它们**。
3. `C1/C2` 的"两档"是**单个实例翻转**：C1 是 3 题 15 个判定点（60% = 9/15，53.33% = 8/15，
   差 1 个点），C2 是 5 个判定点（80% = 4/5，60% = 3/5，同样差 1 个）。**这个 1.33/0.97 分
   的总分摆动（±0.97~1.24）不是我们改出来的，是那一题自己在两态间跳。**

### 数据集成员（`/datasets`，2026-10-14）

- **smoke（10）**：angry, locomo_refined, beam_100k, clbench_0_4k, clbench_16_32k,
  personamem_v2_32k, docpp, memtrapbench, mquake_remastered, tempreason
- **full（12）**：angry, enemy, friends, man_earth, locomo_refined, longmemeval_refined,
  longmemeval_s, clbench_0_4k, clbench_16_32k, personamem_v2_32k, beam_100k, beam_1m
- smoke 独有的 4 个（docpp / memtrapbench / mquake_remastered / tempreason）**不在 full 里**；
  full 独有的 6 个（enemy / friends / man_earth / longmemeval_refined / longmemeval_s / beam_1m）
  **不在 smoke 里** → **smoke 分数不能外推 full**，两者是不同套件。

### 能力树（官方定义，来自前端 bundle `CAPABILITY_DIMENSIONS`）

```
A Explicit Fact Recall         A1 A2 A3
B Compositional Inference      B1 B2 B3 B4
C Temporal & Event Reasoning   C1 C2 C3
D Memory Governance            D1 D2 D3 F1     ← F1 挂在 D 下面（不是笔误）
E Personalization & Care       E1 E2 E3
G Context Learning & Execution G1 G2 G3 G4 G5
H Safety & Privacy             H1 H2
streaming                      online / intra_transfer（流式赛道，独立于 A–H）
```

## ② 平台文档口径（原文摘录；抓取方式见文末）

### 无静默截断（**推翻仓库里那个"4096 截断"旧前提的权威依据**）

> No silent API truncation — The API contract sets no single global character cap. Add and Search
> payloads and archived responses are not silently shortened; unsupported payloads should receive
> a clear 4xx response.

> 返回顺序和数量：我们保留接口返回的顺序；返回数量不得超过 `top_k`。超量或存在非法条目会判为
> 契约错误，**不会静默截断后继续评分**。

### Answer 的上下文预算（**这是真正存在的那个上限**）

> Answer context budget — The shared **128,000-token** Answer window reserves **8,192** output
> tokens and **2,048** safety tokens, leaving **117,760 input tokens**. If needed, Answer keeps a
> **token-counted prefix of Search candidates in returned rank order**; the question, options,
> instructions, and archived Search response remain intact.

对我们的含义（**可算**）：46 道真实题里最长的响应是 17,724 cl100k token（中文医疗 scope），
中位 2,162（英文）。**离 117,760 差 6.6 倍**，即我们从没触发过前缀截断；而且截断若发生，砍的是
**排名靠后的尾巴**（"prefix ... in returned rank order"），不是随机位置。

### 配额与时间线（**影响实验节奏**）

| 项 | 值 |
|---|---|
| smoke | **每小时 1 次**；本届每赛道最多 **30** 次 |
| full | 本届每个 key、每赛道**最多 2 次**；第 2 次须在首次 full 完成后 **30 天** |
| 公榜准入 | 必须**成功完成的 full**；所有任务成功；Answer 模型/评测契约/pipeline hash/dataset hash/题量完整且与现行基线一致；去重 + 审核 |
| Cycle 2 提交截止 | 2026-10-31 23:59 (UTC+8) |
| Cycle 2 评测关闭 | 2026-11-04 23:59 |
| 官方结果 | 2026 年 11 月中旬 |

**已用它解释了我们的历史节奏**：13:25 / 14:27 / 15:28 / 16:31 / 17:38 / 18:38 六轮，间隔
62 / 61 / 63 / 67 / 60 分钟 —— **不是我们想跑就跑，是"每小时 1 次"卡出来的**。
已用 6/30 次 smoke。**full 只剩 2 次，必须留到"证据充分、无可再改"时用。**

### 其它与我们直接相关的条款

- **Add 分段规则**（解释了我们看到的 chunk 边界）："普通文本在达到 **20 条消息**或 **2,000 个词**
  中的任一边界时确定性分段，词数按冻结 Adapter 计数"。
- **Add 必须同步落库**：HTTP 200 之前必须持久化且**立即可检索**；202 仅在绑定了 Add Status 地址时合法。
- **`request_id` / `user_id` / `session_id` 必须原样回显**，`success=true`；否则**即使 HTTP 200 也立即判失败**。
- **Search 不发送** `filters` / `rerank` / `keyword_search`；`options` 只在选择题出现且**不含金标**。
- **数据责任**：评测数据及其派生副本**只能用于完成当前任务**，不得用于训练/微调/产品分析/数据集重建/对外传播；
  **任务完成后 30 天内删除**（`backups/`、`logs/` 里的评测正文都在这个范围内 —— 提交 full 前要清点）。
- **URL 禁止**私有/回环/链路本地地址，也不得内嵌凭据。我们注册的对外地址是
  `http://rainhan.dns.army:10011`，**是 http 不是 https** —— 文档只是"建议"HTTPS，未见强制。

---

## 抓取方式（可复现）

```bash
# 文档正文（SPA，正文嵌在首页 HTML 里）
curl -sS https://agentmemoryleaderboard.ai/api-guide -o /tmp/aml_index.html
python3 - <<'PY'
import re
h = open('/tmp/aml_index.html', encoding='utf-8', errors='replace').read()
t = re.sub(r'<script.*?</script>', ' ', h, flags=re.S)
t = re.sub(r'<[^>]+>', ' ', t)
open('docs/platform-doc-text-2026-10-14.txt', 'w').write(re.sub(r'\s+', ' ', t))
PY

# 前端 bundle（能力树、数据集卡片、路由表都在这里）
curl -sS "https://agentmemoryleaderboard.ai/static/app.js?v=blog-production-20261001-email-preflight-20261003" -o /tmp/aml_app.js

# 结构化的历史分数
.venv/bin/python scripts/fetch_platform_results.py --leaves
.venv/bin/python scripts/fetch_platform_results.py --save backups/platform-api
```

原文快照：`docs/platform-doc-text-2026-10-14.txt`（55k 字符，含中英双版，含 ①② 全部引文）。
