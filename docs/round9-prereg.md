# 第 9 轮预注册：RRF k 值重标定 + 候选预算按平台口径校准

> 状态：**配置已就绪、未跑分**（`23e18c4ad423`，服务已重启）。本文是下次跑 smoke 的依据。
> 上一轮（第 8 轮 `b967d8edfd68`）也**从未跑过** —— 所以这次的变量是两轮改动的**合并**，
> 逐项依据都在下面，每项都可单独回滚。

## 一、相对最后一次跑分（18:38 轮 `def92ae229b1`）的改动

| # | 改动 | 旧 → 新 | 依据 | 本地验证 |
|---|---|---|---|---|
| 1 | **`rrf_k`** | 60 → **20** + `rrf_k_adaptive=True` | RRF 的 k=60 来自 2009 年 TREC 的数十万文档语料；k 越大排名差越小（k=60 时 top-1/2 只差 0.0003），而我们的候选只有 40~254 条。见 [那篇分析](https://robertsong2019.github.io/posts/rrf-k60-wrong-for-agent-memory-2026-06.html) | **强**：两指标同时改善，**两次独立跑数字相同** |
| 2 | `reranker.ordinary_candidate_groups` | 64 → **100** | 289 条消息的 scope 恒被截在"候选 184 / 组 64" | 中：内容中位 +35% |
| 3 | `reranker.ordinary_pair_proxy_budget` | 24000 → **48000** | 配套（与 wide 对齐） | 同上 |
| 4 | `retrieval.ordinary.output_records` | 64 → **100** | 与 wide / 平台 `top_k=100` 对齐 | 中性 |
| 5 | `retrieval.ordinary.output_tokens` | 32768 → **65536** | 平台 Answer 预算是 117,760 token，我们只用 ~3% | 中性（BEAM 上 no-op） |
| 6 | `retrieval.wide.output_tokens` | 49152 → **98304** | 同上 | — |
| 7 | **请求 schema `extra`** | `forbid` → **`ignore`** | 平台会带未声明字段（`session_id`/`filters`/`metadata`/…），旧行为直接 422；平台文档说 422 **不重试 + 当前阶段立即失败** | **机制确证**：全天 161 次 422，**round 5 窗口内 12 次** |

**未纳入但机制已就位**：`retrieval.channel_consensus_bonus`（默认 0，乘法共识奖励）。

## 二、本地证据（`scripts/eval_retrieval.py`，60 查询 + BEAM scope）

| 指标 | 历史基准 | 第 8 轮 | **第 9 轮（`23e18c4ad423`）** |
|---|---|---|---|
| txt_all top1 | 47 | 48 | **49** |
| txt_all third | 55 | 56 | 56 |
| txt_all MRR | 0.855 | 0.8677 | **0.8779** |
| ids_all top1 | 39 | 37 | 37 |
| ids_all top3 | 49 | 49 | 49 |
| ids_all MRR | 0.745 | 0.7238 | **0.7297** |
| split-half（txt top1） | 26/27 | 25/23 | **24/25** |

**k 值扫描全表**（同一批查询，两半都报）：

| k | txt1 | txt3 | txtMRR | ids1 | ids3 | idsMRR | half |
|---|---|---|---|---|---|---|---|
| 10 | 47 | 55 | 0.8606 | 36 | 49 | 0.7181 | 23/24 |
| 15 | 48 | 55 | 0.8688 | 37 | 48 | 0.7250 | 23/25 |
| **20** | **49** | 56 | **0.8779** | 37 | 49 | **0.7297** | **24/25** |
| 25 | 48 | 57 | 0.8716 | 36 | 49 | 0.7237 | 23/25 |
| 60（旧） | 48 | 56 | 0.8677 | 37 | 49 | 0.7238 | 25/23 |
| 自适应 | 49 | 56 | 0.8785 | 37 | 49 | 0.7321 | 24/25 |

## 三、预注册判据

| 观察 | 结论 |
|---|---|
| 总分 **> 71.27**（历史最高） | k 值重标定 + 422 修复有效 → 继续沿"融合/契约"轴 |
| 落在 69~71.3，叶子仍不动 | 交付与融合都已饱和 → 只剩"证据形式"轴（前缀/渲染），但那需要平台配合验证 |
| **< 69**（低于六轮最低） | 回滚到 `def92ae229b1`（`git checkout` 该文件 + 重启） |
| **实例级**：`locomo1` 不是 1.0，或 QA feedback 里出现新的失败模式 | 优先排查，而不是看总分 |

**噪声提醒**：平台自身两轮差 ~0.97；`C1/C2` 同配置也会跳；本地运行间也有 ~12% 的 top-1 翻转。

## 四、下次跑分之后要继续的实验队列（都在 k=20/adative 新基线上）

1. **`channel_weights.leaf_dense` 2.0 → 2.5**（**被中断，未测完**；旧 k 下曾显示 ids1=38）
2. **`similarity_weight` 0.05 / 0.1**（k 变小后，相似度项的相对权重变了）
3. **`fusion_weight` 在 k=20 下重扫**（旧结论 2.0 最优是在 k=60 下测的）
4. **`term_coverage_weight` 0.2 / 0.4**（第 20 条说 MRR 稳定小升，当时没上是因为 Recall 优先）
5. **`speaker_weight`**（第 16 条说小样本有效、全量噪声 —— k 变了可复测）
6. **窗口粒度**（leaf/ctx 的 target/max）—— **必须 bump `pipeline_id`**，留到最后

## 五、已否决的方向（别再花时间，附实测）

| 方向 | 结果 |
|---|---|
| full_bundle 对普通查询开放 | **答案排名崩坏**：locomo 时间题 `19 January 2023` 从第 1 掉到**第 19**，年份题 1→16。原因：整条消息（900+ token）稀释了 rerank 信号，而窗口（256 token）语义聚焦 |
| `fusion_weight` 1.5 / 3.0 | 都差（ids1 37→35） |
| `chinese_fts` 权重 0.5 / 0.75 | 噪声级（两次跑 39/38），且 txt3 掉 2 |
| `channel_consensus_bonus` 0.15 / 0.30 | 0.15 让 txt1 49→48，0.30 更差 |
| 通道召回扩到 wide 参数 | 两项都降 |
| `direct_max_tokens` 1280 | txt1 +2 但 **ids1 −3**（条数被挤掉）→ 保留 640 |
| `per_source_cap=2` | 平台逐叶子 0 变动 |

## 六、平台侧死结（改我们这边无解，见 AGENTS.md 第 40/41 条）

`F1=0.00%`（只灌了 4630/9050 字符）、`intra_transfer`（task index 灌错）、
PersonaMem（题面与库里的档案不符：问 1985 库是 1986、问 Microsoft 库是 Oracle、middle name 查不到）。
**这三类约占去可观测失分的三分之一，且与检索/排序无关。**

## 七、回滚

```bash
git checkout def92ae229b1 -- src/memorysum/config/champion-v1.json src/memorysum/services/search.py \
    src/memorysum/core/ranking.py src/memorysum/settings.py src/memorysum/api/schemas.py
# 然后按 AGENTS.md「本机环境的坑」里的方法重启（用 ss 反查 PID，确认 PID 变了）
```
