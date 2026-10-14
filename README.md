# MemorySum

面向记忆检索系统开发的原文记忆服务，提供与 AML Text Add / Search 兼容的 API 形状。当前仓库侧重本机开发与真实服务链路验证，不代表任何比赛成绩。

> **当前算法版本：`memorysum-hybrid-preview-v0.5.0`（provisional）**
> **embedding**：本机 llama.cpp 的 `Qwen3-Embedding-4B-Q8_0.gguf`（2560 维、`--pooling last`；`MEMORYSUM_EMBEDDING_BASE_URL` 默认 `http://127.0.0.1:7999/v1`。Q8_0 由 `llama-quantize` 从 F16 转出，7.5GB→4.0GB，显存 8010MiB，与 F16 的向量 cos 0.998）。
> **rerank**：本机 `bge-reranker-v2-m3`（`MEMORYSUM_RERANK_BASE_URL` 默认 `http://127.0.0.1:7997`，走 llama.cpp `--reranking --pooling rank`，900MiB），当前 `reranker.enabled=true`。
> **排序**：5 路召回（`leaf_dense 96 / context_dense 48 / english_fts 80 / chinese_fts 24 / options_lexical 16`）→ RRF（`rrf_k=60`，通道权重 `leaf_dense 2.0 / context_dense 1.0 / english_fts 0.5 / chinese_fts 0.25 / options_lexical 1.0`）→ **精排排名与 RRF 排名再融合**（`score = 1/(k + rerank_rank) + w/(k + retrieval_rank)`，`fusion_weight=2.0`）。冻结规格要求的 text-embedding-v4 512 维稠密+稀疏向量尚未接入（无 sparse 通道）；context 窗口仍为 target 320 / max 384 proxy tokens（当初为适配 LFM 路由而缩窄）；当前版本不能描述成已通过正式模型验收。

> **实测边界（2026-10-13，BEAM 单 scope 1147 窗口 + 60 条英文事实查询，hard negative 密集）**
> 基线（纯 RRF）top-1 34/60、top-3 44/60、MRR 0.664。
> **有效的两个杠杆**：① 通道加权（同上，top-1 +21%）；② **精排与 RRF 的排名融合**（top-1 38/60、top-3 48/60、MRR 0.733，线上复测 37/60-48/60-0.721，代价是每次检索约 +2 秒）。
> **实测无效或有害**：`rrf_k` 取 10–200（对 top-1/top-3 无影响）、embedding 输入空白压缩（端到端 net 0）、窗口粒度放大或缩小（都掉到 29/60）、通道配额收紧、1024 维 MRL 截断（几乎无损但**只省存储不省显存**）、去掉精排输入的渲染前缀（top-5 完全一致）。
> **关键教训**：`reranker.enabled=true` 时**不要用纯 rerank 排序**——实测两个半样本分别 +4 与 **−3**，全样本"净 +1"是两半相抵的假象；**任何"用新信号取代旧信号"的改动都必须做 split-half 验证**。

> **配置不变量**：`champion-v1.json` 只保留被代码真正读取的字段（已删除 `features` 整段、`language`、`generation_enabled`、`embedding.sparse_output`、`reranker.tokenizer`/`tokenizer_is_approximate`、`chunking.token_proxy`、`retrieval.unused_quota_transfer`/`sparse_channel`、`coverage.reader_estimator` 等纯装饰字段）；`capacity.max_add_messages`/`max_search_options` 由 `api/schemas.py` 直接读取，不再硬编码。改 embedding provider/model/instruction/dimension 或 window renderer **必须新建 `pipeline_id`**（否则旧 scope 一律 503 `pipeline_mismatch`）；只改排序参数（通道权重、融合权重、rerank 开关）则不必，已有数据仍可复用。

> **Search 返回 harness 标识（2026-10-13）**：`SearchItem` 新增 **`source_ids`**——该命中涉及的各 chunk 级 id 之并集（保序去重；LoCoMo 形如 `D1:3`、BEAM 形如 `1:msg7`、PersonaMem 形如 `m3`），来自 Add 时对每个 chunk 元数据头的一次性解析（`core/metadata.py`，只读、不改写原文）。返回正文若不含该 chunk 的**时间锚点**（如 `Session 1 (1:56 pm on 8 May, 2023)`），会在首行补上——**不用 `Time:` 前缀**（平台会把以 `Time:` 开头的行整行删掉）。动机：平台优先用 item 上的 id 定位 gold，只有兜底才解析正文里的 META 头；而正文含头的窗口只占少数（实测 LoCoMo：3.4 窗口/chunk、27/93 个 leaf 窗口带头），把 id 显式返回可解除这个机制上限。**注意 ids 是 chunk 级**（头里只有 chunk 级 id，无法精确到 turn），因此比窗口粒度宽。
>
> **schema v3 就地迁移**：老 scope 在首次打开时自动补上 `metadata_ids` / `metadata_anchor` 两列并从原文回填，**无需重建数据**（`database.py:_migrate_v2_to_v3`，幂等，中断后可重试）。

> **检索质量怎么量（2026-10-13 建立，2026-10-14 修判据）**：`scripts/eval_locomo.py` 用 LoCoMo 官方标注（`locomo10.json` 的 question / answer / evidence / category）对已灌入的 scope 打分，离线跑 SearchService，不影响线上服务。两个口径一起看：**`ids_all`**（返回项的 `source_ids` 覆盖 gold evidence 的比例，即平台的 Recall 口径）和 **`txt_all`**（gold turn 原文真的出现在返回正文里，决定 accuracy 上限）——两者之差就是 chunk 级 id 粒度掩盖的部分（BEAM 上实测 39/60 vs 47/60，差的那些是没带 META 头的消息）。
>
> **判据不要绑在正文头部**：响应正文允许带渲染前缀（尤其时间锚点，见 AGENTS.md 第 9 条），所以 `txt_all` 用「正文里含 ≥60 字符 gold 逐字片段」而不是「前 100 字符」。实测教训：锚点上线后旧判据把 BEAM 的 top-1 从 38/60 砸到 13/60，而精排名次根本没变（`attach_anchor` 改成恒等即回到 0.718 MRR）。指标异常时先做「检索→融合→精排→输出」四段链路追踪，再决定改排序还是改度量。
>
> `--budgets 0,4096,8192` 模拟平台按 token 截断：**条目更大和条目更多在预算内互斥**，`groups.direct_max_tokens` 从 640 降到 384 能让 4096 预算多塞 5 条记录（ids_all 86.7%→90.7%），但不截断时反而少给内容。改任何排序参数都要做 **split-half** 和 **跨对话交叉验证**（用 `locomo10.json` 按 `[LOCOMO_META chunk_id=… source_ids=…]` 格式自建 scope 灌入即可）。
>
> **本轮结论**：唯一经三折验证的改进是 `retrieval.per_source_cap = 1`（每个 chunk 最多返回 1 条，避免一个话多的 chunk 占满 top-k，给多跳题的第二处证据腾位置），三对话 ids_all +2.5 / 0 / +0.7pp，无负向。相似度分数加权、chunk 共识投票、说话人加权、邻居扩展、规则式多查询扩展均已实现但**默认关闭**——它们只在 80 题小样本上有效，全量或跨对话后回到噪声甚至转负；把 `context_dense` 权重上调到 2.0 属于典型的单对话过拟合（conv-26 +3.8pp，conv-30 −1.2pp），已回滚。

## 实现原则

- **Search 只返回历史证据**，不回答问题、不调用生成式模型，也不把答案伪装成记忆。
- **原文优先**：存储原始消息，窗口只保存指向原文的 Unicode 半开区间；输出可回溯到原消息。
- **按 `user_id` 物理隔离**：每个用户使用一个 SQLite 文件，路径只包含 user ID 的 SHA-256；请求幂等、epoch 发布、FTS/向量安装在同一发布事务中。
- **模型可替换**：HTTP 服务只依赖 `ModelProvider` 协议；远程供应商的路径、请求体和响应解析集中在 `providers/openrouter.py`，本地 OpenAI 兼容端点（llama.cpp / vLLM / LM Studio）在 `providers/local_openai.py`，二者按 `champion-v1.json` 的 provider 名由 `providers/factory.py` 组装（embedding 与 rerank 可由不同供应商承担，用 `SplitProvider` 拼成单个 `ModelProvider`；`reranker.enabled=false` 时精排层整段跳过）。
- **不静默降级**：embedding 或 rerank 失败时 Add/Search 返回非 2xx，不把“模型失败”伪装成成功的词法检索。
- **参数可审计**：算法配置在 `src/memorysum/config/champion-v1.json`，环境变量只用于密钥、服务地址、数据目录和入站鉴权。

## 快速开始

要求 Python 3.12。

```bash
cd memorysum
cp .env.example .env
# 在 .env 中填写 OPENROUTER_API_KEY 和 MEMORYSUM_API_KEYS_FILE；在项目目录 keys.txt 中逐行写入允许访问的 Bearer token
# 依赖安装（本机直连 PyPI CDN 会被墙，走清华镜像）
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install -e ".[dev]"
# 启动前先拉起 embedding / rerank 服务（见下节“服务拓扑”）
.venv/bin/memorysum
```

服务固定监听 `0.0.0.0:8001`，健康检查 `GET /health` 不需要鉴权；所有 Add/Search 请求必须带 `Authorization: Bearer <keys.txt 中的一行 token>`。`keys.txt` 每行一个 token **本体**（不要加 `Bearer ` 前缀），空行与 `#` 注释行会忽略；文件已加入 `.gitignore`，建议权限设为 `600`。启动时读取，修改后需重启服务。`MEMORYSUM_API_KEYS_FILE` 默认是当前目录下的 `keys.txt`，也可配置绝对路径。

`OPENROUTER_API_KEY` 是服务调用模型供应商的凭据；`keys.txt` 中的 Bearer tokens 是调用 Add/Search 的入站凭据，两者用途不同。

## 服务拓扑

本机开发需要三个进程（llama.cpp 一个实例只能加载一个模型，所以 embedding 与 rerank 必须分开）：

| 服务 | 端口 | 模型 | 显存 | 说明 |
|---|---|---|---|---|
| memorysum | 8001 | — | — | `.venv/bin/memorysum` |
| embedding | 7999 | `Qwen3-Embedding-4B-Q8_0.gguf` | ~8.0 GB | 每次 Add/Search 都调用 |
| rerank | 7997 | `bge-reranker-v2-m3-f16.gguf` | ~1.0 GB | 每次 Search 调用（`reranker.enabled=true`） |

```bash
# embedding（注意：embeddings 模式下 -b 必须等于 -ub，llama.cpp 会强制覆盖并告警）
llama-server -m /home/rain/model/gguf/Qwen3-Embedding-4B-Q8_0.gguf \
  --embeddings --pooling last -fa on -c 32768 -b 2048 -ub 2048 \
  -ctk q8_0 -ctv q8_0 -ngl 99 --host 127.0.0.1 --port 7999 --alias qwen3-embedding-4b

# rerank（-ub 必须 ≥ champion-v1.json 的 reranker.single_pair_proxy_tokens，见下）
llama-server -m /home/rain/model/gguf/bge-reranker-v2-m3-f16.gguf \
  --reranking --pooling rank -fa on -c 8192 -b 8192 -ub 8192 -ngl 99 \
  --host 127.0.0.1 --port 7997 --alias bge-reranker-v2-m3
```

**rerank 的 `-ub` 必须覆盖「最坏情况下真的会送进来的 pair 长度」**，而配置里的
`reranker.single_pair_proxy_tokens = 4608` 只是**代理估算预算**，两者关系靠标定维系：

| 量 | 值 | 来源 |
|---|---|---|
| 模型窗口 = `-c/-ub` 实际天花板 | 8192 | `bge-reranker-v2-m3` 训练窗口（llama.cpp 会把更大的 `-c` 压回来并告警） |
| 单对代理预算 | 4608 | `champion-v1.json`，比的是 `reader_token_count()`（tiktoken cl100k） |
| 估值的低估倍率 real/est | 中位 1.166 / p99 1.357 / **最坏 1.479** | 1330 条真实 memory + 对抗样本，`scripts/calibrate_rerank_tokens.py` |
| 预算顶格时的真实长度 | 4608 × 1.479 ≈ **6814** | 离窗口还有 17% 余量 |

低估方向很反直觉：**XLM-R 对英文/URL/标点切得比 cl100k 更碎**（URL/UUID 1.479、标点密集
1.334、英文散文 1.17），而中日韩俄阿都小于 0.7（中文 0.57）。所以预算是按英文那侧留的。
改模型或换 tokenizer 后重跑标定脚本，它会在不安全时以非 0 退出。

为什么服务端不跟着改小显存：`-ub` 已经贴着模型窗口（8192），再小就重新出现
「客户端以为能发、服务端收不了」的悬崖；再大也没用（slot 上下文会被压回 8192，实测
`-ub 16384` 只多占 576 MB）。当前 rerank 实例 1046 MB，全卡 9.1/16.3 GB。

`--pooling rank` 让 llama.cpp 的 `slot.can_split()` 为 false，**每一对 (query, document)
必须整体塞进一个物理 batch**，否则服务端直接拒绝：

```
E srv send_error: error: input (3648 tokens) is too large to process.
                  increase the physical batch size (current batch size: 2048)
```

2026-10-14 的压测就是踩了这个：`-ub 2048` 而那时的预算写着 8192，一条 9050 字符的
query（pair 3648 token）被重试 23 次、每次 3 秒后返回 503；改成 `-ub 8192` 后同一请求
**200 / 289 ms**。

两道保险：

1. **预算留余量**（上表），超预算的候选不进 rerank；
2. 万一还是被服务端拒（换模型、标定失效），`_post_json` 会立刻抛
   `model_provider_input_too_large` 并把服务端原话记进日志，不再空转 1+2 秒。

并发也不是悬崖：实测 4×3570、6×3570、8×6970 token 并发（合计最高 55.7k token，远超
8192 的 unified KV）全部 200，只是排队变慢（p50 3.3 s / max 5.8 s）。

如果 query 太长使得**每个**候选 pair 都超预算，Search 不会返回空数组，而是退回融合排序
（与 `reranker.enabled=false` 同样的排序信号），策略字段记为
`hybrid_retrieval_unreranked` 并记 WARNING——证据照常返回，只是没有精排。

模型来源：embedding 是 HF 的 F16 GGUF 经 `llama-quantize ... Q8_0` 转出；rerank 是 `convert_hf_to_gguf.py` 从 `BAAI/bge-reranker-v2-m3` 的 safetensors 转换。两者都放在 `/home/rain/model/gguf/`。

**实测吞吐**：embedding 封顶约 **21 windows/s**（并发 1→8、批大小 1→64 的所有组合都一样——瓶颈是 GPU 计算，不是显存或 slot 数；提高 ubatch 实测无收益）。单条 query 约 15–60 ms，Add 1000 窗口约 48 秒。

**注意**：这三个都是裸进程，机器重启不会自动恢复；memorysum 依赖 7999 与 7997 可达，否则 Add/Search 返回 503。

## API

同时提供简洁路由 `/add`、`/search` 和兼容路由 `/v1/memories/add`、`/v1/memories/search`。

### Add

```json
{
  "request_id": "add-001",
  "messages": [
    {"role": "user", "timestamp": 1704067200000, "content": "I prefer tea."},
    {"role": "assistant", "content": "I will remember that."}
  ],
  "user_id": "opaque-user-id",
  "session_id": "opaque-session-id"
}
```

成功只在全部必需 embedding、向量和 FTS 索引发布后返回 HTTP 200，且原样回显 `request_id`、`user_id`、`session_id`。同一用户同一 `request_id`、同一规范 payload 幂等；payload 不同返回 422。`timestamp` 可省略，但出现时必须是整数毫秒，显式 `null` / bool 不接受。

### Search

```json
{
  "query": "What do I prefer?",
  "options": ["A. Tea", "B. Coffee"],
  "user_id": "opaque-user-id",
  "top_k": 100
}
```

Search 每条结果包含 `id`、`content`、`score` 和 `created_at`：`score` 是当前 reranker 的原始数值分数，越大越相关（不同模型/版本间不保证可直接比较）；`created_at` 为 UTC ISO 8601，优先使用 Add 消息的来源 `timestamp`，未提供时回退到 Add 请求接收时间。`content` 是原文证据片段，不插入内部 `[Source]`、`Role`、`Time` 标签。窄查询走混合检索和 rerank，按分数排序后由 `top_k` 限制结果数；完全重复的正文会合并，结果仍受服务端记录数与输出预算上限约束。明确的宽泛/上下文查询在完整 bundle 预算与 rerank 容量允许时也会逐条评分；`top_k=0` 返回空数组，未知用户空间返回 `{"data": []}`。不同 `user_id` 的数据不会互相检索。文本接口只接受字符串 `query` / `content`；有序图片 `ContentPart[]` 属于多模态赛道，本版本不实现。

### Smoke 命令

```bash
BEARER_TOKEN="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' keys.txt)"

curl -sS http://127.0.0.1:8001/health

curl -sS http://127.0.0.1:8001/add \
  -H "Authorization: Bearer ${BEARER_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-1","user_id":"demo-user","session_id":"demo-session","messages":[{"role":"user","content":"I prefer tea."}]}'

curl -sS http://127.0.0.1:8001/search \
  -H "Authorization: Bearer ${BEARER_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What do I prefer?","user_id":"demo-user","top_k":10}'
```

默认 `MEMORYSUM_DEBUG=false` 时，每个 HTTP 请求仍会记录一行摘要，包含 UTC 时间戳、方法、路径、状态码和 `latency_ms`，但不记录请求正文。开发时设置 `MEMORYSUM_DEBUG=true`，会额外将 Add/Search 的完整 JSON（正文、问题、options、user_id、request_id、session_id）写入服务日志，便于本机调试。日志不会包含 HTTP headers、Authorization、OPENROUTER_API_KEY 或其他密钥。调试日志可能包含敏感正文，请仅在本机使用，避免共享或长期保留。

如需查看真实请求 JSON，可在本机开发时开启 `MEMORYSUM_DEBUG=true`；不要在共享机器或日志长期留存环境开启。

## 当前垂直链路

1. 严格校验 AML 文本 Add / Search 字段、消息角色、时间戳和容量边界。
2. 以确定性 embedding-token 代理构造 leaf/context 双尺度窗口；句子、标题、列表、表格、代码和引用会映射为精确原文 atom，context 窗口只在同一 Add 请求中按源序跨消息组合，最多 4 个 source spans。
3. embedding 批量请求（provider 由配置决定：本地 OpenAI 兼容端点或 OpenRouter）；校验 vector 个数、维度、finite/norm，并按 float32 归一化存入 `sqlite-vec`。
4. SQLite FTS5 英文词法 + Jieba/中文 bigram 词法与双尺度 dense 召回，按 RRF 排名；options 作为受限词法提名。
5. 对同消息相交窗口合并；跨消息 context groups 保留清晰 span 边界并受窗口数限制，再经 reranker（当前为本机 `bge-reranker-v2-m3`）打分；任何批次失败都不返回部分排序。
6. 宽泛/上下文检索在预算内可按消息返回完整原文；普通检索按**精排与 RRF 融合后的排序**、`top_k`、证据组与 prompt 预算选择原文片段，不把元数据标签拼进 API 正文。
7. 新 Add 通过入口 epoch 与原子发布隔离；用户数据从首次接收起 30 天固定 TTL，过期后退休并清理 SQLite/WAL/SHM。

结构与边界详见 [`docs/architecture.md`](docs/architecture.md)。

## 运行测试和静态检查

测试使用本地 fake provider，不请求 OpenRouter、不包含比赛私有数据或 gold answers：

```bash
uv run ruff check src tests
uv run pytest -q
```

## 直接运行

```bash
# 安装 Python 3.12 和 uv 后，在服务器代码目录执行
# 注意：不要用 `uv sync --frozen`——uv.lock 记录的是被墙的 files.pythonhosted.org 绝对 URL
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install -e .

# 如果 keys.txt 不存在就新建；有既存 token 时不要覆盖
if [ ! -f keys.txt ]; then (umask 077 && : > keys.txt); fi
chmod 600 keys.txt
${EDITOR:-vi} keys.txt

# .env 中配置 OPENROUTER_API_KEY 与 MEMORYSUM_API_KEYS_FILE=keys.txt
export MEMORYSUM_HOST=0.0.0.0
.venv/bin/memorysum
```

API 端口固定为 **8001**；健康检查 `/health` 无鉴权，Add/Search 必须使用 `Authorization: Bearer ...`。云服务器需在安全组开放 TCP 8001；生产访问建议放在 HTTPS 反向代理后。`MEMORYSUM_DEBUG=true` 会将请求正文写入服务日志，仅用于受控开发调试。

## 规则与设计资料

- [Agent Memory Leaderboard API Guide](https://agentmemories.ai/api-guide)
- [Agent Memory Challenge Rules](https://agentmemories.ai/rules)
- Champion v1 冻结规格由项目维护者提供，作为本仓初始实现输入；它不是官方比赛说明或成绩报告。
- 参考了项目维护者的本地评测与方法资料；该资料和其中的评测数据均未复制或打包进本仓。

本仓提供可复用的 Add/Search API 服务骨架。AML API Guide / Rules 用作接口和设计参考；若在其他比赛使用，应按对方的数据留存、鉴权与模型规则单独配置。
