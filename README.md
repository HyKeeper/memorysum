# MemorySum

面向记忆检索系统开发的原文记忆服务，提供与 AML Text Add / Search 兼容的 API 形状。当前仓库侧重本机开发与真实服务链路验证，不代表任何比赛成绩。

> **当前算法版本：`memorysum-hybrid-preview-v0.5.0`（provisional）**
> embedding 走本机 llama.cpp 服务的 `qwen3-embedding-4b`（OpenAI 兼容 `/v1/embeddings`，F16 GGUF、`--pooling last`、2560 维，地址见 `MEMORYSUM_EMBEDDING_BASE_URL`，默认 `http://127.0.0.1:7999/v1`），零调用成本：实测单条 query 约 25ms、10 窗口批次 0.3–0.55s、3000 窗口的 Add 约 79s。reranker 仍走 OpenRouter 的 `qwen/qwen3-reranker-8b`（计费路由约 $0.2/M token，实测单次检索 64 个候选约 $0.0016）。冻结规格要求的 text-embedding-v4 512 维稠密+稀疏向量尚未接入（sparse 关闭）；reranker 的 Qwen3-Reranker-8B 虽在规格内，但走 OpenRouter 托管端点，**官方模板的 `<Instruct>` 槽位不可用**（实测传入 `instruction` 字段被忽略，token 数与打分均不变），instruction 仍拼接在 query 文本里，tokenizer 校验也仍用 `cl100k_base` 近似。维度自动探测（qwen3-embedding-4b 为 2560 维）。查询侧用 Qwen3 官方 `Instruct: …\nQuery:` 模板；本地服务 `-ctk/-ctv q8_0` 的 KV cache 会给同一 query 带来约 3.5e-4 的跨请求余弦漂移（比检索 margin 小两个数量级）。context 窗口仍为 target 320 / max 384 proxy tokens（当初为适配 LFM 路由而缩窄，未回退到冻结 Champion v1 参数）；当前版本不能描述成已通过正式模型验收。

> **rerank 开关（省钱测试模式）**：`champion-v1.json` 的 `reranker.enabled` 控制是否调用精排模型，当前为 **`false`**——Search 在 RRF 融合后直接输出，**不再请求 OpenRouter**。此时 `SearchItem.score` 是 RRF 融合分（`Σ weight/(k+rank)`，通常 < 0.1），而不是 rerank 打分；`api_search` 日志新增 `reranked=%s` 字段用于区分，`rerank_groups` 仍是进入候选池的条数。开关只改变排序，不改变向量/窗口/索引，因此 `pipeline_id` 不变、已有数据无需迁移；但同一 `pipeline_id` 下两种模式的 score 语义不同，跑评测对比时须按 `reranked` 字段分组。

> **RRF 通道权重（实测调参，2026-10-13）**：`retrieval.channel_weights` 按通道加权（未列出的通道回退到 `base_channel_weight`），当前为 `leaf_dense 2.0 / context_dense 1.0 / english_fts 0.5 / chinese_fts 0.25 / options_lexical 1.0`。依据：在 BEAM 单 scope（1147 窗口）+ 60 条英文事实查询（hard negative 密集）上的离线评测——`rrf_k` 在 10–200 间对 top-1/top-3 无影响（RRF 的已知鲁棒性，故保持 60）；**leaf_dense 加权 + 降低词法通道权重**把 top-1 从 28/60 提到 34/60、MRR 0.565→0.664，且在两个半样本上各自成立（交叉验证否决了只在调参半区有效的过拟合解）。两个反直觉点：**① `chinese_fts` 并非中文专用**——它走 jieba 分词，对英文文本同样产出词项（实测 gold 召回 26/30、平均排名 4.2），权重降到 0 反而降到 32/60，故必须保留小权重；**② 收紧通道配额（zh 24→8、en 80→96、leaf 96→64）在实测中全部无益或有害**，未采用。

> **配置清理（2026-10-13）**：`champion-v1.json` 现在只保留被代码真正读取的字段。已删除的纯装饰性字段（`features` 整段、`language`、`generation_enabled`、`embedding.sparse_output`、`reranker.tokenizer`/`tokenizer_is_approximate`、`chunking.token_proxy`、`retrieval.unused_quota_transfer`/`sparse_channel`、`coverage.reader_estimator`）此前只表达能力声明，改它们不会影响任何行为。特别地，`capacity.max_add_messages`/`max_search_options` 曾是**假配置**——声明了但真实限制硬编码在 `api/schemas.py`，现已改为 schemas 直接读 config，并由 `test_wire_limits_track_pipeline_config` 守护。

> **本地 rerank 支持与实测结论（2026-10-13）**：`providers/local_openai.py` 现在同时提供本地 cross-encoder 精排（`LocalRerankProvider`），兼容两种响应形状——llama.cpp `--reranking` 的 `{"results":[{"index","relevance_score"}]}` 与自建 FastAPI 服务的裸数组 `[{"index","score"}]`；端点由 `MEMORYSUM_RERANK_BASE_URL` 指定（默认 `http://127.0.0.1:7997`），`reranker.instruction` 为空表示不给 cross-encoder 加任何指令前缀。**但当前 `reranker.enabled=false`**：在 BEAM 单 scope + 60 条英文事实查询上实测，本地 `bge-reranker-v2-m3`（568M，F16 GGUF，由 `convert_hf_to_gguf.py` 从 safetensors 转换）把 top-1 34→35/60、top-3 44→47、MRR 0.664→0.682，**修复 10 条同时弄坏 9 条（净 +1）**，而 p50 延迟从 71ms 涨到 1961ms（**28 倍**）；判断是“重排”而非“改进”——它常把同一主题的另一段代码排到前面，对单-gold 评测是损失，对真实使用未必有错，但净收益不足以抵消延迟。启动本地 rerank 服务：`llama-server -m bge-reranker-v2-m3-f16.gguf --reranking --pooling rank -fa on -c 8192 -ngl 99 --port 7997`（约 887MiB 显存）。另实测：rerank 输入带/不带 `Source role:/Passage:` 渲染前缀时 top-5 完全一致（5/5），故未改渲染。

## 实现原则

- **Search 只返回历史证据**，不回答问题、不调用生成式模型，也不把答案伪装成记忆。
- **原文优先**：存储原始消息，窗口只保存指向原文的 Unicode 半开区间；输出可回溯到原消息。
- **按 `user_id` 物理隔离**：每个用户使用一个 SQLite 文件，路径只包含 user ID 的 SHA-256；请求幂等、epoch 发布、FTS/向量安装在同一发布事务中。
- **模型可替换**：HTTP 服务只依赖 `ModelProvider` 协议；远程供应商的路径、请求体和响应解析集中在 `providers/openrouter.py`，本地 OpenAI 兼容端点（llama.cpp / vLLM / LM Studio）在 `providers/local_openai.py`，二者按 `champion-v1.json` 的 provider 名由 `providers/factory.py` 组装（本地 embedding + 远程 rerank 用 `SplitProvider` 拼接；`reranker.enabled=false` 时精排层整段跳过）。
- **不静默降级**：embedding 或 rerank 失败时 Add/Search 返回非 2xx，不把“模型失败”伪装成成功的词法检索。
- **参数可审计**：算法配置在 `src/memorysum/config/champion-v1.json`，环境变量只用于密钥、服务地址、数据目录和入站鉴权。

## 快速开始

要求 Python 3.12。

```bash
cd memorysum
cp .env.example .env
# 在 .env 中填写 OPENROUTER_API_KEY 和 MEMORYSUM_API_KEYS_FILE；在项目目录 keys.txt 中逐行写入允许访问的 Bearer token
uv sync --extra dev
uv run memorysum
```

服务固定监听 `0.0.0.0:8001`，健康检查 `GET /health` 不需要鉴权；所有 Add/Search 请求必须带 `Authorization: Bearer <keys.txt 中的一行 token>`。`keys.txt` 每行一个 token **本体**（不要加 `Bearer ` 前缀），空行与 `#` 注释行会忽略；文件已加入 `.gitignore`，建议权限设为 `600`。启动时读取，修改后需重启服务。`MEMORYSUM_API_KEYS_FILE` 默认是当前目录下的 `keys.txt`，也可配置绝对路径。

`OPENROUTER_API_KEY` 是服务调用模型供应商的凭据；`keys.txt` 中的 Bearer tokens 是调用 Add/Search 的入站凭据，两者用途不同。

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
5. 对同消息相交窗口合并；跨消息 context groups 保留清晰 span 边界并受窗口数限制，再经 reranker（当前为 OpenRouter）打分；任何批次失败都不返回部分排序。
6. 宽泛/上下文检索在预算内可按消息返回完整原文；普通检索按 rerank 排序、`top_k`、证据组与 prompt 预算选择原文片段，不把元数据标签拼进 API 正文。
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
uv sync --frozen --no-dev

# 如果 keys.txt 不存在就新建；有既存 token 时不要覆盖
if [ ! -f keys.txt ]; then (umask 077 && : > keys.txt); fi
chmod 600 keys.txt
${EDITOR:-vi} keys.txt

# .env 中配置 OPENROUTER_API_KEY 与 MEMORYSUM_API_KEYS_FILE=keys.txt
export MEMORYSUM_HOST=0.0.0.0
uv run memorysum
```

API 端口固定为 **8001**；健康检查 `/health` 无鉴权，Add/Search 必须使用 `Authorization: Bearer ...`。云服务器需在安全组开放 TCP 8001；生产访问建议放在 HTTPS 反向代理后。`MEMORYSUM_DEBUG=true` 会将请求正文写入服务日志，仅用于受控开发调试。

## 规则与设计资料

- [Agent Memory Leaderboard API Guide](https://agentmemories.ai/api-guide)
- [Agent Memory Challenge Rules](https://agentmemories.ai/rules)
- Champion v1 冻结规格由项目维护者提供，作为本仓初始实现输入；它不是官方比赛说明或成绩报告。
- 参考了项目维护者的本地评测与方法资料；该资料和其中的评测数据均未复制或打包进本仓。

本仓提供可复用的 Add/Search API 服务骨架。AML API Guide / Rules 用作接口和设计参考；若在其他比赛使用，应按对方的数据留存、鉴权与模型规则单独配置。
