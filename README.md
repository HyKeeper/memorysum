# MemorySum

原文保真的记忆检索服务：`Add` 写入、`Search` 检索。FastAPI + SQLite（FTS5 + sqlite-vec）。

**所有模型调用都走远程 HTTP API**：embedding 与 rerank 默认都由阿里云 DashScope 提供（原生 API），
进程内不加载任何模型、不需要 GPU、也不需要在本机跑推理服务。换供应商只需改 `champion-v1.json`
（也支持 OpenRouter）。**不生成答案，只返回可回溯到原文的证据片段。**

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│  Client (Add / Search)                                  │
└────────────────────┬────────────────────────────────────┘
                     │ :8001
┌────────────────────▼────────────────────────────────────┐
│  MemorySum Service (FastAPI + SQLite)                   │
│  ┌──────────┐  ┌───────────┐  ┌───────────────────┐    │
│  │ FTS5     │  │ sqlite-vec│  │ RRF 融合 + 精排   │    │
│  │ (词法)   │  │ (向量)    │  │ (fusion_weight)   │    │
│  └──────────┘  └───────────┘  └─────────┬─────────┘    │
└──────────────────────────────────────────┼──────────────┘
                                           │ HTTPS
                     ┌─────────────────────▼─────────────────────┐
                     │ 模型 API（远程）                          │
                     │ embedding: text-embedding-v4 (2048 维)   │
                     │ rerank:    qwen3-rerank                   │
                     └───────────────────────────────────────────┘
```

### 检索流程

1. **5 路召回**：leaf_dense / context_dense / english_fts / chinese_fts / options_lexical
2. **RRF 融合**：`rrf_k=20` + 自适应（`k = clamp(候选数/3, 10, 30)`）
3. **Reranker 精排**：远程 cross-encoder（默认 `qwen3-rerank`）
4. **最终排序**：`score = 1/(k + rerank_rank) + fusion_weight/(k + retrieval_rank)`

精排只重排、不取代检索排序：融合项保留了多通道共识信号，实测纯精排排序在两个半样本上分别是
+4 / −3，融合后为 +4 / 0。

## 快速开始

### 1. 安装

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. 配置凭据

```bash
# 模型 API（唯一的必需凭据）
export DASHSCOPE_API_KEY='sk-...'
# 可选：账号使用专用 MaaS 部署时，把 base URL 指过去（默认 https://dashscope.aliyuncs.com）
export MEMORYSUM_DASHSCOPE_BASE_URL='https://<workspace>.<region>.maas.aliyuncs.com'

# 入站鉴权：每行一个 Bearer token
echo 'your-caller-token' > keys.txt
```

环境变量只放**密钥、服务地址、数据目录、入站鉴权**；算法参数一律在 `champion-v1.json`。

| 变量 | 作用 |
|---|---|
| `DASHSCOPE_API_KEY` / `MEMORYSUM_DASHSCOPE_API_KEY` | DashScope 密钥（后者优先） |
| `MEMORYSUM_DASHSCOPE_BASE_URL` | DashScope 主机（不带路径），默认 `https://dashscope.aliyuncs.com` |
| `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` | 仅在 `provider = "openrouter"` 时需要 |
| `MEMORYSUM_API_KEYS_FILE` | 入站 Bearer token 文件，默认 `keys.txt` |
| `MEMORYSUM_DATA_ROOT` | 数据目录，默认 `./data` |
| `MEMORYSUM_HOST` | 监听地址，默认 `0.0.0.0`（端口固定 8001） |
| `MEMORYSUM_DEBUG` | 打开后记录请求载荷审计行，默认 `false` |

### 3. 启动

只需要**一个进程**，没有本地模型服务要拉起：

```bash
.venv/bin/memorysum
# 等价于 uvicorn memorysum.asgi:app --host 0.0.0.0 --port 8001
```

后台运行：

```bash
mkdir -p logs
set -a; . ./.env; set +a
setsid nohup .venv/bin/memorysum > logs/memorysum.log 2>&1 &
```

### 4. 验证

```bash
# 健康检查：缺模型凭据时直接 503，不会等到第一次 Add/Search 才暴露
curl http://localhost:8001/health
# → {"status":"ok"}

# 直连 embedding API（应返回 2048 个浮点数）
curl -sS "$MEMORYSUM_DASHSCOPE_BASE_URL/api/v1/services/embeddings/text-embedding/text-embedding" \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-v4","input":{"texts":["dimension smoke test"]},"dimension":2048}' \
  | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print(len(d["output"]["embeddings"][0]["embedding"]))'

# 直连 rerank API（应返回 output.results，含 relevance_score）
curl -sS "$MEMORYSUM_DASHSCOPE_BASE_URL/api/v1/services/rerank/text-rerank/text-rerank" \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-rerank","input":{"query":"capital of France?","documents":["Paris.","Bananas."]},"parameters":{"return_documents":false,"top_n":2}}'

# 添加记忆
curl -X POST http://localhost:8001/add \
  -H "Authorization: Bearer $(head -1 keys.txt)" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "user_id": "test-user",
    "session_id": "session-1",
    "messages": [
      {"role": "user", "content": "My name is Alice. I work as a data scientist at Google."},
      {"role": "user", "content": "I prefer dark mode and use a 27-inch monitor."}
    ]
  }'

# 搜索记忆
curl -X POST http://localhost:8001/search \
  -H "Authorization: Bearer $(head -1 keys.txt)" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Alice'\''s job?", "user_id": "test-user", "top_k": 5}'
```

## 模型供应商

`champion-v1.json` 里 `embedding.provider` 与 `reranker.provider` 各自选一个**远程 API**：

| provider | embedding | rerank | 端点 / 凭据 |
|---|---|---|---|
| `dashscope`（默认） | `text-embedding-v4`（2048 维，需显式声明） | `qwen3-rerank` | `MEMORYSUM_DASHSCOPE_BASE_URL` + `DASHSCOPE_API_KEY` |
| `openrouter` | config 指定（OpenAI 兼容 `/embeddings`） | config 指定（`/rerank`） | `OPENROUTER_BASE_URL` + `OPENROUTER_API_KEY` |

两段可以混用（例如 DashScope embedding + OpenRouter rerank），由 `SplitProvider` 组合；
同一供应商时共用一个 HTTP 客户端。两个供应商都实现 embedding 与 rerank，不再有"只能做一半"的 provider。

> 进程内不加载模型，也没有任何隐式本地端点：**不存在** `127.0.0.1:7999/7997` 之类的回退地址，
> 缺凭据就是 503 `model_provider_not_configured`，不会静默降级。

### Embedding：DashScope `text-embedding-v4`

**请求契约**（2026-10-20 实测）：

```http
POST {base}/api/v1/services/embeddings/text-embedding/text-embedding
Authorization: Bearer $DASHSCOPE_API_KEY
Content-Type: application/json

{"model":"text-embedding-v4",
 "input":{"texts":["...","..."]},
 "dimension":2048}
```

响应：

```json
{"output":{"embeddings":[{"text_index":0,"embedding":["...2048 floats..."]}]},
 "usage":{"total_tokens":23},"request_id":"..."}
```

**硬限制**（文档 + 实测）：

| 项 | 值 | 实测 |
|---|---|---|
| 单请求文本条数 | ≤ 10 | 11 条返回 HTTP 400 `InvalidParameter: batch size is invalid` |
| 单条 token 数 | 文档：≤ 8192 | 实测 9000 个中文字符仍被接受（本管线窗口远小于此） |
| 维度 | 64/128/256/512/768/1024/1536/2048/3072 | 必须显式给出；省略时默认 1024 |

本仓库固定 `dimension: 2048`、`batch_items: 10`、`query_instruction: ""`
（v4 不需要指令前缀；空指令时查询文本原样送进模型）。

> v4 还有一个可选的 `text_type`（`query` / `document`，默认 `document`）用于非对称检索。
> 本实现**没有**设置它：查询与文档走同一编码模式。若将来要用，必须同时改
> `ModelProvider.embed` 的角色参数并 bump `embedding_space_id`，不能只在一侧加前缀。

### Rerank：DashScope `qwen3-rerank`

**请求契约**（2026-11-02 实测）：

```http
POST {base}/api/v1/services/rerank/text-rerank/text-rerank
Authorization: Bearer $DASHSCOPE_API_KEY
Content-Type: application/json

{"model":"qwen3-rerank",
 "input":{"query":"...", "documents":["...","..."]},
 "parameters":{"return_documents":false, "top_n":2}}
```

响应：

```json
{"output":{"results":[{"index":0,"relevance_score":0.8137},
                      {"index":1,"relevance_score":0.1781}]},
 "usage":{"total_tokens":54},"request_id":"..."}
```

要点：

- `relevance_score` 是 **request 内相对分数**（0~1，越大越相关），跨请求不可比。管线只使用它诱导的
  **排名**（`_fuse_rankings` 按 rank 融合），所以请求内归一化没有影响。
- 每请求最多 **500 条文档**、每条（query 或 document）最多 **4,000 token**；实现整体送 `top_n = len(documents)`，
  少返回一条就按 `model_provider_invalid_rerank` 失败，绝不静默丢候选。
- 超长输入服务端返回 **HTTP 400 且不截断**。因此 `reranker.single_pair_proxy_tokens` 与
  `reranker.max_query_proxy_tokens` 都按"代理计数 × 1.494 < 4,000"标定（代理计数会低估真实 token，
  见 `tests/test_settings.py`）。超预算的候选不会被丢弃：不精排、按融合次序接在精排结果之后返回。
- `reranker.instruction` 非空时会拼进 query 文本；DashScope 的 `parameters.instruct` 不重复设置。
- 精排是可选加速项：`reranker.enabled=false` 时直接使用 RRF 分数，行为与"没有精排服务"完全一致。

## 配置

核心配置文件：`src/memorysum/config/champion-v1.json`（版本化、随包发布）。

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `pipeline_id` | `memorysum-hybrid-dashscope-v0.7.0` | 数据契约版本；embedding 变化必须 bump |
| `embedding.provider` | `dashscope` | 供应商（`dashscope` / `openrouter`） |
| `embedding.model` | `text-embedding-v4` | |
| `embedding.dimension` | 2048 | 必须显式；与响应不一致会 503 |
| `embedding.batch_items` | 10 | DashScope 单请求上限也是 10 |
| `reranker.provider` | `dashscope` | 供应商（`dashscope` / `openrouter`） |
| `reranker.model` | `qwen3-rerank` | |
| `reranker.single_pair_proxy_tokens` | 2560 | 单对（query+document）代理预算 |
| `reranker.max_query_proxy_tokens` | 2048 | 超长 query 的裁剪阈值 |
| `reranker.fusion_weight` | 2.0 | 精排 vs 检索的融合权重 |
| `retrieval.rrf_k` | 20 | RRF 融合常数（越小头部优势越大） |
| `retrieval.per_source_cap` | 1 | 每个 chunk 最多返回几条 |
| `groups.direct_max_tokens` | 640 | 每条记录的最大渲染 token 数 |
| `groups.time_anchor_in_text` | true | 时间类问题在正文前补可见锚点 |

`evidence_purify` 是一个**默认关闭**的可选后处理：需要显式配置外部决策 API 的
`endpoint`（不存在隐式本地地址），启用后只给前 N 条记录加 `[Key evidence: …]` 前缀，
原文完整保留，任何失败都原样返回。

### 切换/升级行为（重要）

- `pipeline_id` 与 `embedding_space_id` 随 provider/model/instruction/dimension 变化；
  `api/app.py` 还会把 embedding **端点地址**并入数据库指纹，所以换 `MEMORYSUM_DASHSCOPE_BASE_URL`
  同样会让旧 scope 503。
- 旧 scope 一律 HTTP 503 `pipeline_mismatch`，**不会自动重建或重新向量化**；换 embedding 后需要重新 `Add`。
- 只改 reranker / 排序参数不需要 bump，已有数据可继续用。

## 测试

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
```

测试全部使用 `httpx.MockTransport`，不会触网、不需要 API key。真实 API 的验证见上文「4. 验证」。

## 运行与重启

```bash
REAL=$(ss -tlnpH 'sport = :8001' | grep -oP 'pid=\K[0-9]+' | head -1)
kill -TERM $REAL; sleep 2; kill -9 $REAL 2>/dev/null
setsid nohup .venv/bin/memorysum > logs/memorysum.log 2>&1 &
sleep 4
ss -tlnpH 'sport = :8001'   # 确认 PID 变了
```

## 目录结构

```
memorysum/
├── src/memorysum/          # 核心代码
│   ├── api/                # FastAPI 路由
│   ├── config/             # 配置文件（champion-v1.json）
│   ├── core/               # 分块、融合、排序、渲染
│   ├── providers/          # dashscope / openrouter / split（全部是远程 API）
│   ├── services/           # AddService / SearchService
│   ├── storage/            # SQLite 数据库管理
│   └── settings.py         # 配置加载
├── tests/                  # 测试（mock transport，不触网）
├── data/                   # 运行时数据（不提交）
├── logs/                   # 日志（不提交）
├── keys.txt                # 入站 Bearer token（不提交）
└── .env                    # 环境变量（不提交）
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/add`、`/v1/memories/add` | 添加记忆 |
| POST | `/search` | 搜索记忆 |
| GET | `/health` | 健康检查 |

详细契约：启动后访问 `http://localhost:8001/docs`（Swagger UI）。

## 常见问题

### Search 返回 503

按日志里的 `model_provider_*` code 判断是哪一侧失败：

```bash
# embedding（应返回 2048 维）
curl -sS "$MEMORYSUM_DASHSCOPE_BASE_URL/api/v1/services/embeddings/text-embedding/text-embedding" \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-v4","input":{"texts":["ping"]},"dimension":2048}' | head -c 200

# rerank
curl -sS "$MEMORYSUM_DASHSCOPE_BASE_URL/api/v1/services/rerank/text-rerank/text-rerank" \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-rerank","input":{"query":"ping","documents":["ping"]}}' | head -c 200
```

| code | HTTP | 触发条件 |
|---|---|---|
| `model_provider_not_configured` | 503 | 缺对应供应商的 key |
| `model_provider_unauthorized` | 503 | key 被拒绝（401/403） |
| `model_provider_rate_limited` | 429 | 服务端限流 |
| `model_provider_invalid_embedding` | 503 | 条数/索引/维度不一致、非数值或空向量 |
| `model_provider_invalid_rerank` | 503 | 结果条数/索引不完整、分数非法 |
| `model_provider_input_too_large` | 503 | 服务端明确拒绝超长输入 |
| `model_provider_timeout` | 504 | 模型请求超时 |
| `model_provider_unavailable` | 503 | 连接失败/重试耗尽 |

### PipelineMismatch

scope 的 `pipeline_id` / `embedding_space_id` 与当前配置不匹配（例如换了 embedding 模型或 base URL）。
这是**有意**拒绝混用向量空间，不会自动迁移。重新 `Add` 数据，或清掉开发数据：

```bash
rm -f data/scopes/*.sqlite3 data/registry.sqlite3
```

### 报 1024 维 / `model_provider_invalid_embedding`

说明 `dimension` 没被服务端采纳。最常见原因是用公共端点 `https://dashscope.aliyuncs.com`
（实测会忽略 `dimension`，返回默认 1024 维）。把 `MEMORYSUM_DASHSCOPE_BASE_URL` 指向账号的专用
MaaS 部署，并确认 `embedding.dimension` 与期望一致。

### 返回内容太多 / 想减少 token

用 `groups.direct_max_tokens` 控制单条长度、`retrieval.ordinary.output_records` 控制条数上限。
两者都是排序无关的交付参数，改动不需要 bump `pipeline_id`。

### GitHub push 超时

```bash
for i in $(seq 1 6); do git push origin master && break; sleep 5; done
```

## 许可证

MIT，见 [LICENSE](LICENSE)。
