# MemorySum

原文保真的记忆检索服务：`Add` 写入、`Search` 检索。FastAPI + SQLite（FTS5 + sqlite-vec）。embedding 默认走阿里云 DashScope `text-embedding-v4`（原生 text-embedding API，2048 维），reranker 默认由本机 llama.cpp 提供 cross-encoder 精排；两者都可在 `champion-v1.json` 里替换。**不生成答案，只返回可回溯到原文的证据片段。**

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
                                           │
              ┌────────────────────────────┴──────────────────┐
              │                                               │
   ┌──────────▼─────────────┐                    ┌────────────▼─────────┐
   │ DashScope (HTTPS)      │                    │ Reranker :7997       │
   │ text-embedding-v4      │                    │ bge-reranker-v2-m3   │
   │ 2048 维                │                    │ (本机 llama.cpp)     │
   └────────────────────────┘                    └──────────────────────┘
```

> Embedding 也可以切回本机 llama.cpp `:7999`（`embedding.provider = "local-openai"`），
> 见下文「Embedding 后端」。

### 检索流程

1. **5 路召回**：leaf_dense / context_dense / english_fts / chinese_fts / options_lexical
2. **RRF 融合**：`rrf_k=20` + 自适应（`k = clamp(候选数/3, 10, 30)`）
3. **Reranker 精排**：bge-reranker-v2-m3（cross-encoder）
4. **最终排序**：`score = 1/(k + rerank_rank) + fusion_weight/(k + retrieval_rank)`

## 硬件要求

默认配置下本机只加载 reranker；embedding 在 DashScope 云端。

| 组件 | 最低显存 | 推荐显存 | 说明 |
|---|---|---|---|
| Embedding（DashScope text-embedding-v4） | 0 | 0 | 远程 HTTPS，2048 维 |
| Reranker（bge-reranker-v2-m3-F16） | 1 GB | 2 GB | cross-encoder，本机 llama.cpp |
| **合计** | **1 GB** | **2 GB** | 仅 reranker 占显存 |

可选的本机 embedding 回退（Qwen3-Embedding-4B-Q8_0）另需 6~8 GB，见「回退到本地 embedding」。

## 快速开始

### 1. 环境

```bash
# Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# 安装依赖（用清华镜像，避免被墙）
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install -e ".[dev]"
```

### 2. 下载模型

默认配置的 embedding 在 DashScope 云端（无需下载），本机只需要 reranker：

```bash
mkdir -p models
wget -c https://hf-mirror.com/BAAI/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-f16.gguf \
  -O models/bge-reranker-v2-m3-f16.gguf
```

<details>
<summary>可选：回退到本地 Qwen3-Embedding 时才需要</summary>

```bash
wget -c https://hf-mirror.com/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf \
  -O models/Qwen3-Embedding-4B-Q8_0.gguf
```
</details>

### 3. 启动服务

需要 **2 个常驻进程**（reranker + memorysum），embedding 走 DashScope HTTPS：

```bash
# 环境变量（也可写进本机 .env；.env 已在 .gitignore 中）
export DASHSCOPE_API_KEY='sk-...'
export MEMORYSUM_DASHSCOPE_BASE_URL='https://llm-5qr9sf5t4nflguf4.cn-beijing.maas.aliyuncs.com'

# 终端 1：Reranker 服务（:7997）
llama-server -m models/bge-reranker-v2-m3-f16.gguf \
  --reranking --pooling rank -fa on \
  -c 8192 -b 8192 -ub 8192 --cache-ram 0 \
  -ngl 99 --host 127.0.0.1 --port 7997 --alias bge-reranker-v2-m3

# 终端 2：MemorySum 服务（:8001）
.venv/bin/memorysum
```

**一键后台启动**（适合生产环境）：

```bash
# Reranker（--cache-ram 0 必须保留，见 AGENTS.md 第 50 条）
setsid nohup llama-server -m models/bge-reranker-v2-m3-f16.gguf \
  --reranking --pooling rank -fa on -c 8192 -b 8192 -ub 8192 --cache-ram 0 \
  -ngl 99 --host 127.0.0.1 --port 7997 --alias bge-reranker-v2-m3 \
  > logs/reranker.log 2>&1 &

# MemorySum
mkdir -p logs
set -a; . ./.env; set +a
setsid nohup .venv/bin/memorysum > logs/memorysum.log 2>&1 &
```

### 4. 验证

```bash
# 健康检查
curl http://localhost:8001/health
# → {"status":"ok"}

# 直连 DashScope：确认 base URL、key、维度都对（应返回 2048 个浮点数）
curl -sS --location "$MEMORYSUM_DASHSCOPE_BASE_URL/api/v1/services/embeddings/text-embedding/text-embedding" \
  --header "Authorization: Bearer $DASHSCOPE_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"model":"text-embedding-v4","input":{"texts":["dimension smoke test"]},"dimension":2048}' \
  | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print(len(d["output"]["embeddings"][0]["embedding"]), d.get("usage"))'
# → 2048 {"total_tokens": 4}（token 数随输入变化）

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
  -d '{
    "query": "What is Alice's job?",
    "user_id": "test-user",
    "top_k": 5
  }'
```

## Embedding 后端

`champion-v1.json` 的 `embedding.provider` 决定 embedding 走哪里：

| provider | 默认模型 | 维度 | 端点 / 环境变量 |
|---|---|---|---|
| `dashscope`（默认） | `text-embedding-v4` | 2048（显式） | `MEMORYSUM_DASHSCOPE_BASE_URL` + `DASHSCOPE_API_KEY` |
| `local-openai` | config 指定（如 `qwen3-embedding-4b`） | 自动发现 | `MEMORYSUM_EMBEDDING_BASE_URL`（默认 `http://127.0.0.1:7999/v1`） |
| `openrouter` | config 指定 | 自动发现 | `OPENROUTER_BASE_URL` + `OPENROUTER_API_KEY` |

`dashscope` 走 DashScope **原生** text-embedding 接口（不是 OpenAI 兼容模式）。
reranker 由 `reranker.provider` 单独决定，默认仍是本机 `local-openai`（`:7997`）。

### DashScope text-embedding-v4 接入说明

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
| 单条 token 数 | 文档：≤ 8192 | 实测 9000 个中文字符仍被接受（未触顶；本管线窗口远小于此） |
| 维度 | 64/128/256/512/768/1024/1536/2048/3072 | 必须显式给出；省略时默认 1024 |

本仓库固定 `dimension: 2048`、`batch_items: 10`、`batch_proxy_tokens: 8192`、
`query_instruction: ""`（v4 不需要指令前缀；空指令时查询文本原样送进模型）。

> v4 还有一个可选的 `text_type`（`query` / `document`，默认 `document`）用于非对称检索。
> 本实现**没有**设置它：查询与文档走同一编码模式。若将来要用，必须同时改
> `ModelProvider.embed` 的角色参数并 bump `embedding_space_id`，不能只在一侧加前缀。

**配置项**：

| 键 | 当前值 | 说明 |
|---|---|---|
| `pipeline_id` | `memorysum-hybrid-dashscope-v0.6.0` | 换 embedding 后必须 bump |
| `embedding.provider` | `dashscope` | 走原生 DashScope 嵌入 |
| `embedding.model` | `text-embedding-v4` | |
| `embedding.dimension` | `2048` | 必须显式；与响应维度不一致会 503 |
| `embedding.batch_items` | `10` | 不得超过服务端上限 |
| `embedding.query_instruction` | `""` | v4 无指令；留空避免 `"\n\n"` 前缀 |
| `reranker.provider` | `local-openai` | reranker 仍是本机 bge |

**环境变量**（只放地址/密钥，算法参数仍在 config）：

| 变量 | 作用 |
|---|---|
| `MEMORYSUM_DASHSCOPE_BASE_URL` | DashScope 主机（不带路径）。默认 `https://dashscope.aliyuncs.com` |
| `MEMORYSUM_DASHSCOPE_API_KEY` | 优先读取 |
| `DASHSCOPE_API_KEY` | 上面的变量未设置时的回退 |

> **必须用专用 MaaS 端点**：2026-10-20 实测，公共端点 `https://dashscope.aliyuncs.com`
> 收到 `"dimension": 2048` 后仍返回 **1024 维**向量；专用端点
> `https://llm-5qr9sf5t4nflguf4.cn-beijing.maas.aliyuncs.com` 正确返回 2048 维。
> 服务端逐条校验维度，不匹配时返回 503 `model_provider_invalid_embedding`，
> **不会**把 1024 维向量写进按 2048 维建表的 scope。

**错误码（都是非 2xx，不静默降级）**：

| code | HTTP | 触发条件 |
|---|---|---|
| `model_provider_not_configured` | 503 | 没有配置 DashScope key |
| `model_provider_unauthorized` | 503 | key 被 DashScope 拒绝（401/403） |
| `model_provider_rate_limited` | 429 | 服务端限流 |
| `model_provider_invalid_embedding` | 503 | 条数/索引/维度不一致、非数值向量、空向量 |
| `model_provider_input_too_large` | 503 | 服务端明确拒绝超长输入 |

**切换/升级行为（重要）**：

- `pipeline_id` 与 `embedding_space_id` 都随 provider/model/instruction/dimension 变化；
  另外 `api/app.py` 会把 embedding 端点地址并入数据库指纹，所以**换 `MEMORYSUM_DASHSCOPE_BASE_URL` 也会让旧 scope 503**。
  旧 scope 会被拒绝（HTTP 503 `pipeline_mismatch`），**不会自动重建或重新向量化**。
- 换模型后需要重新 `Add` 数据（平台每轮重建 scope；本地需清库或换 user_id）。
- 只改 reranker/排序参数不需要 bump；改 embedding 必须 bump。

**回退到本地 embedding**：

1. `champion-v1.json`：bump `pipeline_id`，`embedding.provider = "local-openai"`，
   `model` 改成实际 alias，`dimension = "auto-detect-on-first-successful-embedding"`；
2. 启动本机 embedding 服务（`--embeddings --pooling last`，`:7999`）；
3. 重启 memorysum。旧 DashScope scope 同样 503，需要重新灌入。

**本地验证**：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_dashscope.py -q
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
```

## 配置

核心配置文件：`src/memorysum/config/champion-v1.json`

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `embedding.provider` | `dashscope` | embedding 供应商（`dashscope` / `local-openai` / `openrouter`） |
| `embedding.model` | `text-embedding-v4` | 当前使用的 embedding 模型 |
| `embedding.dimension` | 2048 | 显式维度；DashScope 不使用 auto |
| `embedding.batch_items` | 10 | DashScope 单请求上限也是 10 |
| `retrieval.rrf_k` | 20 | RRF 融合常数（越小头部优势越大） |
| `retrieval.rrf_k_adaptive` | true | 自适应 k（`clamp(候选/3, 10, 30)`） |
| `reranker.fusion_weight` | 2.0 | 精排 vs 检索的融合权重 |
| `groups.direct_max_tokens` | 640 | 每条记录的最大渲染 token 数 |
| `groups.time_anchor_in_text` | true | 在正文前补时间锚点 |
| `retrieval.per_source_cap` | 1 | 每个 chunk 最多返回几条 |

### 配置不生效的排查

```bash
# 检查配置是否被正确读取
grep "你的参数名" logs/memorysum.log

# 检查当前生效的配置指纹
sha256sum src/memorysum/config/champion-v1.json | cut -c1-12
```

## 测试

```bash
# 运行测试
.venv/bin/python -m pytest -q

# 代码检查
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests

# 检索质量评测（60 题 BEAM 数据集）
.venv/bin/python scripts/eval_retrieval.py
```

## 重启服务

```bash
# 找到当前进程
REAL=$(ss -tlnpH 'sport = :8001' | grep -oP 'pid=\K[0-9]+' | head -1)

# 停止（先 SIGTERM，等连接断开，再 SIGKILL）
kill -TERM $REAL; sleep 2; kill -9 $REAL 2>/dev/null

# 启动
setsid nohup .venv/bin/memorysum > logs/memorysum.log 2>&1 &
sleep 4

# 验证 PID 变了
NEW=$(ss -tlnpH 'sport = :8001' | grep -oP 'pid=\K[0-9]+' | head -1)
[ "$NEW" != "$REAL" ] && echo "✅ 重启成功" || echo "❌ 没重启成功"
```

## 目录结构

```
memorysum/
├── src/memorysum/          # 核心代码
│   ├── api/                # FastAPI 路由
│   ├── config/             # 配置文件（champion-v1.json）
│   ├── core/               # 检索、融合、排序逻辑
│   ├── providers/          # dashscope / local-openai / openrouter / split
│   ├── services/           # SearchService / AddService
│   ├── storage/            # SQLite 数据库管理
│   └── settings.py         # 配置加载
├── scripts/                # 评测和工具脚本
├── tests/                  # 测试
├── bench/                  # 本地基准测试框架
├── data/                   # 运行时数据（不提交）
├── logs/                   # 日志（不提交）
├── keys.txt                # API 密钥（不提交）
└── .env                    # 环境变量（不提交）
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/add` | 添加记忆 |
| POST | `/search` | 搜索记忆 |
| GET | `/health` | 健康检查 |

详细 API 文档：启动后访问 `http://localhost:8001/docs`（Swagger UI）。

## 常见问题

### 服务启动后 Search 返回 503

默认配置下检查两件事：

```bash
# 1) 本机 reranker
curl http://localhost:7997/health

# 2) DashScope embedding（应返回 2048 维）
curl -sS "$MEMORYSUM_DASHSCOPE_BASE_URL/api/v1/services/embeddings/text-embedding/text-embedding" \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-v4","input":{"texts":["ping"]},"dimension":2048}' | head -c 200
```

日志里的 `model_provider_*` code 会指出是哪一侧失败（见上文错误码表）。
如果改回本地 embedding，则检查 `curl http://localhost:7999/health`。

### PipelineMismatch 错误

scope 的 `pipeline_id` / `embedding_space_id` 与当前配置不匹配（例如刚从本机
Qwen3-Embedding 切到 DashScope text-embedding-v4）。这是**有意**拒绝混用向量空间，
不会自动迁移。需要重新灌入数据，或显式删除本地开发数据：

```bash
rm -f data/scopes/*.sqlite3 data/registry.sqlite3
```

### 服务报 1024 维 / `model_provider_invalid_embedding`

说明 `dimension` 没被服务端采纳。最常见原因是用公共端点
`https://dashscope.aliyuncs.com`（实测会忽略 `dimension`）。把
`MEMORYSUM_DASHSCOPE_BASE_URL` 指向专用 MaaS 部署，并确认
`champion-v1.json` 的 `embedding.dimension` 与请求一致。

### GPU 显存不足

默认只需要本机 reranker：

1. 换更小的 reranker 量化（如 Q4_K_M）；
2. 只在 CPU 上跑 reranker（`-ngl 0`，会慢很多）；
3. 如果用本地 embedding 回退，再考虑减小它的 `-c` 或换量化。

### GitHub push 超时

```bash
# 重试循环（第 4-6 次通常成功）
for i in $(seq 1 6); do
  git push origin master && break
  sleep 5
done
```

## 许可证

私有项目。