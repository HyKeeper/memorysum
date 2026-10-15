# MemorySum

原文保真的记忆检索服务：`Add` 写入、`Search` 检索。FastAPI + SQLite（FTS5 + sqlite-vec），模型侧依赖本机 llama.cpp 提供 embedding 与 cross-encoder rerank。**不生成答案，只返回可回溯到原文的证据片段。**

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│  Client (Add / Search)                                  │
└────────────────────┬────────────────────────────────────┘
                     │ :8001
┌────────────────────▼────────────────────────────────────┐
│  MemorySum Service (FastAPI + SQLite)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ FTS5     │  │ sqlite-vec│  │ Reranker │              │
│  │ (词法)   │  │ (向量)   │  │ (精排)   │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │              │              │                   │
└───────┼──────────────┼──────────────┼───────────────────┘
        │              │              │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │ FTS5    │    │Embedding│    │ Reranker│
   │ (内嵌)  │    │  :7999  │    │  :7997  │
   └─────────┘    └─────────┘    └─────────┘
```

### 检索流程

1. **5 路召回**：leaf_dense / context_dense / english_fts / chinese_fts / options_lexical
2. **RRF 融合**：`rrf_k=20` + 自适应（`k = clamp(候选数/3, 10, 30)`）
3. **Reranker 精排**：bge-reranker-v2-m3（cross-encoder）
4. **最终排序**：`score = 1/(k + rerank_rank) + fusion_weight/(k + retrieval_rank)`

## 硬件要求

| 组件 | 最低显存 | 推荐显存 | 说明 |
|---|---|---|---|
| Embedding（Qwen3-Embedding-4B-Q8_0） | 6 GB | 8 GB | 含 KV cache |
| Reranker（bge-reranker-v2-m3-F16） | 1 GB | 2 GB | cross-encoder |
| **合计** | **7 GB** | **10 GB** | 两个模型同时加载 |

实测：RTX 4060 Ti 16GB，embedding ~8 GB + reranker ~1 GB = 9 GB，空闲 ~6 GB。

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

```bash
# Embedding 模型（Q8_0 量化，4 GB）
mkdir -p models
wget -c https://hf-mirror.com/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf \
  -O models/Qwen3-Embedding-4B-Q8_0.gguf

# Reranker 模型（F16，~2 GB）
wget -c https://hf-mirror.com/BAAI/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-f16.gguf \
  -O models/bge-reranker-v2-m3-f16.gguf
```

### 3. 启动服务

需要 **3 个终端**（或 3 个后台进程）：

```bash
# 终端 1：Embedding 服务（:7999）
llama-server -m models/Qwen3-Embedding-4B-Q8_0.gguf \
  --embeddings --pooling last -fa on \
  -c 32768 -b 2048 -ub 2048 \
  -ngl 99 --host 127.0.0.1 --port 7999 --alias qwen3-embedding-4b

# 终端 2：Reranker 服务（:7997）
llama-server -m models/bge-reranker-v2-m3-f16.gguf \
  --reranking --pooling rank -fa on \
  -c 8192 -b 8192 -ub 8192 \
  -ngl 99 --host 127.0.0.1 --port 7997 --alias bge-reranker-v2-m3

# 终端 3：MemorySum 服务（:8001）
.venv/bin/memorysum
```

**一键后台启动**（适合生产环境）：

```bash
# Embedding
setsid nohup llama-server -m models/Qwen3-Embedding-4B-Q8_0.gguf \
  --embeddings --pooling last -fa on -c 32768 -b 2048 -ub 2048 \
  -ngl 99 --host 127.0.0.1 --port 7999 --alias qwen3-embedding-4b \
  > logs/embedding.log 2>&1 &

# Reranker
setsid nohup llama-server -m models/bge-reranker-v2-m3-f16.gguf \
  --reranking --pooling rank -fa on -c 8192 -b 8192 -ub 8192 \
  -ngl 99 --host 127.0.0.1 --port 7997 --alias bge-reranker-v2-m3 \
  > logs/reranker.log 2>&1 &

# MemorySum
mkdir -p logs
setsid nohup .venv/bin/memorysum > logs/memorysum.log 2>&1 &
```

### 4. 验证

```bash
# 健康检查
curl http://localhost:8001/health
# → {"status":"ok"}

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

## 配置

核心配置文件：`src/memorysum/config/champion-v1.json`

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
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

检查 embedding 和 reranker 服务是否启动：
```bash
curl http://localhost:7999/health  # embedding
curl http://localhost:7997/health  # reranker
```

### PipelineMismatch 错误

scope 的 `pipeline_id` 与当前配置不匹配。删除旧数据：
```bash
rm -f data/scopes/*.sqlite3 data/registry.sqlite3
```

### GPU 显存不足

1. 减少 embedding 的 context 长度：`-c 32768` → `-c 8192`
2. 使用更小的量化模型（Q4_K_M）
3. 只在 CPU 上跑 reranker（`-ngl 0`，会慢很多）

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