# 架构与实现状态

## 依赖方向

```text
api/app.py
  ├── api/schemas.py             # HTTP/Pydantic 契约
  ├── services/add.py            # Add 生命周期与原子发布
  ├── services/search.py         # 可见 epoch、召回、rerank、输出预算
  ├── providers/protocols.py     # embedding/rerank 接口
  │     ├── providers/factory.py   # 按 champion-v1.json 的 provider 名选实现
  │     ├── providers/openrouter.py # 远程 embedding/rerank（OpenRouter wire format）
  │     ├── providers/local_openai.py  # 本地端点：OpenAI 兼容 embedding + cross-encoder rerank
  │     └── providers/split.py     # embedding 与 rerank 分属不同供应商时的组合
  └── storage/database.py        # user_id 路由、SQLite 生命周期、sqlite-vec

core/
  ├── chunking.py                # 原文精确 span + 双尺度/跨消息 bounded context 窗口
  ├── atoms.py                   # 句子、标题、列表、表格、代码/引用的原文 atom
  ├── identity.py                # payload/source/scope/view 稳定身份
  ├── ranking.py                 # deterministic RRF
  ├── rendering.py               # 原文证据输出与 reader-token 估算
  ├── query_modes.py             # 固定英文/中文 wide/context 触发词
  ├── text.py                    # proxy tokenizer、jieba、cl100k 计数
  ├── vector.py                  # L2 normalization / float32 blob
  └── cache.py                   # 有界进程内 TTL/LRU
```

业务层不拼 OpenRouter URL，不直接生成回答；provider 层不访问 SQLite；存储层不知道 rerank 策略。新增实际供应商时，增加 `ModelProvider` 实现并在 `providers/factory.py` 中登记，不把供应商字段散落进 Add/Search 代码。embedding 与 rerank 可来自不同供应商：`provider: "local-openai"` 表示走本地端点（embedding 用 `MEMORYSUM_EMBEDDING_BASE_URL`，默认 `http://127.0.0.1:7999/v1`；cross-encoder rerank 用 `MEMORYSUM_RERANK_BASE_URL`，默认 `http://127.0.0.1:7997`，同时兼容 llama.cpp `--reranking` 的 `{"results":[{"index","relevance_score"}]}` 与自建 FastAPI 服务的裸数组 `[{"index","score"}]` 两种响应形状），两者由 `SplitProvider` 拼成单个 `ModelProvider`；参与 `embedding_space_id` 指纹的是 embedding 一边的端点地址。

**精排不直接取代检索排序**：`SearchService._fuse_rankings` 把 rerank 排名与 RRF 排名再融合一次（`score = 1/(k + rerank_rank) + w/(k + retrieval_rank)`，`reranker.fusion_weight` 默认 2.0），以保留多通道共识信号——实测纯 rerank 排序在两个半样本上分别是 +4 与 -3，融合后为 +4 与 0（见 README 的实测记录）。

## Add 状态流

```text
validate exact payload
  → hash(user scope + request payload)
  → register request/build fence
  → persist immutable messages + unpublished windows + embedding inputs
  → claim pending inputs (lease/fence)
  → call embedding API outside SQL transaction
  → persist normalized vectors
  → one short publication transaction:
       vec rows + both FTS rowids + active_windows + counts
       + request receipt + active_epoch + full-render estimates
  → HTTP 200
```

失败的 Add 不会发布窗口；staging 与 retry 状态保留，重发同一请求可恢复。不同 payload 复用已用 `request_id` 返回 422。scope DB 中的每个 embedding input 按 `(embedding_space_id, input_hash)` 去重；外部模型调用不承诺 exactly-once。

## Search 可见性

```text
entry snapshot (Ecap)
  ├── complete-render gate → one complete raw-memory bundle, or
  └── query embedding → dense leaf/context + English/Chinese FTS + options FTS
                     → RRF → source-span grouping → rerank
                     → whole-record token budget → final scope guard
```

向量 KNN 查询在 SQLite vec0 metadata 条件中直接限制 `published_epoch <= Ecap`；FTS、全文 bundle 与回源窗口同样按 ready request / active epoch 过滤。模型 API 等待期间发布的新 Add 不会进入本次 Search。原文读取事务在 rerank 前结束，scope 最后再做一次生命周期校验。

## 持久化与生命周期

- 全局 `registry.sqlite3` 只存 `sha256(user_id)`、pipeline、first-seen、expiry 与 scope 状态；每个用户 DB 的 `scope_manifest` 保存精确 `user_id`，防止哈希路由误用。
- user 数据文件为 `data/scopes/<scope_sha256>.sqlite3`；WAL + `synchronous=FULL`，scope/filesystem 权限尽量收紧。
- 数据 TTL 固定 30 天，不因活动续期。退休先将 registry 置为 `retired`，等待活动 SQLite connection leases 关闭，再删除 DB/WAL/SHM；registry tombstone 留存，阻止同一 user ID 复活。
- 外部请求不在 SQL 写事务期间执行。Search response cache、query-vector cache 与 full-bundle cache 均有界，cache key 含 scope hash；退休时清理对应 cache。
- 当前运行形态是单服务副本、Uvicorn `workers=1`，HTTP 端口固定 `8001`。scope connection leases 与 cache retirement 以单进程为协调边界；多副本共享 SQLite 数据目录尚未实现，不应直接横向扩容。
- Search 的 `content` 只包含原文 evidence spans；source ID、角色和时间戳保留在内部 reranker 输入中，不以重复标签污染 API 正文。每条结果返回原始 reranker `score`（降序为更相关，不同模型间不保证可比）和 UTC `created_at`（来源时间优先，缺省时回退到 Add 接收时间）。窄查询按 rerank 分数排序后由 `top_k` 限制条数；大小写/空白规范化后完全相同的正文去重，仍受服务端最大记录数与输出预算约束。小 scope 的完整 bundle 仅在宽泛/上下文查询、候选数及 rerank 预算允许时逐条评分。
- 每个 HTTP 请求默认记录一行不含正文的摘要，包含 UTC 时间戳、方法、路径、状态码与耗时；`MEMORYSUM_DEBUG` 默认关闭，设为 true 时额外记录完整 Add/Search JSON，包含消息正文、问题、options 和 payload IDs。请求 headers 与 API credentials 永不写日志。调试日志可能包含敏感正文，不应共享或长期保留。

## 版本边界

`src/memorysum/config/champion-v1.json` 是算法配置唯一来源。`pipeline_id` 和数据库 `PRAGMA user_version` 用来拒绝无迁移的旧数据。改变以下任意项都应新建 pipeline/schema 版本，不能在同一 scope 中混用：embedding provider/model/instruction/dimension、chunk/window renderer、索引格式、reranker 评分模板。

## 当前已实现

- AML Text/Coding 形状的同步 Add/Search、健康检查、Bearer 入站鉴权、请求 body 容量上限与有界队列。
- strict Pydantic 输入边界、user 隔离、request payload hash 幂等与冲突拒绝。
- 双尺度 leaf/context 窗口；source atom 保留 Python Unicode 半开区间、kind、code/quote 标记及 raw hash；`window_spans` / `window_atoms` 回连原文。
- context 窗口只跨同一 Add 请求内的消息源序，最多 4 spans；rerank 前可在组预算内扩回完整 atom，超预算则保留原窗口。
- OpenRouter dense embedding adapter、embedding batch 上限、向量校验/归一、input hash 去重、lease/fence、sqlite-vec 精确 KNN。
- FTS5 英文 + Jieba/中文 bigram，options lexical 提名，固定 RRF，bounded rerank 批次。
- 仅 wide/context 查询可命中小 scope 完整 bundle gate；bundle 按独立原文消息返回，仍受 `top_k` 与 reader-token 预算约束。
- SQLite vec0 / FTS5 本地启动 probe；fake provider 下的 API、幂等、隔离、检索、缓存与退休测试。

## 下一阶段开发项

1. **模型合同**：当前 embedding provider 仅提供 dense vector，维度自动发现；没有原生 sparse postings。reranker 是临时 VL 模型，无法核对冻结版要求的 Qwen3 tokenizer、官方 yes/no pair template SHA 与 24k/48k 的精确真实 token 数。现在使用 `cl100k_base` 代理数，故配置明确为 `provisional`。
2. **结构关系**：heading/table/list/exception/explicit-reference links 尚未建；atom kind 已保存但还没有 relation-aware 召回，embedded-dialogue attribution 识别也未实现。
3. **时间与修订**：时间戳已保存，并作为内部检索/排序元数据；API 正文不再拼接重复时间标签。尚未实现 temporal candidate buckets 或 strict asserted-revision edges；自动 truth supersede/forget 仍保持关闭。
4. **个性化归因**：`is_personal_query` 检测函数已存在但未接入 Search；direct-user-form 候选保底和 embedded-dialogue 防误归因尚未实现。
5. **写入任务优化**：SQLite 已按 input hash 保存 embedding 状态，但还没有跨进程输入级 single-flight worker pool，也没有 query 同 key 并发 single-flight。
6. **输出组与容量**：跨消息 groups 已限制到 4 spans 并扩展 parent atoms；完整 structure paths、oversize atom 的显式 omission telemetry 尚未完成。还没有 1k/10k/50k/100k scope p50/p95、100k DB bytes 外推、Full 吞吐或供应商配额压测。

继续对齐 Champion v1 规格前，需要先补齐并验证这些项目；本仓当前作为本地开发原型运行。
