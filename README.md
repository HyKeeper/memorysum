# MemorySum

面向记忆检索系统开发的原文记忆服务，提供与 AML Text Add / Search 兼容的 API 形状。**不生成答案，只返回可回溯到原文的证据片段。**

> **当前算法版本：`memorysum-hybrid-preview-v0.5.0-qinstr2`（provisional）**
> **embedding**：`Qwen3-Embedding-4B-Q8_0.gguf`（2560 维、`--pooling last`、`:7999`）。embedding 输入**逐条请求**（`embedding.batch_items=1`）——GPU 批推理 run-to-run 非确定（同文本跨批向量差可达 5e-3），逐条后向量跨轮 bit 一致。
> **rerank**：`jina-reranker-v3.5`（listwise 双塔：qwen3 backbone + 独立 projector MLP + 余弦评分）。常驻栈 = fork llama.cpp（PR ggml-org#26286，qwen3 SWA pattern + `--output-token-ids`）的 `llama-server :7995`（backbone）+ `:7997 rerank_server.py`（FastAPI 包装）。**同请求两次执行 (index, score) bit 级相同**，`reranker.score_quantization=0`。
> **排序**：5 路召回（`leaf_dense 96 / context_dense 48 / english_fts 80 / chinese_fts 24 / options_lexical 16`）→ RRF（`rrf_k=20` + `rrf_k_adaptive=true`，`k=clamp(候选/3,10,30)`）→ 精排排名与 RRF 排名再融合（`score = 1/(k + rerank_rank) + w/(k + retrieval_rank)`，`fusion_weight=2.0`）。所有排序 tie-break 用**内容哈希**（`_content_key`），不用 view_id/window_id 等物理 id——物理 id 每轮 scope 重建必变，只有内容跨轮稳定。
> 冻结规格要求的 text-embedding-v4 512 维稠密+稀疏向量尚未接入；当前版本不能描述成已通过正式模型验收。

## 服务拓扑（4 个常驻进程，全是裸进程，机器重启不自动恢复）

| 服务 | 端口 | 模型 | 显存 |
|---|---|---|---|
| memorysum | 8001 | — | — |
| embedding | 7999 | Qwen3-Embedding-4B-Q8_0（官方 llama.cpp） | ~8.0 GB |
| rerank server | 7997 | jina-reranker-v3.5（FastAPI，跑在 `/home/rain/thx01-venv`） | ~0（GPU 在 7995） |
| rerank backbone | 7995 | jina-reranker-v3.5-BF16（fork llama-server） | ~4.9 GB 常驻 |

```bash
# embedding（-b 必须等于 -ub；embeddings 模式下 llama.cpp 会强制覆盖并告警）
llama-server -m /home/rain/model/gguf/Qwen3-Embedding-4B-Q8_0.gguf \
  --embeddings --pooling last -fa on -c 32768 -b 2048 -ub 2048 \
  -ctk q8_0 -ctv q8_0 -ngl 99 --host 127.0.0.1 --port 7999 --alias qwen3-embedding-4b

# jina backbone（fork 版才有 --output-token-ids；构建见 PATCH.md）
cd /home/rain/model/llama.cpp-qwen3-swa
./build/bin/llama-server -m /home/rain/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5-GGUF/snapshots/884f7c67aa3ac24edb89064da8c7bfd03f4a90f5/jina-reranker-v3.5-BF16.gguf \
  --embeddings --pooling none -fa on -c 16384 -np 1 -b 16384 -ub 16384 \
  --output-token-ids 151670,151671 -ngl 99 --host 127.0.0.1 --port 7995 --alias jina-backbone

# rerank server（7995 之后起）
cd /home/rain/model/jina-v3.5 && /home/rain/thx01-venv/bin/python rerank_server.py

# memorysum（最后起；依赖 7999/7997 可达，否则 Add/Search 503）
.venv/bin/memorysum
```

- **显存实测**：常驻 ~13.2GB（embedding 8.0 + backbone 4.9 + 桌面），worst-case 请求峰值 13343/16311 MiB，余 ~3GB。
- **fork 补丁与等价性验证**：`/home/rain/model/llama.cpp-qwen3-swa/PATCH.md`（`--output-token-ids` 过滤、`-DCMAKE_CUDA_ARCHITECTURES=89`——nvcc 12.4 不认 sm_120a、SWA cache 在 ctx 39k~50k 的 SIGSEGV 规避）。
- **性能（BEAM-60）**：p50 6.3s/查询；块容量 16384/ubatch 16384（整块单批）。不要回到 per-block subprocess 形态（每块重载权重，p50 16.5s 且进程时有时无）。

### bge 回退（原 reranker，仍可用）

```bash
llama-server -m /home/rain/model/gguf/bge-reranker-v2-m3-f16.gguf \
  --reranking --pooling rank -fa on -c 8192 -b 8192 -ub 8192 -ngl 99 \
  --host 127.0.0.1 --port 7997 --alias bge-reranker-v2-m3
```

同时把 `champion-v1.json` 的 `reranker.model` 改回 `bge-reranker-v2-m3`、`reranker.score_quantization` 改回 `0.1`（bge 的 GPU 批推理有 ±0.02~0.06 的 run-to-run 波动，量化是必需品；jina 天然确定所以关）。注意 bge 的 rank 模式 `-ub` 必须 ≥ 最坏 pair 长度（`single_pair_proxy_tokens=4608` × 标定最坏倍率 1.479 ≈ 6.8k token，8192 贴顶）。

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

成功只在全部必需 embedding、向量和 FTS 索引发布后返回 HTTP 200，且原样回显 `request_id`、`user_id`、`session_id`。同一用户同一 `request_id`、同一规范 payload 幂等；payload 不同返回 422。`timestamp` 可省略，但出现时必须是整数毫秒。未知字段被忽略（不 422），`top_k` 严格整数类型。

### Search

```json
{
  "query": "What do I prefer?",
  "options": ["A. Tea", "B. Coffee"],
  "user_id": "opaque-user-id",
  "top_k": 100
}
```

Search 每条结果包含 `id`、`content`、`score`、`created_at` 和 `source_ids`：

- `score` 是精排与 RRF 融合后的排名分，**恒正**；不同模型/版本间不保证可直接比较。
- `source_ids` 是该命中涉及的 chunk 级 harness 标识（LoCoMo `D1:3`、BEAM `1:msg7`、PersonaMem `m3`、流式任务 `ALF#86`），来自 Add 时对元数据头的一次性解析（`core/metadata.py`，只读、不改写原文）。平台优先用条目上的 id 定位 gold，正文 META 头只是兜底——这是机制性 Recall 的前提。
- `created_at` 为 UTC ISO 8601（优先 Add 来源 timestamp）；`content` 是原文证据片段（可带 `Session N (date)` 形态的时间锚点行——平台会删掉 `Time:` 前缀的行，所以不用那种形态）。

窄查询走混合检索 + rerank，按融合分排序后由 `top_k` 限制；完全重复的正文合并；`top_k=0` 返回空数组；未知用户空间返回 `{"data": []}`；不同 `user_id` 的数据不互相检索。所有 query pair 都超预算时退回融合排序（`strategy=hybrid_retrieval_unreranked` + WARNING），不返回空数组。有序图片 `ContentPart[]` 属于多模态赛道，本版本不实现。

## 快速开始

要求 Python 3.12。

```bash
cd memorysum
cp .env.example .env
# .env 填 OPENROUTER_API_KEY 与 MEMORYSUM_API_KEYS_FILE；keys.txt 每行一个 Bearer token 本体
# 依赖安装（直连 PyPI 会被墙，走清华镜像；禁止 uv sync --frozen）
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install -e ".[dev]"
# 启动前先拉起 embedding / rerank（见上节）
.venv/bin/memorysum
```

服务固定监听 `0.0.0.0:8001`；`GET /health` 无鉴权，Add/Search 必须 `Authorization: Bearer <keys.txt 中一行>`。`keys.txt` 已在 `.gitignore`，权限 600，启动时读取、改后需重启。

## 检索质量评测

```bash
.venv/bin/python scripts/eval_retrieval.py             # 全量 + split-half
.venv/bin/python scripts/eval_retrieval.py --dump-rank # 逐条看排名
```

**必须同时看 split-half**——全样本净值会被两半相抵骗到（历史实例：+4/−3 合成"净 +1"）。

当前基准（2026-10-15，**jina + 旧 query instruction**，BEAM scope 1147 窗口 + 60 条英文查询）：
**txt_all top-1 49/60、top-3 56/60、MRR 0.8822；ids_all top-1 39/60（历史最好）、top-3 50/60、MRR 0.7459**
（ids split-half 19/20 两半成立）。注意：`qinstr2`（新 query instruction）上线后**未重测**——BEAM 的库向量不变（document 侧无指令）但 query 向量已变，数字会漂；重测需重建/复用 scope（实测旧 scope 未被 503 拦截，行为待查）。

历史参考：bge-reranker-v2-m3 时代同口径 txt1 47~49、ids1 37~39（其中 jina 上线当天 ids 39 首次突破）；纯 RRF 基线 34/60。**任何"用新信号取代旧信号"的改动都要两半各自成立 + 跨对话验证**（单对话调出的权重是典型过拟合）。

## 实现原则

- **Search 只返回历史证据**，不回答问题、不调生成式模型。
- **原文优先**：存原始消息，窗口存 Unicode 半开区间，输出可回溯原文。
- **按 `user_id` 物理隔离**：一个用户一个 SQLite（路径只含 user id 的 SHA-256）；请求幂等、epoch 发布、FTS/向量同事务发布。
- **模型可替换**：HTTP 服务只依赖 `ModelProvider` 协议；本地 OpenAI 兼容端点在 `providers/local_openai.py`，远程在 `providers/openrouter.py`，按 provider 名由 `factory.py` 组装（`SplitProvider` 可拆 embedding/rerank 两家）。
- **不静默降级**：模型失败返回非 2xx，不把模型失败伪装成词法检索成功。
- **参数可审计**：算法参数只在 `champion-v1.json`，环境变量只放密钥/服务地址/数据目录/入站鉴权。改 embedding provider/model/instruction/dimension 或 renderer **必须 bump `pipeline_id`**；只改排序参数不必。

## 开发

```bash
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
.venv/bin/python -m pytest -q
```

测试用本地 fake provider，不请求外部服务、不含私有数据。

## 资料与平台对接

- [Agent Memory Leaderboard API Guide](https://agentmemories.ai/api-guide) / [Rules](https://agentmemories.ai/rules)
- 平台分数与逐叶子：`.venv/bin/python scripts/fetch_platform_results.py --leaves/--diff`（详见 `docs/platform-api-notes.md`——哪些能实测、哪些只是文档声明）
- 算法/跑分台账：`docs/score_ledger.md`、`AGENTS.md`（给 agent 的实测教训与约定）
