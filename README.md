# MemorySum

面向记忆检索系统开发的原文记忆服务，提供与 AML Text Add / Search 兼容的 API 形状。当前仓库侧重本机开发与真实服务链路验证，不代表任何比赛成绩。

> **当前算法版本：`memorysum-openrouter-preview-v0.2.0`（provisional）**
> 本地开发暂用 OpenRouter 的 `liquid/lfm-2.5-embedding-350m:free` 与 `nvidia/llama-nemotron-rerank-vl-1b-v2:free`。免费路由的可用性、吞吐与数据处理条款尚未完成正式评估。冻结规格要求的 text-embedding-v4 512 维稠密+稀疏向量、Qwen3-Reranker 官方模板/tokenizer 校验尚未接入。因此配置会自动探测维度、稀疏通道关闭、pair tokens 使用 `cl100k_base` 估算，不能把当前版本描述成已通过正式模型验收。

## 实现原则

- **Search 只返回历史证据**，不回答问题、不调用生成式模型，也不把答案伪装成记忆。
- **原文优先**：存储原始消息，窗口只保存指向原文的 Unicode 半开区间；输出可回溯到原消息。
- **按 `user_id` 物理隔离**：每个用户使用一个 SQLite 文件，路径只包含 user ID 的 SHA-256；请求幂等、epoch 发布、FTS/向量安装在同一发布事务中。
- **模型可替换**：HTTP 服务只依赖 `ModelProvider` 协议，OpenRouter 的路径、请求体和响应解析集中在 `providers/openrouter.py`。
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

响应只包含 `data[].id` 与 `data[].content`；`top_k=0` 返回空数组，未知用户空间也返回 `{"data": []}`。不同 `user_id` 的数据不会互相检索。文本接口只接受字符串 `query` / `content`；有序图片 `ContentPart[]` 属于多模态赛道，本版本不实现。

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

默认 `MEMORYSUM_DEBUG=false`，不会记录完整请求。开发时设置 `MEMORYSUM_DEBUG=true`，会将 Add/Search 的完整 JSON（正文、问题、options、user_id、request_id、session_id）输出到 stdout，便于本机调试。日志不会包含 HTTP headers、Authorization、OPENROUTER_API_KEY 或其他密钥。调试日志可能包含敏感正文，请仅在本机使用，避免共享或长期保留。

如需查看真实请求 JSON，可在本机开发时开启 `MEMORYSUM_DEBUG=true`；不要在共享机器或日志长期留存环境开启。

## 当前垂直链路

1. 严格校验 AML 文本 Add / Search 字段、消息角色、时间戳和容量边界。
2. 以确定性 embedding-token 代理构造 leaf/context 双尺度窗口；句子、标题、列表、表格、代码和引用会映射为精确原文 atom，context 窗口只在同一 Add 请求中按源序跨消息组合，最多 4 个 source spans。
3. OpenRouter embedding 批量请求；校验 vector 个数、维度、finite/norm，并按 float32 归一化存入 `sqlite-vec`。
4. SQLite FTS5 英文词法 + Jieba/中文 bigram 词法与双尺度 dense 召回，按 RRF 排名；options 作为受限词法提名。
5. 对同消息相交窗口合并；跨消息 context groups 保留清晰 span 边界并受窗口数限制，再经 OpenRouter rerank；任何批次失败都不返回部分排序。
6. 小空间满足完整渲染 gate 时返回完整原文 bundle，不做“前缀全文”；普通检索按完整证据组与 prompt 预算选择输出。
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

API 端口固定为 **8001**；健康检查 `/health` 无鉴权，Add/Search 必须使用 `Authorization: Bearer ...`。云服务器需在安全组开放 TCP 8001；生产访问建议放在 HTTPS 反向代理后。`MEMORYSUM_DEBUG=true` 会将请求正文写入 stdout，仅用于受控开发调试。

## 规则与设计资料

- [Agent Memory Leaderboard API Guide](https://agentmemories.ai/api-guide)
- [Agent Memory Challenge Rules](https://agentmemories.ai/rules)
- Champion v1 冻结规格由项目维护者提供，作为本仓初始实现输入；它不是官方比赛说明或成绩报告。
- 参考了项目维护者的本地评测与方法资料；该资料和其中的评测数据均未复制或打包进本仓。

本仓提供可复用的 Add/Search API 服务骨架。AML API Guide / Rules 用作接口和设计参考；若在其他比赛使用，应按对方的数据留存、鉴权与模型规则单独配置。
