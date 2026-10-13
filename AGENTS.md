# AGENTS.md

给在本仓库工作的 AI agent 的说明。面向人的使用说明见 [README.md](README.md)，结构设计见 [docs/architecture.md](docs/architecture.md)。

## 项目是什么

原文保真的记忆服务：`Add` 写入、`Search` 检索。FastAPI + SQLite（FTS5 + sqlite-vec），模型侧依赖本机 llama.cpp 提供 embedding 与 cross-encoder rerank。**不生成答案，只返回可回溯到原文的证据片段。**

## 环境与运行

- Python 3.12（`>=3.12,<3.13`），虚拟环境在 `.venv/`
- 三个进程：memorysum `:8001`、embedding `:7999`、rerank `:7997`（llama.cpp 一实例一模型，所以 embedding 与 rerank 必须分开；启动命令见 README 的「服务拓扑」）
- 常用命令（**都用 `.venv/bin/*`，不要用 `uv run`**）：
  ```bash
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
  .venv/bin/python scripts/eval_retrieval.py          # 检索质量评测（见下）
  ```
- **禁止 `uv run` / `uv sync --frozen`**：uv.lock 记录的是被墙的 `files.pythonhosted.org` 绝对 URL，会卡死。安装依赖用
  `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install -e ".[dev]"`。

## 不可破坏的约定

1. **原文保真**：只存原始消息，窗口存 Unicode 半开区间；API 返回的正文必须是原文切片（可以加 `Source role:`/`Passage:` 渲染前缀，但不能改写正文）。
2. **参数可审计**：算法参数只在 `src/memorysum/config/champion-v1.json`；环境变量只放密钥、服务地址、数据目录、入站鉴权。**新增配置项必须让代码真的读它**（见"踩过的坑"第 4 条）。
3. **不静默降级**：模型失败必须返回非 2xx，绝不退化成"纯词法检索还报成功"。
4. **改数据契约要 bump `pipeline_id`**：embedding provider/model/instruction/dimension、chunk/window renderer、索引格式变化 → 新建版本，否则旧 scope 一律 503 `pipeline_mismatch`（不会自动退休重建）。**只改排序参数**（通道权重、融合权重、reranker 开关）**不必 bump**，已有数据可复用。

## 改排序/检索后必须做的验证

仓库自带评测器：[`scripts/eval_retrieval.py`](scripts/eval_retrieval.py)（60 条英文事实查询 + 1 个 BEAM scope，1147 窗口；离线实例化 `SearchService`，不会打扰运行中的 HTTP 服务）。

```bash
.venv/bin/python scripts/eval_retrieval.py            # 全量 + split-half
.venv/bin/python scripts/eval_retrieval.py --dump-rank # 逐条看排名
```

**必须同时看 split-half，不能只看全样本净值。** 实测教训：纯 rerank 排序的全样本成绩是"净 +1"，看着像小胜；拆成两半却是 **+4 / −3**——两半相抵的假象掩盖了它其实不可靠。凡是"用新信号取代旧信号"的改动，都要用两半各自成立来证明。

当前基准（2026-10-13，`fusion_weight=2.0`）：**top-1 38/60、top-3 48/60、MRR ≈ 0.73**（线上复测 37/48/0.721，波动来自本地 embedding 的非确定性）。

## 踩过的坑（都已实测，别重复尝试）

1. **`rrf_k` 调参无效**：10 / 20 / 30 / 60 / 100 / 200 六个值下 top-1 与 top-3 完全相同（RRF 的已知鲁棒性）。保持 60。
2. **纯 rerank 排序不可靠**：必须与检索排名融合，见 `SearchService._fuse_rankings`。`reranker.enabled=true` 时走这条路径，`fusion_weight` 默认 2.0（以共识为主、精排为辅）。
3. **别指望"换个更强的小模型"单独解决问题**：`bge-reranker-v2-m3`（568M）单独排序净 +1 且不稳定。
4. **配置项必须接线**：曾出现 `capacity.max_add_messages` 声明了、真实限制却硬编码在 `api/schemas.py` 的情况——改 config 静默无效。现在由 `test_wire_limits_track_pipeline_config` 守护。
5. **空白压缩无效**：proxy token 规则本就不计空白，压缩不改变窗口切分，端到端 net 0。
6. **窗口粒度已经是最优**：leaf 512/ctx 640 或 leaf 256/ctx 320 都掉到 29/60。
7. **`chinese_fts` 不是中文专用**：它走 jieba 分词，对英文同样产出词项（gold 召回 26/30、平均排名 4.2，优于 `english_fts`）；权重降到 0 反而掉到 32/60，所以必须保留小权重。
8. **1024 维 MRL 截断不省显存**（只省存储与 KNN 时间），且 llama.cpp 不支持服务端截断（`dimensions` 参数被忽略，实测返回 2560 维）。
9. **提高 embedding 吞吐没有配置空间**：并发 1→8、单请求批大小 1→64 的所有组合吞吐都封顶 ~21 windows/s，ubatch 2048→4096 无差异——瓶颈是 GPU 计算，不是显存/slot 数。

## 本机环境的坑

- **GitHub push 间歇性完全不通**：单次命令常失败（`GnuTLS recv error (-110)` 或 135 s 连接超时）。必须写成**重试循环**（实测第 4-6 次才通，成功时仅 2-4 秒）。`git ls-remote`（只读）通常 1 秒成功，可用它区分"网络不通"与"认证失败"。
- **HF 下载同理**：`export HF_ENDPOINT=https://hf-mirror.com` 后 `hf download` 或 `wget -c`（支持断点续传），必须带重试；直连 huggingface.co 基本不通。
- **重启服务前先看活动连接**：`ss -tn state established '( sport = :8001 )'`。有 keep-alive 连接时 uvicorn 会等满超时（实测停机 10.9 s）；无连接时 ~0.16 s。要么等连接空闲，要么 SIGTERM 后 2 秒直接 SIGKILL（~0.2 s）。**用户要求尽量别让服务停太久。**
- **`pkill -f '<pattern>'` 会杀掉执行它的 shell 自身**（pattern 出现在自己的命令行里，实测 exit 143）→ 用 `ps -eo pid,args | grep -F ... | grep -v grep | awk '{print $1}'` 取 PID，或 `pkill -x <进程名>`。
- **后台任务用 `setsid`**：`nohup` 只挡 SIGHUP，工具调用超时的 SIGTERM 会带走子进程。
- **三种服务都是裸进程**，机器重启不自动恢复（尚无 systemd unit）；memorysum 依赖 7999/7997 可达，否则 Add/Search 返回 503。
- 数据目录 `data/`、`logs/`、`keys.txt`、`.env` 都在 `.gitignore` 里；`keys.txt` 每行一个裸 Bearer token。

## 提交约定

- 提交身份用仓库既有身份：`git -c user.name="HyKeeper" -c user.email="22790389+HyKeeper@users.noreply.github.com" commit ...`（本机没配 `user.name`/`user.email`，直接 commit 会报"作者身份未知"）。
- 提交信息用英文祈使句标题 + 要点式正文，说明**改了什么、依据是什么**；实测结论要写进提交或 README。
