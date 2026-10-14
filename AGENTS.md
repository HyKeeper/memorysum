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

5. **`rrf_k` 调参无效**：10 / 20 / 30 / 60 / 100 / 200 六个值下 top-1 与 top-3 完全相同（RRF 的已知鲁棒性）。保持 60。
6. **纯 rerank 排序不可靠**：必须与检索排名融合，见 `SearchService._fuse_rankings`。`reranker.enabled=true` 时走这条路径，`fusion_weight` 默认 2.0（以共识为主、精排为辅）。
7. **别指望"换个更强的小模型"单独解决问题**：`bge-reranker-v2-m3`（568M）单独排序净 +1 且不稳定。
8. **配置项必须接线**：曾出现 `capacity.max_add_messages` 声明了、真实限制却硬编码在 `api/schemas.py` 的情况——改 config 静默无效。现在由 `test_wire_limits_track_pipeline_config` 守护。
9. **Search 必须回传 `source_ids` 与时间锚点**：平台的 gold 定位优先读 item 上的 id（`source_ids` → `sourceIds` → `chunk_ids` → `ids` → …），只在兜底时解析正文里的 META 头；正文带头的窗口只占约 29%，所以把 id 落库并显式返回是机制性的 Recall 前提，别把它当成可选装饰。锚点必须原样返回（`Session 1 (…)`），**不能加 `Time:` 前缀**（平台会删掉该行）。
10. **空白压缩无效**：proxy token 规则本就不计空白，压缩不改变窗口切分，端到端 net 0。
11. **窗口粒度已经是最优**：leaf 512/ctx 640 或 leaf 256/ctx 320 都掉到 29/60。
12. **`chinese_fts` 不是中文专用**：它走 jieba 分词，对英文同样产出词项（gold 召回 26/30、平均排名 4.2，优于 `english_fts`）；权重降到 0 反而掉到 32/60，所以必须保留小权重。
13. **1024 维 MRL 截断不省显存**（只省存储与 KNN 时间），且 llama.cpp 不支持服务端截断（`dimensions` 参数被忽略，实测返回 2560 维）。
14. **`PRAGMA user_version` 会加入未提交事务**：写 schema 迁移时，`ALTER TABLE` 是被隐式提交的，而 PRAGMA 的修改会随连接关闭被回滚——不显式 `commit()` 就导致"列加了但版本号没变"，每次打开都重复迁移、重复回填。见 `_migrate_v2_to_v3`。
15. **提高 embedding 吞吐没有配置空间**：并发 1→8、单请求批大小 1→64 的所有组合吞吐都封顶 ~21 windows/s，ubatch 2048→4096 无差异——瓶颈是 GPU 计算，不是显存/slot 数。
16. **给排序"加料"（相似度分数 / chunk 共识 / 说话人）在 80 题上有效、全量 150 题上无效**：`similarity_weight`（把 dense 余弦分数从 RRF 里救回来）、`consensus_weight`（同 chunk 候选互相投票）、`speaker_weight`（问题点名的人优先）都实现了，开关默认 0。80 题分层抽样时各自能做出 +1~2pp，全量后回到噪声（±1 题）。**不要靠小样本给这些信号开开关。**
17. **邻居扩展收益极小**：`neighbor_expand_chars` 200/600/1500/3000/6000 全试过，txt_all 最多 +1.6pp 且 3000 后饱和，而每条变长会挤掉 token 预算内的条数。默认 0。
18. **规则式多查询扩展更差**：去停用词 / 切前后半后各自检索再合并，ids_all single 92.5% > vote 88.8% > union 86.2%。规则变体破坏语义、合并又稀释原排序；没有 LLM 就别做查询改写。
19. **单对话调出来的权重会过拟合**：`context_dense` 1.0→2.0 在 conv-26 上 ids_all +3.8pp / txt_all +6pp，看着很香；换 conv-30（我按同格式自建 scope）立刻 −1.2pp，三折下来是负的。**只在 conv-26 有效的参数不许上生产**，先跨对话验证。
20. **`term_coverage_weight`（问题实词在窗口里的覆盖率加分）同样只在单对话有效**：conv-26 上 80 题 ids_all +1.3pp、MRR +0.067（+9%），看着是最漂亮的一个；换 conv-42/50/49/44 各 40 题立刻变成 1 正 1 平 2 负（平均 −0.6pp）。MRR 倒是稳定小升（平均 +0.014），如果哪天要优化"第一条对不对"而不是 Recall@10，可以再拿出来。默认 0。
21. **`groups.direct_max_tokens` 是"平台截断与否"的赌注，不是单纯的改进**：640→384 让 4096 预算内能塞的记录从 ~7.5 条涨到 ~14 条。四个对话实测：**截断口径下 4/4 提升**（ids_all +3.3 / +6.0 / +6.4 / +9.0pp），**不截断口径下 4/4 下降**（−0 ~ −3.2pp，txt_all −3.3~−7.1pp）。只要平台真截断的概率超过约 26%，期望值就是正的；本机无法验证平台行为，所以保持 640，等平台侧反馈再定。
22. **两段式输出（前 N 条全量 + 尾部裁剪）在截断口径下最强但差异太大**：`head_records=3 + tail_cut_chars=800` 在 4096 预算下 ids_all 92.7%（比 baseline 高 6pp），但不截断口径下 txt_all 只有 78.7%（比 baseline 低 16.6pp）——两头不讨好，默认关闭。
23. **`per_source_cap=1`（每个 chunk 最多返回 1 条）是本轮唯一经跨对话验证的改进**：conv-26 +2.5pp、conv-30 0、conv-41 +0.7pp（ids_all），无负向。机理是避免一个话多的 chunk 占满 top-k，让多跳题的第二处证据有位置。

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
