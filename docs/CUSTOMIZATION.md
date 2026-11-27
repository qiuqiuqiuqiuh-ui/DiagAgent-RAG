# DiagAgent-RAG 二次开发与工程定制指南

本指南面向希望将 **DiagAgent-RAG** 直接作为工业设备/垂直领域诊断模板复用的工程师，按改造深度分为 **L1（零改代码）**、**L2（轻度扩展）** 与 **L3（深度集成）** 三档路径。

---

## 一、三档适配路径

### L1：零改代码快速上线（约 10 分钟）

适用于：替换自有设备手册、故障码表、传感器阈值表，并切换为中文/英文/双语在线模型。

1. **准备配置 Profile**：复制 `configs/production_bilingual.yaml`（中文/双语）或 `configs/production.yaml`（英文），修改：
   - `domain.name` 与 `domain.language`（设为 `zh` 时，工作流固定拒答/越界/升级文案及重试扩展词自动切换为中文，Demo UI 预设也会按 `/v1/meta` 动态生成中文示例）；
   - `paths.corpus_dir`、`paths.index_dir`、`paths.cache_file`；
   - `rules.fault_codes_path`、`rules.sensor_thresholds_path`（纯手册无规则表时设 `rules.allow_missing_rule_tables: true`）；
   - `llm`、`judge_llm`、`embedding`、`reranker` 的模型与网关地址；
   - `api.api_key` / `api.admin_api_key`（建议放 `.env` 或环境变量 `DIAGAGENT_API__*`）以及按需配置 `api.enable_demo_ui: false`。
2. **替换语料目录 `data/raw/`**：
   - 支持 UTF-8 Markdown（扫描 PDF 建议先由 PaddleOCR / MinerU 导出为带 `## Page N` 二级标题的 Markdown 以保留页码 provenance）、文本 PDF、CSV 与 JSON；
   - 规则表：`fault_codes.csv`（列：`fault_code,equipment_id,severity,meaning,first_check,escalation`）与 `sensor_thresholds.json`（字段：`equipment_id,sensor,unit,warning_threshold,critical_threshold,direction`）。
3. **准备评测集 `data/eval/`**（JSONL，每行符合 `EvalCase` Schema）并执行上线验证：
   ```powershell
   python scripts/validate_corpus.py --config configs/your_profile.yaml
   python scripts/ingest.py --config configs/your_profile.yaml
   python scripts/evaluate.py --config configs/your_profile.yaml --dataset data/eval/your_golden.jsonl --judge --require-heterogeneous-judge --save-answers
   # 若为纯手册无规则表语料，追加 --allow-no-tools
   python scripts/check_eval_gate.py --report <evaluation.json> --mode hybrid_rerank
   ```
4. **启动或容器化部署**：
   - 本地：`python scripts/run_api.py --config configs/your_profile.yaml --host 0.0.0.0 --port 8000`
   - 若部署在可信 TLS 反向代理之后，追加 `--forwarded-allow-ips <proxy_ip>` 使限流按真实客户端 IP 生效。

---

### L2：轻度业务扩展（修改规则 Schema、单位、分块与 Prompt）

适用于：已有业务字段与默认表格不完全一致，或需定制领域推理规则。

1. **扩展故障码 / 传感器规则 Schema（例如支持双边正常区间 `[low, high]`）**：
   - 在 `src/agents/schemas.py` 扩展 `FaultCodeRule` / `SensorThresholdRule` 字段与校验器；
   - 在 `src/agents/tools.py` 的 `DiagnosticTools.evaluate()` 实现对应判定逻辑（例如双向超限告警）。
2. **新增自定义行业单位**：
   - 在 `src/agents/tools.py` 的 `_SENSOR_UNITS` 中注册规范单位及其别名；
   - `src/agents/citations.py` 已复用 `KNOWN_SENSOR_UNITS`，新增单位后会自动免于被误判为非法引用方括号。
3. **调整路由、合成与 Critic 审查 Prompt**：
   - 修改 `src/agents/policy.py` 中 `LLMDiagnosticPolicy.route()` / `synthesize()` / `critique()` 的系统指令；
   - 若策略行为有实质性变更，建议同步递增 `src/api/cache.py` 中 `profile_hash()` 的策略版本标识（如 `llm-critic-v4` → `llm-critic-v5`），自动隔离旧策略缓存。
4. **调整中文/技术文档分块策略**：
   - 修改 `src/ingestion/chunking.py` 的 `_SEPARATORS` 与 `configs/*.yaml` 中的 `ingestion.chunk_size` / `chunk_overlap`。

---

### L3：深度系统集成（实时数据源、外部工单与多轮对话）

适用于：需要将诊断智能体接入企业 SCADA、时序数据库（InfluxDB/TDengine）、CMMS/工单系统或支持多轮追问。

1. **接入实时遥测 / 工单工具**：
   - 在 `src/agents/tools.py` 封装只读查询适配器（注意设置超时、降级与脱敏）；
   - 在 `src/agents/schemas.py` 新增对应的 `ToolFinding` 判别联合分支；
   - 在 `src/agents/workflow.py` 的 `_run_tools()`（或新增独立 LangGraph 节点）中调用并将结果注入 `EvidenceContext`。
2. **引入多轮追问与会话记忆**：
   - 在 `src/agents/workflow.py` 编译 `StateGraph` 时传入 LangGraph `Checkpointer`（如 `SqliteSaver` / `PostgresSaver`），在请求 Schema 增加 `thread_id` 并通过 `config={"configurable": {"thread_id": ...}}` 调用；
   - 注意同步在 `src/api/cache.py` 中调整多轮对话缓存策略（对多轮上下文请求跳过单轮缓存或将对话摘要纳入 `context_hash`）。
3. **接入企业统一认证（JWT / OAuth2 / 网关鉴权）**：
   - 替换或扩展 `src/api/security.py` 中的 `AccessControl.check()` 与 `ProtectedRoute`，保留请求体解析前鉴权和限流的执行顺序。

---

## 二、常见定制需求 → 文件速查表

| 定制需求 | 需修改的文件 |
|---|---|
| 切换模型、网关、语言、路径、限流、Demo UI 开关 | `configs/*.yaml`、`.env`、`src/core/settings.py` |
| 替换设备手册、OCR Markdown、故障码表与阈值表 | `data/raw/*`（无需改代码） |
| 调整文档解析（Markdown / PDF / CSV / JSON） | `src/ingestion/parsers.py`、`src/ingestion/preflight.py` |
| 调整分块边界或中英文分词规则 | `src/ingestion/chunking.py`、`src/core/tokenizer.py` |
| 调整混合检索候选数、RRF 常数或重排逻辑 | `src/retrieval/hybrid.py`、`src/retrieval/bm25.py`、`src/retrieval/dense.py` |
| 扩展故障码/传感器规则判定或新增行业单位 | `src/agents/schemas.py`、`src/agents/tools.py` |
| 修改意图路由、回答合成或 Critic 审查 Prompt | `src/agents/policy.py`、`src/api/cache.py`（策略版本号） |
| 修改多语言固定文案、重试次数或 LangGraph 拓扑 | `src/agents/workflow.py` |
| 调整 API 路由、鉴权、限流、缓存隔离或指标 | `src/api/app.py`、`src/api/security.py`、`src/api/cache.py`、`src/api/metrics.py` |
| 定制前端 Demo 样式或交互行为 | `src/api/static/index.html`、`src/api/static/app.js`、`src/api/static/styles.css` |
| 调整离线评测指标、Judge Prompt 或上线门禁阈值 | `src/evaluation/metrics.py`、`src/evaluation/runner.py`、`src/evaluation/gate.py` |
| 调整容器构建、反代参数或 CI 流程 | `Dockerfile`、`docker-compose.yml`、`scripts/run_api.py`、`.github/workflows/ci.yml` |
