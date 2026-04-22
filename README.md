# Nike Multi-Agent 智能客服

> 关键词：**Qwen / Planner / Multi-Agent / RAG / Tool Calling / Long-term Memory / Evaluation / Observability**

这是一个面向电商客服场景的 AI Agent 项目，强调“**可演示 + 可评测 + 可复现 + 可讲清楚工程设计**”。

## 1. 项目能力

- **结构化 Planner**：将用户请求转为可执行计划（agent/use_rag/tool_calls/response_style）
- **多角色 Agent**：sales / after_sales / promo / inventory
- **RAG 检索链路**：向量召回 + BM25 召回 + Hybrid 融合（RRF）+ reranker 重排
- **结构化工具调用**：inventory / order / promotion，带安全执行器
- **长期记忆**：会话历史 + 摘要记忆持久化
- **可观测性**：trace_id + 分步耗时（plan/rag/tools/llm/summary/total）
- **评测体系**：agent/router/rag 全链路离线评测并产出 JSON 报告

## 2. 架构流程

1. 接收用户请求
2. Planner 生成结构化计划（带 fallback）
3. 按计划执行 RAG 与工具
4. 组装上下文，调用 Qwen 生成回答
5. 更新长期记忆与摘要
6. 返回 response + trace_id + timings

## 3. 目录结构

- `app/main.py`：FastAPI 入口、接口与 health
- `app/services/agent_orchestrator.py`：主编排逻辑
- `app/services/planner.py`：计划生成与兜底策略
- `app/services/tool_executor.py`：工具安全执行器
- `app/services/rag.py`：检索与重排
- `app/services/memory.py`：长期记忆
- `app/services/schemas.py`：结构化 schema
- `scripts/eval_full.py`：全量评测脚本

## 4. 快速开始

### 4.1 本地运行

```bash
conda create -n nike-agent python=3.10 -y
conda activate nike-agent
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

访问：`http://localhost:8000`

### 4.2 Docker 运行

```bash
docker compose up --build
```

健康检查：`GET /health`

## 5. 环境变量

`.env`（基于 `.env.example`）至少配置：

- `QWEN_API_KEY`
- `QWEN_BASE_URL`（默认 DashScope 兼容地址）
- `QWEN_MODEL`（默认 qwen-plus）

RAG 相关配置：

- `QWEN_EMBED_MODEL`：Qwen embedding 模型名（默认 `text-embedding-v3`）
- `QWEN_RERANK_MODEL`：Qwen 重排模型名（默认 `qwen-rank`）
- `RAG_TOP_K`：最终返回文档数（默认 `3`）
- `RAG_CANDIDATE_K`：向量候选数（默认 `8`）
- `RAG_BM25_CANDIDATE_K`：BM25 候选数（默认 `8`）

## 6. 评测

### 6.1 基础组件评测

```bash
python scripts/eval_agent.py
python scripts/eval_router.py
python scripts/eval_rag.py
```

`eval_rag.py` 会输出 `Hybrid/Vector/BM25` 三类 Hit@3 指标，便于观察混合召回收益。

### 6.2 端到端评测（E2E）

```bash
python scripts/eval_e2e.py
```

输出：`reports/e2e_eval_report.json`

核心指标：
- `task_success_rate`
- `factual_pass_rate`
- `hallucination_rate`
- `policy_violation_rate`

### 6.3 工具正确性评测（Tool Quality）

```bash
python scripts/eval_tool.py
```

输出：`reports/tool_eval_report.json`

核心指标：
- `tool_selection_precision`
- `tool_selection_recall`
- `tool_arg_valid_rate`
- `tool_forbidden_violation_rate`
- `missing_args_ask_rate`

### 6.4 线上效果聚合报告（Online KPI）

```bash
python scripts/eval_online_report.py
```

默认读取：`reports/online_events.jsonl`

输出：`reports/online_eval_report.json`

核心指标：
- `resolution_rate`
- `handoff_rate`
- `tool_error_rate`
- `p95_latency_ms`
- `avg_cost_per_session`
- `csat`

### 6.5 全量汇总评测

```bash
python scripts/eval_full.py
```

输出：`reports/full_eval_report.json`

汇总包含：
- `agent`（规划准确率）
- `router`（路由准确率）
- `rag`（命中率）
- `e2e`（端到端任务成功）
- `tool_quality`（工具调用质量）

### 6.6 SFT 训练数据构建与检查

构建 Planner SFT 数据：

```bash
python scripts/build_sft_planner_dataset.py
```

输出：
- `data/sft/planner_train.jsonl`
- `data/sft/planner_hard_negative.jsonl`
- `data/sft/planner_train_split.jsonl`
- `data/sft/planner_valid_split.jsonl`

说明：
- `planner_train.jsonl` 由评测集 + 线上高质量样本构建
- `planner_hard_negative.jsonl` 收集未解决、工具报错或失败阶段样本，便于后续对比学习/人工清洗
- 每条样本附带 `meta.sample_weight`，可用于训练时加权（hard case 权重更高）
- 自动生成 `train/valid` 切分，便于直接进入训练流程

训练前分布检查：

```bash
python scripts/build_sft_stats.py
```

输出：`reports/sft_planner_stats.json`

导出训练框架常用格式（input/target/weight）：

```bash
python scripts/export_sft_for_trainer.py
```

输出：
- `data/sft/trainer/planner_train_trainer.jsonl`
- `data/sft/trainer/planner_valid_trainer.jsonl`

建议检查项：
- 各 agent 样本占比是否失衡
- `tool_calls` 分布是否过度集中
- `use_rag` 是否几乎全为 True（必要时补充反例）


## 7. 下一步优化

- 增加在线 AB 实验与自动回归
- 增加缓存层（RAG/LLM）
- 接入 Prometheus/OpenTelemetry
- 扩展多模态商品图检索
