# Agent-TripPilot

Agent-TripPilot 是一个基于 LangChain、RAG、本地 BGE 向量模型和 MCP 工具的智能出行策划与执行 Agent。

项目目标是跑通一个可解释的出行规划闭环：

```text
用户自然语言需求
-> 参数抽取
-> RAG 检索城市旅行知识
-> MCP 查询天气、POI、酒店价格
-> 预算估算
-> 行程生成
-> Reflection 质检
-> 可选 Seedream 封面图
-> Markdown 导出
```

## 当前能力

已完成：

- 本地 RAG：杭州、长沙旅行知识种子文档
- 本地向量模型：默认使用 `BAAI/bge-small-zh-v1.5`，路径通过 `.env` 配置
- Chroma 向量库：持久化到 `data/chroma_db`
- 参数抽取：自然语言转 `TripRequest`
- 多轮记忆：保存对话历史，并维护结构化出行需求状态
- 参数确认：正式生成前确认日期、人数、天数、交通方式、预算范围等必要信息
- 高德 MCP：天气、POI 工具接入
- 酒店 MCP：酒店搜索和价格查询接入，支持 ModelScope 代理或 RollingGo 官方接口
- 预算估算：城市消费系数 + 酒店 MCP 中位参考价
- 运行时上下文：注入当前时间，支持相对日期理解
- 计算工具：支持安全基础数学计算
- Plan-and-Solve：Planner 输出工具计划，Executor 按计划选择工具
- Planner 输出容错：字符串步骤会自动规范化，避免模型格式波动导致崩溃
- ReAct 调试轨迹：显示 Thought、Action、Observation
- Reflection 迭代：最多 2 轮质检和自动优化
- 草案/正式分层：粗略规划跳过 Reflection，正式报告进入完整质检
- 生成后追问：酒店、住宿、预算、路线等补充问题默认走 follow-up，不重新生成完整行程
- 模型分工：DeepSeek 负责主规划和反思；豆包 mini 负责轻量理解、抽取和后续多模态；Seedream 负责可选封面图
- 导出可选：只有用户明确要求“导出/保存”才写入 `outputs/trip_plan.md`
- Streamlit UI：聊天、参数表单、调试轨迹、Markdown 下载和封面图生成
- 本轮评分：UI 展示需求理解、约束满足、工具使用、输出质量等 8 维评分
- 产品化交互：左侧参数表单和快捷调整，中间对话，右侧 tabs 展示结果、状态、过程、评分和封面

## 快速开始

激活环境：

```text
conda activate agent
```

构建 RAG 向量库：

```text
python -m trip_pilot.rag.build_index
```

测试 RAG：

```text
python -m trip_pilot.rag.retriever
```

测试 MCP：

```text
python -m trip_pilot.app.test_mcp_calls
```

运行固定 Demo：

```text
python -m trip_pilot.app.demo
```

运行交互式 CLI：

```text
python -m trip_pilot.app.cli
```

运行 Streamlit UI：

```text
python -m trip_pilot.app.run_ui
```

如果一定要直接用 Streamlit，也建议使用下面这种方式，避免 Windows 下 `streamlit.exe` 绑定到旧虚拟环境：

```text
python -m streamlit run trip_pilot/app/streamlit_app.py
```

UI 不依赖 CLI。两者只是不同入口，都会调用同一个 `TripPilotConversationAgent`。

如果 UI 内显示“运行失败”，右侧会出现“错误详情”，可展开查看完整 traceback。项目已在 `.streamlit/config.toml` 中关闭 Streamlit 文件监听，避免 Windows + 本地模型环境下出现不稳定的 watcher 错误。

关闭 UI：

- 如果 UI 是在当前终端前台启动的，按 `Ctrl+C`。
- 如果浏览器关了但服务还在，说明 Streamlit 服务器进程仍在运行。先powershell查看：

```text
Get-Process python,streamlit -ErrorAction SilentlyContinue
```

再按具体 PID 关闭：

```text
Stop-Process -Id 你的PID
```

CLI 支持普通聊天和多轮补充，例如：

```text
你好
我想去杭州玩
两天，两个人，喜欢文化和美食
生成行程
坐普速火车，1500包含往返大交通
导出 Markdown
```

## 环境变量

复制 `.env.example` 为 `.env`，并填写本地配置：

```text
api_key=your_deepseek_api_key
base_url=https://api.deepseek.com
model_id=deepseek-chat

api_key_ark=your_volc_ark_api_key
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MULTIMODAL_MODEL=doubao-seed-2-0-mini-260428
ARK_IMAGE_MODEL=doubao-seedream-4-5-251128
ARK_IMAGE_SIZE=1920x1920

GAODE_MCP_URL=your_gaode_mcp_url
GAODE_MCP_TRANSPORT=sse

HOTEL_MCP_URL=your_hotel_mcp_url
HOTEL_MCP_TRANSPORT=streamable_http
HOTEL_MCP_BACKEND=auto
HOTEL_MCP_AUTH_TOKEN=

BGE_MODEL_PATH=D:\models\bge-small-zh-v1.5
CHROMA_DIR=./data/chroma_db
```

Seedream 4.5 对图片尺寸有最低像素要求，`1024x1024` 会被拒绝。默认使用 `1920x1920`，刚好满足至少 `3686400` 像素的要求。

## 隐私与安全

- 不要提交 `.env`、真实 MCP URL、API Key、Bearer Token、生成报告、生成图片或本地向量库。
- `.gitignore` 已默认忽略 `.env*`、`outputs/`、`data/chroma_db/`、`data/processed/` 和 Streamlit secrets。
- `.env.example` 只保留占位配置，用于说明变量名。
- 如果使用 RollingGo 官方酒店 MCP，请把 `HOTEL_MCP_AUTH_TOKEN` 放在 `.env` 中，不要写入代码或文档。

## 项目结构

```text
trip_pilot/
  config.py
  models.py
  schemas.py
  mcp_client.py
  memory.py

  rag/
    embeddings.py
    loaders.py
    vector_store.py
    build_index.py
    retriever.py

  tools/
    budget_tools.py
    image_tools.py
    gaode_mcp_tools.py
    hotel_mcp_tools.py
    runtime_tools.py
    rag_tools.py
    export_tools.py

  agents/
    conversation_agent.py
    constraint_checker.py
    extractor_agent.py
    trip_agent.py
    reflection_agent.py

  evaluation/
    evaluator.py

  app/
    streamlit_app.py
    cli.py
    demo.py
    list_mcp_tools.py
    inspect_mcp_tools.py
    test_mcp_calls.py
```

## 开发文档

详细开发说明统一维护在：

- `docs/PROJECT_DEVELOPMENT.md`
