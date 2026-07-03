# Agent-TripPilot 项目详细开发文档

## 1. 项目定位

Agent-TripPilot 是一个基于 LangChain、RAG、本地向量模型和 MCP 工具的智能出行策划与执行 Agent。

目标不是只生成一段旅游攻略，而是完成一个可解释的工作流：

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

第一阶段已经跑通 CLI、Streamlit UI 和 Markdown 输出，后续再继续增强路线图、图表和 DOCX/HTML 导出。

## 2. 当前项目状态

已完成：

- 本地 RAG：读取杭州、长沙 Markdown 种子文档
- 本地向量模型：`BAAI/bge-small-zh-v1.5`
- Chroma 向量库存储与检索
- 参数抽取 Agent：自然语言转 `TripRequest`
- 会话 Agent：支持普通聊天、多轮补充、参数确认和记忆
- 行程生成 Agent：结合 RAG、预算、MCP 信息生成 Markdown
- ReAct 调试轨迹：显示 Thought、Action、Observation
- Reflection Agent：最多 2 轮检查和自动优化
- 生成后补充信息：默认基于上一版局部修订，避免重复跑完整流程
- 生成后追问：例如“住西湖附近，按预算推荐酒店”会作为 follow-up 处理，优先查询酒店 MCP，不重新生成完整行程
- 粗略规划模式：用户说“粗略/大概/先看看”时允许先生成草案，并标记待确认信息
- Planner 控制 Executor：Planner 输出 `tool` 字段，Executor 只执行计划中选择的工具
- Planner 输出兼容：如果模型返回字符串步骤，会自动规范化为 `step/name/tool/reason`
- 运行时工具：注入当前时间，支持安全数学计算
- 模型分工：DeepSeek 负责主规划、正式报告、Reflection；豆包 mini 负责意图识别、参数抽取、轻量聊天和后续多模态；Seedream 负责可选封面图
- 状态管理：新增 `ConversationStateSnapshot`，显式保存用户偏好、当前需求、当前方案、待确认项和工具摘要
- 约束校验：新增 `validate_constraints()`，在生成前检查预算、偏好、节奏、人数、天数和待确认项
- 工具稳定性：新增会话级工具缓存，减少重复 MCP 调用；工具失败时进入保守降级
- 输出体验：支持 brief/standard/detailed 三种回复粒度
- 评估反馈：UI 展示 8 维本轮评分表
- 酒店 MCP：已接入，可查询酒店候选和价格
- 高德 MCP：已接入，可查询天气和 POI
- Markdown 导出：默认不导出，用户明确要求后输出到 `outputs/trip_plan.md`
- Streamlit UI：支持侧边栏参数录入、聊天、调试轨迹查看、Markdown 下载和封面图生成

仍需增强：

- POI 详情和点位经纬度用于路线排序
- 酒店结果结构化摘要
- 高德路线规划正式接入主流程
- 工具路由细化：根据 Planner 参数进一步控制 POI 关键词、距离计算对象和预算拆分
- 多模态输入：上传景点截图、酒店截图或票据后由豆包 mini 识别
- 更丰富的 RAG 原始资料

## 3. 目录结构

```text
Agent-TripPilot/
  .env                 # 本地密钥与 MCP URL，不提交
  .env.example         # 配置模板
  README.md
  requirements.txt

  data/
    raw/               # RAG 原始文本
    chroma_db/         # Chroma 向量库

  docs/
    PROJECT_DEVELOPMENT.md

  outputs/
    trip_plan.md

  trip_pilot/
    config.py          # 配置读取
    models.py          # LangChain 模型工厂
    schemas.py         # Pydantic 数据结构
    mcp_client.py      # MCP 工具加载和调用
    memory.py          # LangChain 会话记忆和结构化状态

    rag/
      embeddings.py    # 本地 BGE embedding 封装
      loaders.py       # Markdown 加载和切分
      vector_store.py  # Chroma 创建/读取
      build_index.py   # 构建向量库
      retriever.py     # 检索测试

    tools/
      rag_tools.py
      budget_tools.py
      image_tools.py
      gaode_mcp_tools.py
      hotel_mcp_tools.py
      export_tools.py

    agents/
      conversation_agent.py
      constraint_checker.py
      extractor_agent.py
      trip_agent.py
      reflection_agent.py

    app/
      streamlit_app.py
      cli.py
      demo.py
      list_mcp_tools.py
      inspect_mcp_tools.py
      test_mcp_calls.py

    evaluation/
      evaluator.py
```

## 4. 环境配置

虚拟环境：

```text
conda activate agent
```

`.env` 至少需要：

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

注意：不要把 `.env` 内容发到公开仓库或文档里。

## 5. 运行命令

构建 RAG 向量库：

```text
python -m trip_pilot.rag.build_index
```

测试 RAG 检索：

```text
python -m trip_pilot.rag.retriever
```

列出 MCP 工具：

```text
python -m trip_pilot.app.list_mcp_tools
```

查看 MCP 工具参数：

```text
python -m trip_pilot.app.inspect_mcp_tools
```

测试 MCP 调用：

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

备用命令：

```text
python -m streamlit run trip_pilot/app/streamlit_app.py
```

UI 不依赖 CLI。CLI 和 UI 都只是入口层，核心都调用 `TripPilotConversationAgent`。

如果 UI 内部报错，右侧“错误详情”会显示完整 traceback。项目已通过 `.streamlit/config.toml` 关闭 Streamlit 文件监听，避免 Windows + 本地模型环境下 watcher 扫描 torch/FlagEmbedding 对象导致不稳定异常。

关闭 UI：

- 前台运行时按 `Ctrl+C`。
- 关闭浏览器标签页只会断开页面，不会停止 Streamlit 服务。
- 如果有后台残留，使用 `Get-Process python,streamlit -ErrorAction SilentlyContinue` 查看 PID，再用 `Stop-Process -Id 你的PID` 停止。

CLI 支持普通聊天和多轮补充：

```text
用户：你好
Agent：正常问候，不强制规划

用户：我想去杭州玩
Agent：追问旅行天数和人数

用户：两天，两个人，喜欢文化和美食
Agent：记录需求，但不默认生成报告

用户：生成行程
Agent：如果还缺正式规划参数，继续追问

用户：坐普速火车，1500包含往返大交通
Agent：进入 Plan/ReAct/Reflection 流程

用户：把第二天改轻松一点
Agent：基于上一版行程局部修订，不重新跑完整工具链
```

## 6. RAG 设计

当前 RAG 资料：

- `data/raw/hangzhou_travel_seed.md`
- `data/raw/changsha_travel_seed.md`

`source_manifest.md` 只作为资料说明，不进入向量库。

当前策略：

- 稳定知识进 RAG：城市定位、经典景点、文化路线、美食街区、注意事项
- 实时信息走工具：天气、酒店价格、路线、距离、票价、开放时间

本地 BGE 模型封装在 `trip_pilot/rag/embeddings.py`，避免依赖 OpenAI Embeddings。

## 7. MCP 设计

当前接入两个 ModelScope MCP。

高德 MCP 当前支持的工具：

- `maps_weather`
- `maps_text_search`
- `maps_geo`
- `maps_direction_walking`
- `maps_direction_driving`
- `maps_direction_transit_integrated`
- `maps_distance`

酒店 MCP 当前支持的工具：

- `searchHotels`
- `getHotelDetail`
- `getHotelSearchTags`

酒店后端支持两种模式：

- ModelScope 代理：继续使用现有 `HOTEL_MCP_URL`，无需额外 token，但偶尔会在 streamable_http 会话收尾时出现 `502 Bad Gateway` 或 `Session termination failed`。
- RollingGo 官方接口：设置 `HOTEL_MCP_URL=https://mcp.rollinggo.cn/mcp`、`HOTEL_MCP_BACKEND=rollinggo`、`HOTEL_MCP_AUTH_TOKEN=你的 Bearer Token`。该模式按文档直接走 JSON-RPC，包含 `Accept: application/json, text/event-stream` 和 `Authorization` 请求头。

当前酒店工具已做兼容：

- 使用 `filterOptions.starRatings` 做星级筛选。
- 支持 `placeType`，例如城市、景点、机场、地铁站。
- 酒店结果按“有则使用、无则忽略”解析，兼容不同返回字段。
- 查询失败时降级为城市消费系数估算，不中断主行程规划。

当前已落地：

- `get_weather_via_gaode`
- `search_poi_via_gaode`
- `geocode_via_gaode`
- `distance_via_gaode`
- `search_hotels_via_mcp`
- `query_hotel_reference_price`
- `get_current_time`
- `calculate`

后续计划：

- 用 `maps_geo` 把景点名转经纬度
- 用 `maps_distance` 或路线规划工具计算景点间距离
- 用 `maps_direction_transit_integrated` 规划公交/地铁路线
- 用酒店 MCP 的价格结果改进住宿预算和住宿区域建议

## 8. 预算策略

旧版预算工具只按“低/中/高”固定估算，不够合理。

当前预算分三层：

1. 城市消费系数：不同城市餐饮、市内交通、门票使用不同系数。
2. 酒店 MCP 实时价格：如果酒店 MCP 可用，优先查询酒店候选价格。
3. 中位参考价：预算不使用最低价，而使用候选酒店的中位参考价，避免被偏远或低质量酒店误导。

预算输出会明确说明：

- 住宿价格是否来自酒店 MCP
- 本地消费估算是否粗略
- 是否不含往返大交通

## 9. Agent 工作流

### 9.0 模型路由

文件：`trip_pilot/models.py`

当前模型分工：

- `get_chat_model()`：DeepSeek，负责正式行程生成、Refine、复杂文本推理。
- `get_fast_model()`：优先豆包 mini，负责意图识别、参数抽取、普通聊天；如果没有 `api_key_ark`，自动回退 DeepSeek。
- `get_multimodal_model()`：豆包 mini，多模态输入预留入口。
- `get_reflection_model()`：DeepSeek 低温输出，负责 Reflection 稳定质检。
- `get_image_client()`：火山 Ark OpenAI-compatible 客户端，给 Seedream 图片工具使用。

这样做的原因是：主规划需要稳定推理和长文本输出；抽取和聊天需要速度与成本；图片生成不应进入默认规划链路，只在用户明确触发时调用。

注意：图片模型需要在 Ark 控制台单独开通。如果返回 `ModelNotOpen`，不是本地代码问题，而是账号尚未激活该模型服务。

Seedream 4.5 图片尺寸至少需要 `3686400` 像素，因此默认 `ARK_IMAGE_SIZE=1920x1920`。如果改成 `1024x1024` 会触发 `InvalidParameter`。

### 9.1 多轮会话与记忆

文件：`trip_pilot/agents/conversation_agent.py`

主入口是 `TripPilotConversationAgent`。

它维护两类记忆：

- LangChain `InMemoryChatMessageHistory`：保存用户和助手消息
- `TripRequest`：保存结构化出行需求状态

每轮输入都会先做意图识别：

- `chat`：普通聊天，不强行规划
- `trip_plan`：新的旅行规划需求
- `update`：补充或修改已有需求
- `export`：导出文件
- `cancel`：重置状态

### 9.2 参数抽取

文件：`trip_pilot/agents/extractor_agent.py`

当前没有使用 `with_structured_output`，因为部分模型的 tool calling / thinking mode 兼容性不稳定。现在采用：

```text
JSON 提示词 -> json.loads -> Pydantic 校验
```

### 9.3 参数确认

正式生成前必须确认：

- 目的地城市
- 出发日期
- 旅行天数
- 出行人数
- 如果有始发地，需要确认往返交通方式
- 如果有预算，需要确认预算是否包含往返大交通

缺少这些信息时，Agent 只追问，不调用 RAG/MCP/LLM 生成正式行程。

例外：如果用户明确说“粗略”“大概”“先看看”，Agent 可以先生成草案，但必须标记待确认字段，不能把草案当最终方案。

### 9.4 Plan-and-Solve

正式生成时先进入 Planner。Planner 会基于当前需求和可用工具生成步骤，例如：

```text
1. 检索目的地 RAG 旅行知识
2. 查询高德天气
3. 查询高德 POI
4. 查询酒店 MCP 参考价
5. 估算预算
6. 生成行程初稿
7. Reflection 质检并优化
```

Planner 输出必须包含 `tool` 字段。Executor 只执行计划中选择的工具。

示例：

```json
[
  {"step": 1, "name": "获取当前时间", "tool": "time", "reason": "解析本周末"},
  {"step": 2, "name": "检索旅行知识", "tool": "rag", "reason": "获取稳定知识"},
  {"step": 3, "name": "查询天气", "tool": "weather", "reason": "安排室内外活动"}
]
```

如果 Planner 输出不是合法 JSON，会使用内置保底计划。

如果 Planner 返回的是字符串数组，例如 `["查询酒店", "估算预算"]`，系统会自动根据关键词补齐 tool 字段，避免执行阶段崩溃。

### 9.5 ReAct 执行

执行阶段显式打印：

```text
Thought: 为什么要调用这个工具
Action: 工具名和参数
Observation: 工具返回结果
```

当前 ReAct 执行会调用：

- `time`：当前时间和本周末参考日期
- `rag`：RAG 检索
- `weather`：高德天气
- `poi`：高德 POI
- `geocode`：地址转经纬度
- `distance`：高德距离测量
- `hotel`：酒店 MCP
- `budget`：预算工具
- `calculate`：安全数学计算

当前已减少无效 POI 查询：不再默认搜索目的地城市名，而是根据偏好选择“博物馆、历史文化街区、美食街、小吃、地铁站附近景点”等更具体关键词。

当前仍有一个设计缺口：Planner 能控制工具类别，但还没有细化到每个工具的参数对象，例如 `distance` 需要哪些 POI 的经纬度。后续应让 Planner 输出更结构化的参数建议。

### 9.6 草案与正式报告

生成模式分两层：

- `draft`：用户说“粗略/大概/先看看”时进入草案模式，可以缺少部分正式参数，跳过 Reflection。
- `official`：正式行程报告，必须确认关键参数，生成后进入 Reflection 质检和优化。

### 9.7 Reflection 迭代

文件：`trip_pilot/agents/reflection_agent.py`

检查点：

- 预算是否明显不合理
- 路线是否绕路
- 时间是否过紧
- 是否满足用户偏好
- 是否过度确定实时信息
- 是否缺少预约、天气、交通拥堵等提醒

新版主流程在 `conversation_agent.py` 中执行 Reflection 迭代：

- 默认最大迭代次数：2
- 每轮 Reflection 输出 JSON
- `need_refine=true` 时调用 Refine Agent 修改
- `need_refine=false` 时终止
- 达到最大迭代次数也终止，提示人工确认剩余实时信息

终止条件：

- 没有严重问题
- 只剩需要用户或官方渠道确认的实时信息
- 达到最大迭代次数

### 9.8 导出策略

默认不导出文件。

只有用户明确说导出、保存、下载、生成文件时，才写入 `outputs/trip_plan.md`。

## 10. 当前调试结论

本地开发阶段已验证：

- RAG 能命中杭州、长沙知识
- 高德天气工具可用于补充天气信息
- 高德 POI 工具可用于补充点位信息
- 酒店 MCP 可用于补充酒店候选和价格参考；远程服务偶发失败时会降级为预算估算
- “你好”普通聊天不会崩
- 信息不足时会追问
- 多轮补充后能保留结构化记忆
- 正式生成前会确认交通方式和预算范围
- 粗略规划会进入草案模式，跳过 Reflection
- Planner 输出的工具列表会影响 Executor 实际工具调用
- 当前时间会注入运行时上下文，避免模型只依赖静态知识
- 正式生成时能显示 Plan/ReAct/Reflection 调试轨迹
- 完整 Demo 可导出 Markdown 行程
- Streamlit UI 已加入，适合后续调试和展示
- 已修复 Planner 字符串步骤导致的 `.get` 崩溃
- 已支持完整行程后的酒店/住宿追问，不再默认重跑完整规划链路

需要注意：

- 远程 MCP 调用结束时偶尔出现 `Session termination failed: 404/502`，但如果前面工具结果已经返回，可以暂时忽略。
- Windows 下 `conda run` 可能出现 GBK 编码问题，建议激活环境后直接运行 `python -m ...`。
- Codex 沙箱内 Chroma/SQLite 可能需要提权，本机终端一般不受影响。

## 11. 下一步开发计划

优先级 1：

- 细化 ToolRouter 参数，让 Planner 不只选择工具类别，还能指定工具参数
- 将高德路线规划正式并入 `conversation_agent.py`
- 根据 POI 经纬度和距离优化每日游览顺序
- 将 Reflection 反馈结构化保存，便于 UI 展示

优先级 2：

- 接入 `maps_geo` 和 `maps_distance`
- 对每日景点顺序做距离检查
- Reflection 中加入“路线距离证据”

优先级 3：

- 增加 `TripPlan` 结构化输出
- 增加 UI 上传图片，多模态识别酒店、票据或景点截图

优先级 4：

- 增加更多城市 RAG 文档
- 增加网页搜索工具补充票价和开放时间
- 导出 DOCX 或 HTML 报告
