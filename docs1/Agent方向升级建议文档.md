# Agent 平台升级建议文档

> **生成时间**：2026-06-23
> **当前分支**：`graduation-project`（已交付的毕设版本，冻结）
> **升级目标分支**：`agent-platform`（新建，用于 LangGraph Agent 化演进）
> **升级性质**：非破坏性演进，不动 `graduation-project`，新分支独立迭代

---

## 目录

- [一、升级背景与整体判断](#一升级背景与整体判断)
- [二、LangGraph Agent 化：赞同与陷阱](#二langgraph-agent-化赞同与陷阱)
- [三、语义切割方案辨析](#三语义切割方案辨析)
- [四、依赖与工程风险](#四依赖与工程风险)
- [五、渐进式迁移路径（三阶段）](#五渐进式迁移路径三阶段)
- [六、分支管理策略](#六分支管理策略)
- [七、关键工程细节清单](#七关键工程细节清单)
- [八、风险清单与应对](#八风险清单与应对)
- [九、验收标准](#九验收标准)

---

## 一、升级背景与整体判断

### 当前架构的本质

当前 `router.py` + `chain.py` 本质上是一个**手写的、硬编码的状态机**：

```
query → route_query() → 分发到 4 条链路 → 每条链路各自调 LLM
```

四种路由：`rag` / `chitchat` / `general` / `database`，每条链路独立实现。

### 当前架构的天花板

这种结构有一个根本缺陷：**只能走预设的单一路径**。典型翻车场景：

> 用户问："知识库里 LangChain 的用法，顺便统计一下提到它的文档有多少篇"

这同时需要 RAG 检索 + Text-to-SQL 统计，当前架构只能二选一。

### 演进判断

`chain.py` 已有 400+ 行，`_handle_database_query`、`_chat_stream`、`_chat_stream_database` 等分支纠缠，**已经在向 Agent 演化的边缘**，只是没用 Agent 框架表达。LangGraph 的图模型能把隐式状态变成显式节点，是合理的下一步。

**核心判断：方向正确，但"用 LangGraph + LangChain"背后藏着两个不同性质的决定，要分开评估。**

---

## 二、LangGraph Agent 化：赞同与陷阱

### ✅ 赞同 Agent 化的部分

Agent 模式天然解决"多步推理、跨工具组合"的问题：LLM 自主决定调哪些 tool、调几次、怎么组合。对于以下场景，Agent 明显优于硬编码路由：

- 多步查询（先检索后统计、先理解后翻译）
- 条件分支（根据检索结果决定是否追问）
- 跨工具组合（RAG + SQL + 计算）

### ⚠️ 最大陷阱：不要为了 Agent 而 Agent

Agent 不是银弹。**当前的四分类路由在很多场景下比 Agent 更好**：

| 维度 | 当前路由分发 | Agent 自主决策 |
|------|------------|--------------|
| 延迟 | 1 次路由调用（规则层零成本拦截） | 至少多 1-2 轮 LLM 推理（tool selection） |
| 成本 | 固定 1 次 LLM | Agent loop 可能 3-5 次 LLM |
| 可控性 | 路径确定，好调试 | LLM 可能"想歪了"，难复现 |
| 简单查询 | "你好"秒回 | "你好"也要走一遍 Agent 思考 |

**结论：闲聊、单一检索这种高频简单场景，路由分发永远比 Agent 快且便宜。**

### 🎯 正确架构：混合模式

不要"全部改成 Agent"，而是**混合**：

```
简单意图（chitchat / general / 单纯 RAG）→ 保留现有快速路由
复杂意图（多步推理 / 跨工具组合 / 不确定走哪）→ 进 Agent
```

LangGraph 完全能表达这种"先路由，部分进 Agent 子图"的结构。

**保留 `router.py` 的规则层作为 Agent 的前置网关**——它零成本拦截了大量简单流量，这个价值不能丢。

### 🚧 迁移要注意的三个工程问题

**1. 流式输出协议对接成本高**

当前前端靠 `[SOURCES]...[/SOURCES]` 自定义协议 + SSE 流式。LangGraph 的流式是按节点/按 token 两种粒度，**和现有前端协议不是一回事**。迁移时这块要重写对接，不是包装一下就行。

**2. 记忆系统要重构**

当前 `ConversationMemory` 是自实现 JSON 持久化。LangGraph 有自己的 state 管理（checkpoint + memory）。两套记忆系统并存会乱：

- **要么全迁**——LangGraph 接管所有状态
- **要么明确边界**——LangGraph 管 Agent 内部状态，现有 memory 管跨会话持久化

**3. 调试难度跳一个量级**

当前链路出问题，看日志就能定位 router 还是 retriever。Agent 出问题要追"LLM 为什么决定调这个 tool""为什么循环了 3 次"——**没有 trace 工具会非常痛苦**。

**建议提前接入 LangSmith 或 OpenTelemetry**，别等上线才发现排障困难。

---

## 三、语义切割方案辨析

### ⚠️ 常见误解

"语义切割"这个表述容易被理解为"基于语义的切割"，但 **LangChain 的 `RecursiveCharacterTextSplitter` 不是语义切割，是"结构感知的字符切割"**：

| 切割方式 | 原理 | LangChain 对应类 |
|---------|------|-----------------|
| 当前做法 | 固定 500 字符硬切 | （无，自实现 `chunker.py`） |
| **RecursiveCharacterTextSplitter** | 按分隔符优先级递归切（段落 > 句子 > 字符） | `RecursiveCharacterTextSplitter` |
| **真正的语义切割** | 用 embedding 相似度，语义跳变处切 | `SemanticChunker` |

### 方案对比

**`RecursiveCharacterTextSplitter` 已经比当前的好很多**（优先在 `\n\n` → `\n` → `。` 处切，不切断句子），但**基于标点，不是基于语义**。

**真正的 `SemanticChunker` 听起来美好，但有两个坑**：

1. **开销巨大**：切一个文档要调用 N 次 embedding API（每个句子一次）
2. **质量不稳定**：相似度阈值难调，切出来的块大小差异极大，影响检索

### 🎯 建议：用 RecursiveCharacterTextSplitter

理由：

- Recursive 已解决当前最大痛点（切断中文句子）
- 开销和现在一样（零额外 API 调用）
- 切块大小可控，检索友好
- 真正的语义切割工业界 ROI 低，用得少

**关键配置——必须加中文分隔符**，LangChain 默认是英文标点：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
    # 关键：中文标点优先级要靠前
)
```

这一行配置就解决了当前 `chunker.py` 的核心问题。

### ⚠️ 注意：换切割器后必须重新全量入库

切割方式改变后，`chunk_id` 计算逻辑变化，旧向量作废。**切换前必须清空 Milvus + 重新入库所有文档**。否则新旧切片共存，检索结果混乱。

---

## 四、依赖与工程风险

### LangChain 生态依赖管理是公认痛点

1. **版本碎片化**：拆成 `langchain-core` / `langchain` / `langchain-community` / `langchain-experimental` / `langgraph`，版本号各自演进，组合不当就冲突
2. **API 频繁破坏性变更**：0.1 → 0.2 → 0.3 多次 breaking change，`pip install langchain` 装到的版本和看的教程可能 API 都不一样
3. **和现有依赖冲突**：项目已用 `pydantic==2.6.0`、`openai==1.7.1`、`httpx==0.26.0`，LangChain 生态对这些有版本要求，很可能要升级一圈

### 实操建议

- **锁版本**：用 `pip-tools` 或 poetry 生成精确 lock 文件，记录每个子包精确版本
- **隔离**：独立 venv 给 Agent 平台，和现有 RAG 服务隔离
- **最小依赖**：初期只装真正需要的子包（`langgraph` + `langchain-core` + `langchain-text-splitters`），不要一开始就 `pip install langchain` 全家桶

### 推荐版本起点（2026-06 时点）

| 包 | 用途 | 选型理由 |
|----|------|---------|
| `langgraph` | Agent 图编排 | 核心依赖 |
| `langchain-core` | 基础抽象（Runnable/LanguageModel） | 轻量，被 langgraph 依赖 |
| `langchain-text-splitters` | 语义切割 | 独立轻量子包，不含重型依赖 |
| `langchain-openai` | LLM 封装（可选） | 如果用 OpenAI 兼容客户端 |

**避免引入**：`langchain-community`（臃肿）、`langchain-experimental`（不稳定）。这些包拖进来一堆传递依赖，是依赖冲突的源头。

---

## 五、渐进式迁移路径（三阶段）

### 核心原则：不要一次性重写

"All in LangChain/LangGraph"最后被框架绑架、改不动了的项目太多了。**渐进迁移、每步可回滚、每步可验证**，比"全面升级"稳妥得多。

### 第一阶段（1 周）：语义切割先行，独立可验证

**目标**：零风险换取检索质量提升，建立迁移信心。

**范围**：
- 只替换 `embeddings/chunker.py` → 用 `RecursiveCharacterTextSplitter`
- 不引入 LangGraph，只装 `langchain-text-splitters`（轻量子包）
- 切完重新全量入库，跑检索质量对比

**步骤**：

1. 新建 `agent-platform` 分支（见第六章）
2. `pip install langchain-text-splitters`，锁版本
3. 重写 `chunker.py`，对外 API 保持兼容（`chunk()` / `chunk_with_pages()` 签名不变）
4. 清空 Milvus + 重新入库所有文档
5. 用同一批测试 query 跑检索质量对比（召回率、平均相似度、主观判断）

**验收标准**：
- ✅ 切割不再出现"句子被切断"的切片
- ✅ 检索召回率不下降
- ✅ 现有四条路由链路全部能正常工作
- ✅ 前端功能无变化

**这一步零风险，纯收益。**

### 第二阶段（2-3 周）：Text-to-SQL 链路试点 Agent 化

**目标**：在最复杂的单条链路上验证 Agent 模式的可行性和稳定性。

**为什么选 Text-to-SQL 先迁**：这条链路本身就是"多步推理"（表结构 → 生成 SQL → 执行 → 总结），Agent 模式收益最大，且相对独立（失败不影响其他链路）。

**范围**：
- 把 `chain.py:_handle_database_query` 用 LangGraph 重写
- 其他三条链路（chitchat / general / rag）保持原样
- 前端协议保持不变（`[SOURCES]...[/SOURCES]`），LangGraph 流式输出要做适配

**LangGraph 图设计草案**：

```
[database_intent] 
      ↓
[fetch_schema] ──(缓存表结构)──→ [generate_sql] 
                                          ↓
                                  [validate_sql] ──(不安全)──→ [refuse]
                                          ↓ (安全)
                                  [execute_sql]
                                          ↓
                                  [summarize_result] (流式)
```

**关键决策点**：
- SQL 安全校验节点用现有 `database.py:validate_sql`（成熟逻辑，复用）
- `schema_text` 缓存（当前每次查询都查库，浪费）
- 失败时降级返回明确的错误信息（不能让 Agent 无限循环）

**验收标准**：
- ✅ database 路由功能与升级前一致或更好
- ✅ SQL 注入防护依然有效（白名单 + 黑名单）
- ✅ 延迟不超过升级前的 1.5 倍
- ✅ 至少支持一个"多步组合"场景（如"先查表再筛选"）

### 第三阶段（1-2 月）：评估是否全链路 Agent 化

**目标**：基于第二阶段实测数据，决定 Agent 化范围。

**评估维度**：
- Agent 在第二阶段的实际效果（延迟、成本、准确率）
- 简单查询用 Agent 是否反而更慢更贵
- 是否需要支持"多步组合"复杂查询（如果用户场景里没有，可能不值得全链路）

**决策树**：

```
第二阶段效果如何？
├─ 明显提升 + 可控 → 扩展到其他链路，全链路 Agent 化
├─ 提升有限 + 成本上升 → 保持混合架构（database 用 Agent，其他保留路由）
└─ 出现稳定性问题 → 回滚第二阶段，重新评估技术选型
```

**保持混合架构是完全合理的终点**，不一定要走到"全 Agent"。很多生产级 RAG 系统就是"路由 + 部分 Agent"的混合形态。

---

## 六、分支管理策略

### 分支拓扑

```
main / master（理论主干）
  │
  ├── graduation-project（毕设交付版，冻结，不再提交）
  │
  └── agent-platform（新建，Agent 升级主开发分支）
        │
        ├── feat/semantic-chunker      （第一阶段）
        ├── feat/agent-text-to-sql     （第二阶段）
        └── feat/agent-full-rework     （第三阶段，视情况）
```

### 关键规则

1. **`graduation-project` 冻结**：这是已交付的毕设版本，不再提交任何改动。所有 bug 修复（即使是 P0）也走 `agent-platform` 或独立 hotfix 分支。
2. **`agent-platform` 作为集成分支**：各 feature 分支从这里拉出，完成后合并回来。
3. **feature 分支生命周期**：每个阶段对应一个 feature 分支，合并后可删除。
4. **可回滚**：每个阶段合并前，确保 `agent-platform` 在 `graduation-project` 基础上只增不减（或至少可一键回滚）。

### 建基命令参考

```bash
# 在 graduation-project 上确保工作区干净
git checkout graduation-project
git pull
git status  # 必须是 clean

# 创建 agent-platform
git checkout -b agent-platform
git push -u origin agent-platform

# 第一阶段 feature 分支
git checkout -b feat/semantic-chunker agent-platform
```

### 从 graduation-project 同步 bugfix

如果 `graduation-project` 后续有紧急 bugfix（理论上不该有，但万一），用 cherry-pick 同步：

```bash
git checkout agent-platform
git cherry-pick <commit-hash>
```

---

## 七、关键工程细节清单

### 7.1 流式输出适配

当前前端的 `[SOURCES]...[/SOURCES]` 协议不能直接对接 LangGraph 的流式。两个选择：

- **方案 A（保前端不动）**：在 LangGraph 输出和前端之间加一个适配层，把 LangGraph 的 token stream 重新封装成现有协议格式。优点：前端零改动。缺点：适配层本身有复杂度。
- **方案 B（前端配合改）**：前端改用 LangGraph 原生流式协议。优点：长期更干净。缺点：前端工作量大。

**建议第二阶段用方案 A**，第三阶段视情况评估方案 B。

### 7.2 记忆系统边界

明确两套记忆系统的职责：

| 记忆系统 | 管什么 | 存哪 |
|---------|--------|------|
| 现有 `ConversationMemory` | 跨会话持久化、会话列表、历史记录 | `data/sessions/*.json` |
| LangGraph checkpoint | Agent 单次执行内的中间状态 | 内存 / Redis / Postgres |

**边界规则**：LangGraph 执行结束后，把最终结果（user query + assistant answer）写回 `ConversationMemory`。LangGraph checkpoint 只在执行期间有效，不跨会话。

### 7.3 路由层保留策略

保留 `router.py` 的规则层作为 Agent 网关：

```python
# 伪代码：升级后的入口
def chat(query, session_id):
    # 第一层：零成本规则拦截（chitchat）
    if rule_based_route(query) == CHITCHAT:
        return chitchat_direct(query)  # 不进 Agent
    
    # 第二层：简单 RAG 也可直接走（不进 Agent）
    if is_simple_rag(query):
        return rag_direct(query)
    
    # 第三层：复杂意图进 Agent
    return agent_graph.invoke({"query": query, "session_id": session_id})
```

### 7.4 可观测性

Agent 调试比链路难一个量级，必须提前建设：

- **LangSmith**：LangGraph 原生支持，可视化 Agent 执行轨迹。开发期必装。
- **OpenTelemetry**：生产期用，把 LLM 调用、tool 调用、状态变迁都打成 trace。
- **指标埋点**：Agent 循环次数、tool 调用次数、平均延迟、失败率，这些是后续优化的依据。

### 7.5 配置兼容

`agent-platform` 的 `settings.py` 要向后兼容 `graduation-project` 的配置项。新增 Agent 相关配置（如 `agent_max_iterations`、`agent_model_name`），不要删改现有配置。

---

## 八、风险清单与应对

| # | 风险 | 严重度 | 应对 |
|---|------|--------|------|
| 1 | 一次性大重构把现有能跑的系统搞挂 | 🔴 高 | 渐进迁移（三阶段），每阶段可回滚 |
| 2 | LangChain 依赖冲突导致环境崩溃 | 🔴 高 | 独立 venv + pip-tools 锁版本 + 最小依赖 |
| 3 | Agent 调试困难，线上排障抓瞎 | 🔴 高 | 提前接 LangSmith + OpenTelemetry |
| 4 | 简单查询被 Agent 化后变慢变贵 | 🟡 中 | 保留规则路由作为前置网关 |
| 5 | 流式输出协议迁移导致前端不可用 | 🟡 中 | 第二阶段加适配层（方案 A） |
| 6 | 记忆系统职责混乱 | 🟡 中 | 明确边界（LangGraph 管执行内，memory 管跨会话） |
| 7 | 换切割器后向量库新旧切片共存 | 🟡 中 | 切换前清空 Milvus + 重新入库 |
| 8 | LangChain API breaking change | 🟢 低 | 锁版本，关注 release notes |

---

## 九、验收标准

### 第一阶段（语义切割）

- [ ] 切割不再出现"句子被切断"的切片
- [ ] 检索召回率（同 query 集合）不下降
- [ ] 现有四条路由链路全部正常
- [ ] 前端功能无变化
- [ ] 性能无回退（入库时间、查询延迟）

### 第二阶段（Text-to-SQL Agent 化）

- [ ] database 路由功能与升级前一致或更好
- [ ] SQL 注入防护依然有效
- [ ] 延迟不超过升级前 1.5 倍
- [ ] 至少支持一个多步组合场景
- [ ] LangSmith 能看到完整执行轨迹

### 第三阶段（全链路评估，视情况）

- [ ] Agent 平台在 P95 延迟、token 成本上可控
- [ ] 复杂多步查询场景有明确收益
- [ ] 简单查询路径（保留的路由）不受影响
- [ ] 有完整的 trace 和指标看板

---

## 附：一页纸总结

**方向**：另起 `agent-platform` 分支，渐进式 Agent 化。

**核心原则**：
1. `graduation-project` 冻结，不破坏已交付版本
2. 保留规则路由作为 Agent 前置网关（零成本拦截简单流量）
3. 渐进迁移：语义切割 → Text-to-SQL Agent → 评估全链路
4. 每阶段可回滚、可验证
5. 提前建设可观测性（LangSmith + OTel）

**关键认知**：
- `RecursiveCharacterTextSplitter` 不是语义切割，是结构感知字符切割，但已解决当前痛点
- Agent 不是银弹，简单场景路由永远更快更便宜
- 混合架构（路由 + 部分 Agent）是合理的终点，不一定要"全 Agent"

**最大风险**：不是技术选型，而是"一次性大重构把能跑的系统搞挂"。渐进迁移是唯一稳妥的路径。
