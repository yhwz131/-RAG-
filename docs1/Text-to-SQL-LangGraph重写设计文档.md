# Text-to-SQL LangGraph 重写设计文档

> **生成时间**：2026-06-23
> **目标分支**：`agent-platform`（新建，与 `graduation-project` 隔离演进）
> **升级阶段**：Agent 化第二阶段（在语义切割/Reranker/RRF 优化完成之后）
> **文档定位**：实现前的设计稿，含 State 设计、节点划分、条件路由、重试机制

---

## 目录

- [一、重写动机：现有实现的三个结构性问题](#一重写动机现有实现的三个结构性问题)
- [二、概念澄清：Reranker 不是 Embedding](#二概念澄清reranker-不是-embedding)
- [三、State 设计（图的核心）](#三state-设计图的核心)
- [四、节点划分](#四节点划分)
- [五、图结构与条件路由](#五图结构与条件路由)
- [六、关键节点的代码骨架](#六关键节点的代码骨架)
- [七、重试机制（现有实现做不到的核心能力）](#七重试机制现有实现做不到的核心能力)
- [八、流式输出对接方案](#八流式输出对接方案)
- [九、与现有路由的衔接](#九与现有路由的衔接)
- [十、收益与风险对比](#十收益与风险对比)
- [附：检索链路三个同步优化点](#附检索链路三个同步优化点)

---

## 一、重写动机：现有实现的三个结构性问题

当前 `chain.py:_handle_database_query` 是一个 100+ 行的线性函数，把 7 步逻辑揉在一起：

```python
def _handle_database_query(self, query, session_id, stream=False):
    db_source = self._get_db_source()            # 1. 获取连接
    schema_text = db_source.get_schema_for_llm()  # 2. 取表结构（每次都查库）
    raw_sql = self._call_llm(sql_messages)        # 3. 生成 SQL
    # 清理 markdown / 检查 ERROR                  # 4. 后处理
    result = db_source.execute_sql(raw_sql)       # 5. 执行
    result_text = self._format_sql_result(result) # 6. 格式化
    answer = self._call_llm(answer_messages)      # 7. 总结
```

### 结构性问题 1：没有错误恢复

SQL 执行失败就直接抛异常返回错误，不会"换个写法重新生成 SQL"。用户遇到 SQL 执行报错时，必须自己改问法重试。

### 结构性问题 2：schema 每次都查库

`get_schema_for_llm()` 每次查询都连数据库取一遍 `SHOW TABLES` + `DESCRIBE`。表结构不会频繁变化，这是纯粹的浪费。

### 结构性问题 3：不可观测

7 步揉在一个函数里，哪一步慢、哪一步出错、为什么生成出错的 SQL——全靠日志猜，没有结构化的执行轨迹。

### LangGraph 恰好能解决这三点

- 图模型把步骤变成显式节点，每个节点独立可测
- 条件边支持"失败 → 重试"的循环
- 节点级状态可被 LangSmith 可视化追踪

---

## 二、概念澄清：Reranker 不是 Embedding

> 本节澄清一个常见概念混淆。检索链路有三个同步优化点（详见附录），其中"引入 bge-reranker-v2-m3"常被误解为"换向量模型"。

**必须区分两类模型**：

| 模型类型 | 作用 | 阶段 | 项目当前 |
|---------|------|------|---------|
| **Embedding（向量模型）** | 文本 → 向量，用于召回 | 召回阶段（全库筛 top-N） | `bge-large-zh-v1.5` |
| **Reranker（重排模型）** | 对 (query, doc) 对打分精排 | 精排阶段（top-N → top-K） | 无 |

**`bge-reranker-v2-m3` 是 Reranker（cross-encoder），不是 Embedding。** 它的输入是 (query, doc) 对，输出相关性分数，**不能用来生成向量入库**。

正确理解：

```
当前：  bge-large-zh (embedding) → 向量检索 + BM25 → RRF 融合 → 返回
优化后：bge-large-zh (embedding) → 向量检索 + BM25 → RRF 融合 → bge-reranker 精排 → 返回
                                                              ↑ 这才是 reranker 的位置
```

**Embedding 模型继续用 bge-large-zh 不动，Reranker 是叠加在检索末端的精排层。**

引入 Reranker 后的附带好处：RRF 只负责粗排（尽量召回全），最终顺序由 Reranker 决定，**RRF 的权重不用再精调**——这正好解决了"BM25 在 RRF 中占比难定"的纠结。

---

## 三、State 设计（图的核心）

LangGraph 的灵魂是 State——所有节点共享、读写的数据结构。设计好 State，图就成了一半。

```python
from typing import TypedDict, Optional, List, Dict

class TextToSQLState(TypedDict):
    # ===== 输入（整个流程不变） =====
    query: str                    # 用户自然语言问题
    session_id: str
    history: List[Dict]           # 对话历史（从 ConversationMemory 取）

    # ===== 中间产物（各节点读写） =====
    db_type: str                  # mysql / postgresql
    db_name: str
    schema_text: str              # 表结构描述（缓存）
    schema_cached: bool           # schema 是否命中缓存
    raw_sql: str                  # LLM 生成的 SQL
    is_safe: bool                 # SQL 安全校验结果
    validate_error: str           # 校验/执行失败原因（供重试参考）
    query_result: Optional[Dict]  # 执行结果 {columns, rows, row_count}
    retry_count: int              # 重试次数（循环防护）

    # ===== 输出 =====
    final_answer: str             # 最终自然语言回答
    error: Optional[str]          # 错误信息（有值则流程异常终止）
    sources: List[Dict]           # 前端展示用的引用信息
```

### 设计要点

1. **`retry_count` 是循环防护的关键**——防止"生成失败 → 重试 → 又失败"死循环
2. **`schema_cached` 让 schema 节点能判断是否跳过查库**
3. **`error` 是全局错误通道**——任何节点都能写入并终止流程
4. **`validate_error` 同时承载校验失败和执行失败**——重试时把错误信息喂回 LLM

---

## 四、节点划分

把现有 7 步拆成 6 个业务节点 + 2 个终态节点：

| 节点 | 职责 | 输入字段 | 输出字段 |
|------|------|---------|---------|
| `fetch_schema` | 获取表结构（带 TTL 缓存） | db_type, db_name | schema_text, schema_cached |
| `generate_sql` | LLM 生成 SQL（支持重试） | query, schema_text, retry_count | raw_sql, retry_count+1 |
| `validate_sql` | SQL 安全校验（复用现有逻辑） | raw_sql | is_safe, validate_error |
| `execute_sql` | 执行 SQL | raw_sql | query_result, validate_error |
| `summarize` | LLM 总结结果（流式） | query, raw_sql, query_result, history | final_answer, sources |
| `refuse` | 拒绝场景（LLM 主动拒绝/校验不安全） | query, validate_error | final_answer |
| `error_handler` | 异常兜底 | error | final_answer |
| `format_output` | 统一输出格式化 | final_answer, sources | （终态） |

---

## 五、图结构与条件路由

### 流程图

```
START
  │
  ▼
[fetch_schema] ────────────────────────────────┐
  │                                             │
  ▼                                             │
[generate_sql] ◄──────────────────────────┐     │
  │                                       │     │
  ▼                                       │     │
[route_after_generate] (条件路由)          │     │
  │                                       │     │
  ├─ LLM 拒绝(ERROR) ──► [refuse]         │     │
  │                                       │     │
  ├─ 正常 ──► [validate_sql]              │     │
  │              │                        │     │
  │              ▼                        │     │
  │      [route_after_validate]           │     │
  │              │                        │     │
  │              ├─ 不安全 ──► [refuse]    │     │
  │              │                        │     │
  │              ├─ 安全 ──► [execute_sql] │     │
  │              │              │          │     │
  │              │              ▼          │     │
  │              │      [route_after_execute]    │
  │              │              │          │     │
  │              │              ├─ 失败且  │     │
  │              │              │  retry<2 ┘     │
  │              │              │  (回到 generate)
  │              │              │                │
  │              │              ├─ 失败且        │
  │              │              │  retry>=2 ─►[error_handler]
  │              │              │                │
  │              │              └─ 成功 ──► [summarize]
  │              │                          │    │
  │              ▼                          ▼    │
  ▼                                     [format_output]
[refuse] ─────────────────────────────────────►│
                                               │
                                               ▼
                                             END
```

### 三处条件路由

```python
def route_after_generate(state: TextToSQLState) -> str:
    """生成 SQL 后：LLM 拒绝 or 继续校验"""
    if state["raw_sql"].upper().startswith("ERROR"):
        return "refuse"
    return "validate"

def route_after_validate(state: TextToSQLState) -> str:
    """校验后：不安全拒绝 or 执行"""
    if not state["is_safe"]:
        return "refuse"
    return "execute"

def route_after_execute(state: TextToSQLState) -> str:
    """执行后：成功总结 / 失败重试 / 失败兜底"""
    if state["query_result"] is not None:
        return "summarize"
    if state["retry_count"] < 2:
        return "retry_generate"  # 回到 generate_sql
    return "error_handler"
```

---

## 六、关键节点的代码骨架

### 节点 1：fetch_schema（带缓存）

解决"每次查库"的浪费：

```python
import time

# 模块级缓存（进程内）
_schema_cache: Dict[str, tuple] = {}  # db_key -> (schema_text, timestamp)
_SCHEMA_TTL = 300  # 5 分钟

def fetch_schema(state: TextToSQLState) -> dict:
    db_key = f"{state['db_type']}:{state['db_name']}"

    # 命中缓存
    if db_key in _schema_cache:
        cached_text, ts = _schema_cache[db_key]
        if time.time() - ts < _SCHEMA_TTL:
            return {"schema_text": cached_text, "schema_cached": True}

    # 未命中，查库
    db_source = create_database_source()
    if not db_source:
        return {"error": "未配置数据库连接"}
    try:
        schema_text = db_source.get_schema_for_llm()
        _schema_cache[db_key] = (schema_text, time.time())
        return {"schema_text": schema_text, "schema_cached": False}
    except Exception as e:
        return {"error": f"获取表结构失败: {e}"}
    finally:
        db_source.close()
```

### 节点 2：generate_sql（LLM 生成）

直接复用现有的 `SQL_GENERATION_PROMPT`：

```python
def generate_sql(state: TextToSQLState) -> dict:
    prompt = SQL_GENERATION_PROMPT.format(
        db_type=state["db_type"],
        db_name=state["db_name"],
        schema_info=state["schema_text"],
        query=state["query"],
    )

    # 重试时把上次错误喂给 LLM
    if state.get("retry_count", 0) > 0 and state.get("validate_error"):
        prompt += (
            f"\n\n注意：上次生成的 SQL 执行失败，"
            f"错误信息：{state['validate_error']}。请修正后重新生成。"
        )

    messages = [
        {"role": "system", "content": "你是 SQL 专家，只输出可执行的 SQL 语句，不要输出任何解释。"},
        {"role": "user", "content": prompt},
    ]
    raw_sql = call_llm(messages).strip()

    # 清理 markdown
    if raw_sql.startswith("```"):
        raw_sql = raw_sql.split("\n", 1)[-1]
    if raw_sql.endswith("```"):
        raw_sql = raw_sql.rsplit("```", 1)[0]
    raw_sql = raw_sql.strip().rstrip(";")

    return {
        "raw_sql": raw_sql,
        "retry_count": state.get("retry_count", 0) + 1,
    }
```

### 节点 3：validate_sql（复用现有安全校验）

**重写时最该保留的成熟逻辑**——直接 import 复用，不要在 LangGraph 里重写：

```python
from api.pipeline.engines.database import validate_sql

def validate_sql_node(state: TextToSQLState) -> dict:
    is_safe, err = validate_sql(state["raw_sql"])
    return {"is_safe": is_safe, "validate_error": err}
```

`validate_sql` 是经过设计的白名单（只允许 SELECT/WITH）+ 黑名单（禁 INSERT/DROP 等），重写容易漏掉边界。**复用，不重写。**

### 节点 4：execute_sql

```python
def execute_sql(state: TextToSQLState) -> dict:
    db_source = create_database_source()
    try:
        result = db_source.execute_sql(state["raw_sql"])
        return {"query_result": result, "validate_error": ""}
    except Exception as e:
        # 执行失败，记录原因（供重试时让 LLM 看到错误信息）
        return {"query_result": None, "validate_error": str(e)}
    finally:
        db_source.close()
```

### 节点 5：summarize（流式）

```python
def summarize(state: TextToSQLState) -> dict:
    result_text = format_sql_result(state["query_result"])
    prompt = SQL_RESULT_PROMPT.format(
        query=state["query"],
        sql=state["raw_sql"],
        result=result_text,
    )
    messages = [
        *state["history"],
        {"role": "user", "content": prompt},
    ]
    answer = call_llm(messages, stream=False)
    return {
        "final_answer": answer,
        "sources": [{
            "source": f"SQL: {state['raw_sql']}",
            "content_snippet": result_text[:300],
            "score": 1.0,
        }],
    }
```

### 组装图

```python
from langgraph.graph import StateGraph, END

builder = StateGraph(TextToSQLState)

# 添加节点
builder.add_node("fetch_schema", fetch_schema)
builder.add_node("generate_sql", generate_sql)
builder.add_node("validate_sql", validate_sql_node)
builder.add_node("execute_sql", execute_sql)
builder.add_node("summarize", summarize)
builder.add_node("refuse", refuse_node)
builder.add_node("error_handler", error_handler_node)
builder.add_node("format_output", format_output_node)

# 入口
builder.set_entry_point("fetch_schema")

# 线性边
builder.add_edge("fetch_schema", "generate_sql")
builder.add_edge("summarize", "format_output")
builder.add_edge("refuse", "format_output")
builder.add_edge("error_handler", "format_output")
builder.add_edge("format_output", END)

# 条件边
builder.add_conditional_edges(
    "generate_sql",
    route_after_generate,
    {"refuse": "refuse", "validate": "validate_sql"},
)
builder.add_conditional_edges(
    "validate_sql",
    route_after_validate,
    {"refuse": "refuse", "execute": "execute_sql"},
)
builder.add_conditional_edges(
    "execute_sql",
    route_after_execute,
    {
        "summarize": "summarize",
        "retry_generate": "generate_sql",
        "error_handler": "error_handler",
    },
)

graph = builder.compile()
```

---

## 七、重试机制（现有实现做不到的核心能力）

### 核心思路

`retry_generate` 回到 `generate_sql` 时，**必须把上次的错误信息喂给 LLM**，否则它会犯同样的错：

```python
# generate_sql 节点里（见上方节点 2）
if state.get("retry_count", 0) > 0 and state.get("validate_error"):
    prompt += f"\n\n注意：上次生成的 SQL 执行失败，错误信息：{state['validate_error']}。请修正后重新生成。"
```

### 与现有实现的对比

| 场景 | 现有线性实现 | LangGraph 图实现 |
|------|------------|----------------|
| SQL 执行报错 | 直接返回错误，用户自己换问法 | 自动重试，把错误信息喂回 LLM 重新生成 |
| LLM 输出格式异常 | 整个流程挂掉 | 可在 route_after_generate 检测，走 error_handler |
| DB 连接抖动 | 抛异常 | execute_sql 捕获，走重试 |

### 循环防护

`retry_count` 是必须的——没有它，"生成失败 → 重试 → 又失败"会死循环。`route_after_execute` 里 `retry_count < 2` 的硬上限保证最多重试 2 次。

---

## 八、流式输出对接方案

这是迁移最麻烦的部分。

### 现状

前端协议是 `[SOURCES]...[/SOURCES]\n\n` + SSE 流式（见 `chain.py:_chat_stream_database`）。LangGraph 的流式是按节点（`stream_mode="updates"`）或按 token 两套机制，**和现有前端协议不是一回事**。

### 务实方案：只在 summarize 节点流式

其他节点同步执行，只在最后的 `summarize` 节点做 token 级流式。用一个适配函数把 LangGraph 输出包装成现有协议：

```python
def run_text_to_sql_stream(query, session_id):
    # 1. 同步执行到 summarize 之前（非流式）
    # 2. summarize 节点用流式 LLM 调用
    # 3. 先 yield [SOURCES] 元信息
    # 4. 再 yield summarize 的 token stream
    for event in graph.stream(
        {"query": query, "session_id": session_id, ...},
        stream_mode="updates",
    ):
        # 检测到 summarize 节点时，切换到 token 级流式
        ...
```

### 备选：图跑完后单独流式

如果 token 级流式对接太复杂，可以在 LangGraph 外面包一层：图同步跑完拿到 `query_result` 后，把 `summarize` 拎出来单独流式调用。这样前端协议完全不动。

**建议**：第二阶段先用备选方案（图跑完后单独流式），降低风险。token 级流式留到第三阶段优化。

---

## 九、与现有路由的衔接

不要让 LangGraph 接管所有查询。保留 `router.py` 的 database 检测作为入口：

```python
def chat(query, session_id):
    query_type = route_query(query)

    if query_type == QueryType.DATABASE:
        # database 类型走 LangGraph
        return text_to_sql_graph.invoke({
            "query": query,
            "session_id": session_id,
            "db_type": settings.db_type,
            "db_name": settings.db_name,
        })

    # 其他类型（chitchat/general/rag）走原链路
    return original_chat(query, session_id)
```

**路由规则层保留，LangGraph 只接管 database 路径。** 这样回滚成本最低——如果 Agent 化效果不好，注释掉 LangGraph 调用即可回到原链路。

---

## 十、收益与风险对比

### 收益

| 维度 | 现有线性实现 | LangGraph 图实现 |
|------|------------|----------------|
| 错误恢复 | 失败即终止 | 自动重试 2 次，带错误反馈 |
| schema 缓存 | 每次查库 | TTL 缓存，5 分钟内复用 |
| 可观测性 | 全靠日志猜 | LangSmith 可视化每个节点 |
| 可测试性 | 整个函数一起测 | 每个节点独立单测 |
| 流程可视化 | 读代码才能看懂 | 图结构一目了然 |

### 风险与代价

| 风险 | 严重度 | 应对 |
|------|--------|------|
| LangChain 依赖冲突 | 🔴 高 | 独立 venv + pip-tools 锁版本 |
| 流式输出对接复杂 | 🟡 中 | 第二阶段先用备选方案（图跑完后单独流式） |
| 简单查询延迟无改善 | 🟡 中 | 图的优势在边界情况，简单查询延迟与线性相当 |
| 调试难度跳一个量级 | 🟡 中 | 提前接 LangSmith |

### 诚实的 ROI 评估

对于"单次成功"的简单查询（可能占 80-90% 的流量），线性实现和图实现的延迟差不多——图的优势在边界情况（失败、重试、多分支）。

**所以如果 Text-to-SQL 使用频率低，重写的 ROI 也低**，优先级可以让位给前面三个检索优化（切片/Reranker/RRF）。建议在第一阶段的检索质量优化完成、确认 Text-to-SQL 是长期保留功能后，再启动本重写。

---

## 附：检索链路三个同步优化点

> 这是与本重写文档同步进行的三个检索优化，定位与本文档互补。

### 优化 1：RecursiveCharacterTextSplitter + 中文分隔符

替换 `chunker.py`，重新全量入库。关键配置：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
)
```

**注意**：换切割器后必须清空 Milvus 重新入库，否则新旧切片共存。

### 优化 2：引入 bge-reranker-v2-m3（精排层）

叠加在 RRF 融合之后，对 top-15 精排到 top-5。Embedding 模型 bge-large-zh 不动。

### 优化 3：重新考虑 BM25 在 RRF 中的占比

引入 Reranker 后，RRF 退化为粗排，**权重不用再精调**——让 RRF 尽量召回全（宁可多召回让 Reranker 筛），Reranker 会修正排序错误。这正好解决了"BM25 在 RRF 中占比难定"的纠结。

---

## 一页纸总结

**重写什么**：`chain.py:_handle_database_query`（100+ 行线性函数）→ LangGraph 图（6 业务节点 + 2 终态节点）。

**核心收益**：
1. 失败自动重试（带错误反馈给 LLM）
2. schema TTL 缓存（不再每次查库）
3. 节点级可观测（LangSmith 可视化）

**核心保留**：
- `validate_sql` 安全校验直接复用，不重写
- 路由规则层保留作为入口网关
- 前端协议保持不变（备选方案）

**核心原则**：渐进迁移、可回滚、不破坏 `graduation-project`。
