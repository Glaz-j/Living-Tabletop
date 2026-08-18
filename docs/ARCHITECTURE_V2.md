# Living Tabletop Architecture V2

## 目标

V2 将 LLM 放在“理解与表达”位置，把事实、检定、状态变化和披露权限留在可验证的系统层。自由输入与按钮拥有同一条执行流水线；玩家可以偏离剧本，但自由输入本身不等于偏离主线。

## 主流水线

```mermaid
flowchart LR
    A[IntentOption / Free text] --> B[PlayerIntentEnvelope]
    B --> C[ContextAssembler]
    C --> D[TurnPlanner LLM]
    D --> E[PlanValidator]
    E --> F[KnowledgeResolver]
    F --> G[DisclosurePolicy]
    G --> H[ValidatedActionPlan]
    H --> I[GameEngine / RuleEngine]
    I --> J[WorldKernel]
    J --> K[OutcomeEnvelope]
    K --> L[Deterministic Director]
    L --> M[GroundingValidator]
    M --> N[Narrator LLM]
    N --> O[NarrativeSequence / UI]
```

只有 `TurnPlanner` 与 `Narrator` 是主路径上的生成式角色。Director 是确定性策略；KnowledgeResolver、DisclosurePolicy、PlanValidator 与 GroundingValidator 都不生成世界事实。

## 核心决策

### 1. 输入统一

按钮与自由文本先转换为 `PlayerIntentEnvelope`：

- `source`: `option` 或 `free_text`
- `text`: 玩家实际表达；按钮可使用其第一人称台词或标签
- `intent_seed`: 按钮携带的动作 ID、动作类型和硬约束；自由输入为空
- `actor_id` 与 `scene_id`: 输入发生时的确定性世界坐标

按钮种子可以让系统离线执行，但不会绕过统一的验证、追踪、结算和叙事边界。

### 2. Planner 不写结果

`TurnPlanner` 只解析：

- 动作类型、目标、目的地、时长和风险
- 是否存在真实不确定性，是否需要检定
- 对话行为、说话对象、指代与 `KnowledgeQuery`

Planner 输出契约中没有成功文本、失败文本、效果列表或事实值。任何旧式 `success_text` / `failure_text` 即使由兼容输入提供，也会在验证前被丢弃。

### 3. 知识以结构化图为主

`KnowledgeResolver` 只检索 `WorldState.facts` 与 `npc_knowledge`。它不会把 Narrator 的 prose 当作新事实，也不会因为语义相似而让 NPC 知道未写入其知识表的内容。

`KnowledgeRetriever` 是稳定接口；V2 首版使用结构化过滤和轻量词法排序。以后可增加 semantic RAG，但 RAG 只能召回候选，不能直接授权披露。

### 4. 披露是规则决策

`DisclosurePolicy` 将候选分为：

- `automatic`: NPC 知道且不隐瞒，直接披露
- `check`: NPC 隐瞒或存在真实阻力，经过规则检定
- `refuse`: 世界规则明确禁止
- `unknown`: 没有可支持的知识

只有 WorldKernel 提交的 `fact_disclosed` / `player_learned_fact` 事件能增加玩家知识。事件必须携带 `fact_id` 和来源 NPC。

### 5. Director 只制造机会

Director 可以改变可见机会、时间压力、威胁和可供选择的 affordance；不能把“可调查的便笺”直接升级为“玩家已经得出结论”。

`open__*` 不是偏离主线的证据。只有进入标记为 `off_main` 的场景并继续在该支线活动，才会影响偏离遥测。

### 6. Narrator 只消费本轮结果

WorldKernel 结算后生成 `OutcomeEnvelope`，其中只包含：

- 本轮动作与机械结果
- 本轮新增或获准重述的事实
- 本轮可见事件
- 当前可见场景与在场实体
- Director 本轮制造的机会（若有）

Narrator 不再接收“玩家全部已知事实”。GroundingValidator 会在生成文本进入 UI 前检查未批准事实值、未在场角色和跨话题内容；失败时保留确定性演出，不污染世界状态。

### 7. 每轮可追踪

`TurnTrace` 保存输入、上下文摘要、Planner 输出、验证结果、知识查询、证据候选、披露决策、Kernel 事件、状态差异、OutcomeEnvelope 和 grounding 结果。开发者视图可读，玩家视图不可见。

## 兼容策略

- `GameEngine`、`RuleEngine`、`WorldKernel`、`ScenarioDefinition`、SQLite 快照和 EventLog 继续保留。
- 旧回放中的 `OpenActionPlan.success_text` / `failure_text` 仍可读取；新 Planner 不产生这些字段。
- 新字段全部有默认值，旧存档可以直接加载。
- `Keeper` 暂保留为旧 API 适配器；主执行链改用 `TurnPlanner`。
- authored action 继续由场景定义提供效果和演出，但新增事实仍必须对应 Kernel 的事实事件。

## 不变量

1. 玩家输入永远先被接受为意图，物理上不可能的结果可以失败，但不能把输入误路由成另一件事。
2. 自由输入不自动计为 off-main。
3. Director 机会不等于玩家结论。
4. NPC 披露必须能追溯到 `npc_knowledge`、`fact_id` 和来源 NPC。
5. Narrator prose 不能修改 `WorldState`。
6. Narrator 不能获得本轮 OutcomeEnvelope 之外的隐藏或无关事实。
7. 按钮和自由文本都产生 `PlayerIntentEnvelope` 与 `TurnTrace`。
