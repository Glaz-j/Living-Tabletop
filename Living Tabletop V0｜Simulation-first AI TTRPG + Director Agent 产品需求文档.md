# Living Tabletop V0
## Simulation-first AI TTRPG + Director Agent 产品需求文档

**文档状态：** V0 Draft  
**项目代号：** Living Tabletop（暂名）  
**首发类型：** 单人、CoC 风格、悬疑 / 调查 / 恐怖短篇  
**核心技术重点：** Director Agent  
**核心工程原则：** 极简 World Kernel、结构化世界状态、事件驱动模拟、LLM 不直接修改世界事实

---

# 0. 一句话定义

Living Tabletop 是一个 **Simulation-first AI TTRPG Runtime**。

它不让大模型“记住并编造整个世界”，而是：

> **由轻量 World Kernel 维护真实世界，由 Rule Engine 决定行为结果，由 NPC/World Simulation 推动后台事件，由 Narrator 负责表达，由 Director Agent 根据玩家体验状态动态调节节奏、危险、线索、喘息与戏剧机会。**

核心理念：

> **The DM doesn't write the world.  
> The world runs itself, and the Director makes it worth playing.**

---

# 1. 为什么做这个项目

目前大多数 AI 跑团 / 酒馆式产品的基础循环是：

```text
玩家输入文本
↓
LLM 阅读上下文
↓
LLM 推测当前世界状态
↓
LLM 继续生成故事
↓
上下文越来越长
↓
遗忘 / 矛盾 / 时间错乱 / NPC 全知 / 剧情失控
```

典型问题包括：

- NPC 忘记已经发生的事情
- NPC 知道自己不应该知道的信息
- 地点、物品、角色状态前后矛盾
- 玩家离开一个地方后世界停止运行
- 时间流速完全由语言感觉决定
- “十分钟调查”和“睡八小时”没有本质区别
- 玩家连续成功导致游戏失去张力
- 玩家连续失败导致体验崩溃
- LLM 为了推动剧情随时制造 Deus Ex Machina
- AI GM 过度迎合玩家
- AI GM 过度保护玩家
- AI GM 对所有行动都回答“可以”
- 剧情节奏完全依赖一次次 prompt
- 长上下文越来越昂贵

Living Tabletop 的目标不是训练一个更聪明的 DM 模型。

而是：

> **通过更好的 Agent Harness 和 World Architecture，让普通强模型也能稳定主持一个长期一致、会自行运行、具有动态节奏的游戏世界。**

---

# 2. V0 产品目标

V0 不追求完整 TTRPG 平台。

V0 只需要证明四件事。

## G1｜世界可以独立于 LLM 稳定存在

游戏必须拥有唯一可信的 Canonical World State。

LLM：

- 可以阅读允许读取的世界信息
- 可以提出行动
- 可以建议世界变化

但不能：

- 自己声明世界事实已经改变
- 直接写数据库
- 私自改变角色状态

---

## G2｜时间是真实存在的游戏资源

世界拥有 Virtual Clock。

玩家每个行动消耗虚拟时间。

例如：

```text
快速查看桌面        2 min
仔细搜索房间        20 min
阅读病历            30 min
步行前往教堂        25 min
睡觉                8 h
```

时间推进时：

> 世界中的 Scheduled Events 和 NPC Plans 同时推进。

---

## G3｜玩家可以自由行动，而不是只能点菜单

游戏同时提供：

### Suggested Actions

例如：

- 🔍 搜索办公室
- 🗣️ 询问 Anna
- 🚪 检查地下室入口
- ↩️ 离开医院

以及：

### Free Action

```text
或者，做任何你想做的事……
```

Suggested Actions 是 affordance，而不是 action whitelist。

玩家可以输入：

> 我骗 Anna 说警察已经来了，观察她的反应。

系统应尝试理解并执行。

---

## G4｜Director Agent 能明显改变游戏体验

Director 不负责写故事。

Director 负责回答：

> **现在这个玩家的体验缺什么？**

例如：

```text
Tension      太低
Progress     太快
Danger       太低
Success      连续过多
```

Director 可以：

> 推进威胁、制造 complication。

反之：

```text
Danger       极高
Resources    极低
Failures     连续发生
Frustration  上升
```

Director 可以：

> 提供喘息空间、释放合理线索、延迟威胁、创造逃生机会。

V0 的核心 Demo 必须能够让观察者看见：

> **相同的世界，因为玩家状态不同，Director 做出了不同的干预。**

---

# 3. V0 非目标

以下内容明确不属于第一版。

## 不做完整 CoC 规则实现

V0 使用：

> **CoC-inspired lightweight d100 rules**

包括：

- 属性
- 技能
- d100 检定
- HP
- Stress / Sanity-like 数值
- Inventory
- 简单伤害
- 简单 opposed check

不复制、内置或依赖完整商业规则书文本。

---

## 不做多人游戏

V0：

> Single Player + AI World

Party、多人协作、多人 session 暂不考虑。

---

## 不做完整游戏引擎

没有：

- 物理
- 3D
- 碰撞
- animation
- pathfinding
- frame loop

世界是：

> **symbolic world**

---

## 不让几十个 NPC 持续调用 LLM

NPC 默认：

> event-driven / plan-driven

只有需要：

- 对话
- 重大决策
- 计划失败
- 新信息
- Director intervention

时才调用模型。

---

## 不做程序化世界生成

首个 Scenario 完全手工设计。

目的：

> 验证 Runtime 和 Director。

而不是证明 AI 会写模组。

---

## 不做长期玩家画像

V0 只维护：

> 当前 session 内的 player model。

不要求用户长期养成账号状态。

---

## 不做完全自动的“AI 写小说”

这是游戏，不是 interactive novel generator。

所有叙事都必须来源于：

> World State + resolved Action + observable information。

---

# 4. 核心设计原则

## P1｜World Truth 与 Narrative 分离

数据库里的世界事实：

> **Truth**

玩家看到的文字：

> **Projection**

Narrator 只能表达已经发生的事情。

---

## P2｜World Truth 与 NPC Knowledge 分离

例如：

```text
WORLD TRUTH

killer = Wilson
```

不意味着：

```text
所有 NPC 都知道 killer = Wilson
```

不同 NPC 可以：

```text
Anna:
不知道凶手
但知道 Wilson 昨晚去过地下室

Bob:
错误认为 Anna 是凶手

Cultist:
知道 Wilson 是凶手
但不会承认
```

---

## P3｜Director 不能成为作弊上帝

Director 可以影响：

> 世界中尚未确定的可能性。

Director 不能改变：

> 已经确定的事实。

---

## P4｜Simulation 优先于 Narrative

优先级：

```text
World State
    ↓
Rules
    ↓
Simulation
    ↓
Director
    ↓
Narration
```

而不是：

```text
Narrative
↓
反推世界
```

---

## P5｜Event-driven，而不是 Continuous Simulation

服务器不需要：

```text
tick()
tick()
tick()
tick()
```

世界只在：

- 玩家行动
- 时间推进
- Scheduled Event
- NPC replan
- Director intervention

出现时更新。

没有事件时：

> CPU 几乎不工作。

---

## P6｜所有副作用都必须结构化

LLM 不能：

> “Wilson 已经逃离医院。”

LLM 可以：

```text
move_entity(
  entity="wilson",
  destination="street"
)
```

Kernel：

```text
VALID
```

才真正发生。

---

# 5. V0 体验

## 5.1 首个 Scenario

建议首个 Demo：

# 《圣玛丽医院》

暂定设置：

```text
时间：
1927 年冬夜

地点：
一座逐渐废弃的私人医院

规模：
5–7 个核心地点
5–6 个 NPC
1 个核心秘密
2 个敌对目标
10–15 个关键线索
1 条隐藏威胁时间线
```

例如地点：

```text
医院大厅
办公室
病房区
档案室
地下室
院长住宅
附近教堂
```

不追求大世界。

追求：

> 一个足够密集，可以出现时间、信息差和 Director 行为的 sandbox。

---

# 6. 玩家主循环

```text
① Narrator 描述当前 Scene

↓

② 系统生成 3–4 个 Suggested Actions

↓

③ 玩家
   ├─ 点击 Suggested Action
   └─ 输入 Free Action

↓

④ Action Parser 结构化玩家意图

↓

⑤ Rule Engine 判断是否合法 / 是否需要检定

↓

⑥ 执行 Action

↓

⑦ 推进 Virtual Time

↓

⑧ World Kernel 执行期间触发的 Events

↓

⑨ NPC / Faction 必要时更新

↓

⑩ Director 读取 Experience Telemetry

↓

⑪ Director 判断是否干预

↓

⑫ World Kernel 应用合法 intervention

↓

⑬ Narrator 根据结果生成下一幕

↓

回到①
```

---

# 7. World Kernel

World Kernel 是 V0 最基础的确定性运行时。

它不聪明。

它只负责：

> **保持世界是真的。**

建议实现：

```text
Python
Pydantic
SQLite
heapq / simple Event Queue
```

---

# 8. Canonical World State

建议核心数据模型分成以下部分。

## 8.1 Entity

```text
Entity
├── id
├── type
├── name
├── location
├── attributes
├── tags
└── active
```

type：

```text
PLAYER
NPC
CREATURE
ITEM
LOCATION
FACTION
OBJECT
```

---

# 9. Fact

用于表达世界事实。

例如：

```text
door_01.locked = true

painting_03.contains_hidden_key = true

wilson.is_killer = true
```

Fact 可以拥有：

```text
fact_id
subject
predicate
object/value
visibility
created_at
source
immutable
```

重要事实可以标记：

```text
immutable = true
```

Director 永远无法修改。

---

# 10. Relationship

用于：

```text
Anna trusts Player

Wilson fears Cult

Cult hates Police
```

表示：

```text
subject
relation
object
value
```

例如：

```text
trust = 0.7
fear = 0.2
hostility = 0.4
```

---

# 11. Virtual Clock

世界维护唯一：

```text
world_time
```

例如：

```text
1927-11-18 21:34
```

玩家行动：

```text
search_room()
```

Rule Engine：

```text
duration = 20 min
```

执行：

```text
advance_time(+20m)
```

---

# 12. Event Queue

每个未来事件：

```text
ScheduledEvent

id
time
type
actor
target
payload
condition
priority
cancelable
```

例如：

```text
21:40
cultist_leave_church

21:55
police_arrive

22:10
ritual_stage_2
```

Event Queue 使用：

```text
priority queue
```

即可。

V0 不需要复杂 simulation framework。

---

# 13. Event Log

所有已发生事件 append-only。

例如：

```text
21:20 player_entered_hospital

21:24 anna_saw_player

21:31 player_lied_to_anna

21:40 cultist_left_church

21:54 player_found_medical_record
```

原则：

> Event 不修改。

必要时通过新的 Event 修正状态。

Event Log 用于：

- debug
- save/load
- NPC memory
- Director telemetry
- narrative recall
- replay
- automated testing

---

# 14. NPC Knowledge Model

每个 NPC 拥有独立 Knowledge State。

最简单形式：

```text
KnowledgeEntry

npc_id
fact
confidence
source
timestamp
```

例如：

```text
Anna

Wilson entered basement
confidence 0.9
source eyewitness
```

NPC 允许：

- 不知道事实
- 知道事实
- 错误相信事实
- 怀疑事实
- 故意隐藏事实

---

# 15. NPC Memory

Knowledge 与 Memory 分离。

Knowledge：

> NPC 认为世界是什么样。

Memory：

> NPC 记得发生过什么。

例如：

```text
Player threatened me.

Player saved Anna.

I heard an explosion last night.
```

Memory 可以使用：

```text
Embedding retrieval
```

但 V0 不要求复杂 memory architecture。

---

# 16. NPC Goal 与 Plan

NPC V0 不持续 reasoning。

使用：

```text
goal
current_plan
scheduled_actions
```

例如：

```text
Wilson

Goal:
destroy evidence

Plan:
21:50 burn medical records
22:10 leave hospital
```

这些行为直接进入 Event Queue。

只有发生：

```text
player interrupts
critical information received
plan becomes impossible
Director intervention
```

才触发：

```text
NPC.replan()
```

---

# 17. Faction Simulation

V0 可以只实现极轻量 faction。

例如：

```text
Cult

goal:
complete ritual

progress:
43%

resources:
3 cultists

alert:
52%
```

Faction 不需要复杂 Agent。

它可以通过：

```text
scheduled event
+
simple rules
```

推进。

---

# 18. Rule Engine

Rule Engine 是确定性层。

负责：

```text
action validity
skill check
dice
damage
stress
inventory
movement
duration
```

---

# 19. Dice

使用可记录 PRNG。

每次 roll 保存：

```text
roll_id
dice
result
modifier
reason
timestamp
```

开发模式可以使用固定 seed：

> 完全 replay 某局游戏。

正式游戏允许随机 seed。

---

# 20. Action Parser

Free Text：

> “我假装警察马上会来，骗 Anna 说院长已经被捕，然后观察她。”

LLM 负责转成：

```text
ActionIntent

type:
DECEIVE

actor:
player

target:
anna

content:
警察即将到达，院长已经被捕

goal:
observe reaction

possible_skill:
persuasion/deception

estimated_duration:
3 min
```

Action Parser：

> **只解释意图。**

它无权决定成功。

---

# 21. Open Action（当前实现决策）

如果输入：

> “我处理一下这个房间。”

系统将它视为一个有效的开放意图，并交给 KP 仲裁器。

应该：

> 在不替玩家宣告成功的前提下，推断最保守、最贴近原句的尝试；必要时进行 CoC 检定。

预设动作不是玩家意图白名单，只是快捷方式。玩家可以离开现场、回家休息、前往模组外地点或永久偏离主线。物理上无法完成的声明仍会被接纳为尝试，但不会改写硬事实；Director 只能用世界内事件柔性提示未解决线索，不能强迫玩家回主线。

---

# 22. Suggested Actions

每轮提供约 3–4 个 Suggested Actions。

建议保证不同 action style：

```text
🔍 Investigate

🗣 Social

⚠ Risk / Advance

↩ Leave / Reposition
```

不是每轮强制四类齐全。

具体根据场景产生。

---

# 23. Suggested Action 的硬约束

Suggested Actions：

### 必须

来源于玩家已经可观察到的世界。

### 不允许

泄漏 Hidden Knowledge。

错误：

> 搜查凶手 Wilson 的包。

正确：

> 搜查 Wilson 留在桌边的包。

---

# 24. Narrator Agent

Narrator 只负责：

> **如何表达已经发生的事情。**

Narrator 输入：

```text
visible world state
resolved action
dice result
new events
visible NPC responses
scene tone
Director narrative guidance
```

Narrator 不应该获得：

> 完整 hidden world truth。

至少在 prompt 中严格限制。

---

# 25. Narrator 禁止行为

不能：

- 创建新 NPC
- 创建关键物品
- 修改 HP
- 修改位置
- 决定检定结果
- 修改 world time
- 揭露玩家尚不知道的信息
- 修改核心秘密
- 自己推进敌人计划

---

# 26. Director Agent

这是整个项目 V0 的重点。

Director 的定位不是：

> Dungeon Master。

而是：

# Experience Manager

它持续回答：

> 当前体验处于什么状态？

以及：

> 接下来应该增加什么、减少什么？

---

# 27. Director 观察的 Experience State

V0 可以维护如下指标：

```text
Tension
Danger
Progress
Mystery
ResourcePressure
SuccessStreak
FailureStreak
Agency
Frustration
ReliefNeed
Novelty
TimePressure
```

范围统一：

```text
0–100
```

---

# 28. Experience State 不全部由 LLM 猜

采用：

> deterministic telemetry + LLM interpretation

例如：

### Danger

可以根据：

```text
HP
enemy proximity
resource level
threat clock
escape routes
```

计算基础值。

### FailureStreak

直接统计。

### Progress

根据：

```text
关键 clue 已发现比例
location coverage
scenario milestone
```

计算。

### Tension

可以综合：

```text
danger
recent threat events
unknown threat
time pressure
scene history
```

---

# 29. Session-local Player Model

只分析最近行为。

例如：

```text
investigation_preference

social_preference

risk_tolerance

combat_preference

caution

exploration

novelty_seeking
```

例如玩家过去十个 action：

```text
SEARCH
SEARCH
ASK
SEARCH
AVOID
ASK
```

Director 可以推断：

> investigation-heavy / cautious

不需要用户维护 profile。

---

# 30. Director 的目标

Director 不是最大化：

```text
difficulty
```

而是优化：

# Experience Quality

核心原则：

```text
Challenge ≠ Punishment
Relief ≠ Rescue
Drama ≠ Railroading
Mystery ≠ Withholding Everything
```

---

# 31. Pacing Model

V0 建议采用简单阶段状态：

```text
EXPLORE

BUILD

PRESSURE

PEAK

RELIEF
```

大致循环：

```text
Explore
↓
Build
↓
Pressure
↓
Peak
↓
Relief
↓
重新开始
```

Director 可以改变阶段，但不能固定机械轮转。

---

# 32. Director Decision Cycle

Director **不必每轮调用**。

以下情况触发：

```text
N 个 meaningful actions 后

重大事件发生

场景切换

危险显著变化

连续成功

连续失败

长时间无进展

玩家接近死亡

玩家发现关键真相

Threat Clock 进入重要阶段
```

这样减少：

> LLM cost + latency。

---

# 33. Director Intervention API

Director 无权直接写世界。

它只能调用有限工具。

V0 建议：

## Threat

```text
advance_threat()
```

推进已有 Threat Clock。

---

## Complication

```text
introduce_complication()
```

从合法 complication pool 中选择。

例如：

- 灯突然熄灭
- NPC 离开
- 敌人改变计划
- 天气恶化
- 道路关闭

必须满足世界条件。

---

## Clue

```text
surface_clue()
```

让已有但未发现的 clue 获得新的合理暴露机会。

注意：

> 不能凭空创造核心证据。

---

## Respite

```text
offer_respite()
```

例如：

- 暂时安全空间
- 敌人行动延迟
- 安全 NPC 出现
- 获得补给机会

---

## Pressure

```text
increase_pressure()
```

例如：

- 时间窗口缩短
- NPC 开始行动
- 某资源损耗
- 敌人察觉玩家

---

## Spotlight

```text
spotlight_npc()
```

让某个已有 NPC 更主动参与。

---

## Encounter Adjustment

```text
adjust_encounter()
```

只允许修改：

> 尚未开始的 encounter 参数。

---

## Choice Guidance

```text
guide_affordances()
```

不修改世界。

只要求 Suggested Action Generator：

> 当前更突出调查 / 逃生 / 社交 / 风险选择。

---

# 34. Director 永久禁止的事情

Director 不可以：

### 修改已经公开的世界事实

### 修改核心秘密

### 修改骰子结果

### 复活死亡角色

### 删除玩家已经获得的线索

### 让 NPC 获得不可能知道的信息

### 直接把关键道具传送给玩家

### 因为玩家失败就自动让失败无效

### 因为玩家成功就取消成功

### 强制玩家选择某条剧情路径

### 无因果来源生成剧情转折

---

# 35. Director Intervention 的因果合法性

每次 intervention 必须提供：

```text
reason

world justification

affected entities

expected experience effect
```

例如：

```text
Action:
advance_threat

Reason:
player progress high,
tension low,
three consecutive successes

World justification:
cultists already know investigation is underway

Expected effect:
increase tension and time pressure
```

Kernel 再检查：

> 是否合法。

---

# 36. Director Fairness Principle

玩家不应该感觉：

> 游戏在针对我作弊。

所以：

> Director 操纵的是 opportunity，不是 outcome。

例如：

可以：

> 增加 encounter 发生概率。

不能：

> 强制玩家失败。

---

# 37. Director Relief Principle

玩家濒死时：

错误：

> 怪物突然滑倒死掉。

正确：

> 原本存在的逃生门现在成为明显机会。

或者：

> 敌人因另一个世界事件暂时改变目标。

---

# 38. Director 与 Scenario Authoring

Scenario 不只定义剧情。

还应该定义：

```text
Threats

Clues

Complications

Respite Opportunities

Escalation Hooks

NPC Secrets

Critical Facts
```

Director：

> 在这些合法 affordance 之间进行调度。

这比完全自由生成安全得多。

---

# 39. Threat Clock

V0 强烈建议加入。

例如：

```text
Ritual Progress

0 ───── 25 ───── 50 ───── 75 ───── 100
```

不同阶段触发世界变化。

玩家：

> 不行动。

世界：

> 仍然前进。

---

# 40. Mystery Progress

核心谜团可以由多个 clue 支撑。

例如：

```text
Murder Mystery

Clue A
Clue B
Clue C
Clue D
Clue E
```

不要求发现全部。

Scenario 可以定义：

```text
minimum evidence threshold
```

防止只有单一路径。

---

# 41. 防 Softlock

如果核心线索永久丢失：

Director 可以：

> 创建新的“发现机会”。

注意：

不是创建新的真相。

例如：

原始线索：

> 病历里记录某药物。

病历烧掉以后：

仍然可以通过：

> 护士记忆

获得同一个 fact。

这叫：

> Information Redundancy。

---

# 42. RAG

RAG 不是世界状态。

V0 RAG 只用于：

## Lore Retrieval

地点背景、人物资料、世界设定。

## Rule Retrieval

需要解释复杂规则时。

## NPC Memory Retrieval

从 NPC 历史事件中找相关记忆。

## Narrative Recall

需要自然引用过去事件时。

---

# 43. 数据权威层级

必须严格：

```text
Canonical World State
        >
Event Log
        >
NPC Knowledge
        >
Memory Retrieval
        >
Narrative Text
```

Narrative 永远不能覆盖 State。

---

# 44. Agent Tool Interface

所有 Agent 都通过工具访问系统。

例如：

```text
get_visible_scene()

get_entity()

get_npc_knowledge()

get_recent_events()

get_experience_state()

propose_action()

advance_threat()

surface_clue()

schedule_event()
```

Agent：

> 永远不直接访问 SQLite。

---

# 45. Agent 分工

V0 最少只需要三个 LLM role。

## Action Interpreter

负责：

> 玩家说了什么。

---

## Director

负责：

> 接下来体验需要什么。

---

## Narrator

负责：

> 发生的事情怎么讲。

---

NPC Agent：

> 按需调用。

并非常驻 Agent。

---

# 46. 前端

V0 不追求复杂视觉。

核心是：

> 让它“像游戏”，而不是“像 ChatGPT”。

---

# 47. 主界面

建议：

```text
┌─────────────────────────────┐
│                             │
│       Narrative Area        │
│                             │
│  地下室里传来滴水声……       │
│                             │
├─────────────────────────────┤
│ 🔍 检查血迹                 │
│ 🗣️ 询问 Anna                │
│ 🔦 继续深入                  │
│ ↩️ 离开                     │
├─────────────────────────────┤
│ 或者，做任何你想做的事……    │
└─────────────────────────────┘
```

---

# 48. 玩家可见状态

尽量少。

例如：

```text
HP
Stress
Inventory
Time
```

不显示：

```text
Tension
Director State
Threat Progress
NPC Hidden Knowledge
```

---

# 49. Developer Mode

项目 Demo 极其重要。

建议增加：

# Director Console

可以看到：

```text
Tension        74
Danger         52
Progress       63
Frustration    18

Phase:
PRESSURE

Director decision:

advance_threat()

Reason:
player has succeeded
four times consecutively
```

旁边：

# World Inspector

显示：

```text
World Clock

Event Queue

NPC Locations

Threat Clocks

Known Facts

Event Log
```

这将成为项目最好的技术 Demo。

---

# 50. Save / Load

V0 必须支持。

至少保存：

```text
World State Snapshot

Event Log

Event Queue

Player State

NPC Knowledge

Director State

RNG Seed
```

重新加载以后：

> 游戏必须完全继续。

---

# 51. 性能目标

World Kernel：

> 不应该成为性能瓶颈。

预期一个 Scenario：

```text
< 100 entities

< 1000 facts

< 500 scheduled events

< 10000 event logs
```

都属于极小规模。

运行时主要资源消耗来自：

> LLM。

---

# 52. LLM Cost 原则

不要：

> 每个 NPC 每一分钟 reasoning。

采用：

```text
event-driven invocation

structured state

small contextual retrieval

NPC sleep

Director conditional invocation
```

目标：

> 一轮玩家行动通常只需要 1–3 次 LLM 调用。

---

# 53. Fail Closed

如果 LLM 输出：

```text
非法 world mutation
```

Kernel：

> 拒绝。

如果 Director 输出：

```text
违反 immutable fact
```

Kernel：

> 拒绝。

如果 Action Parser 无法解析：

> clarification。

不要：

> 猜。

---

# 54. Observability

每次 Agent 调用记录：

```text
agent role
input state id
tool calls
structured output
validation
latency
token usage
result
```

尤其 Director：

必须能够回放：

> 为什么这个时候增加难度？

---

# 55. 自动化测试

World Kernel 必须大量 deterministic test。

例如：

### Test

锁门时：

```text
open_door
```

必须失败。

### Test

NPC 不在现场：

> 不得获得现场知识。

### Test

advance_time 跨过 Event：

> Event 必须执行。

### Test

reload：

> State 必须完全一致。

### Test

Director 修改 immutable fact：

> 必须拒绝。

---

# 56. Director 测试

可以用 scripted players。

例如：

# Player A

连续高速推进。

期望：

> Director 增加 pressure。

# Player B

连续失败。

期望：

> Director 不继续无脑提高危险。

# Player C

长期无进展。

期望：

> Director 增加 clue availability。

# Player D

HP 很低但继续冒险。

Director 可以：

> 保留危险。

不能强行保护。

---

# 57. Director 最重要的评估不是“聪明”

而是：

## Consistency

有没有破坏世界规则？

## Fairness

有没有明显作弊？

## Responsiveness

有没有根据玩家状态改变策略？

## Pacing

有没有持续制造压力/释放循环？

## Agency

玩家是否仍能自由选择？

## Explainability

开发模式能不能解释 intervention？

---

# 58. V0 成功标准

项目 V0 完成时，必须可以：

### 1

玩家从头到尾完成一个：

> 30–60 分钟单人悬疑 Scenario。

### 2

玩家可以：

> 全程只用自由文本完成游戏。

### 3

也可以：

> 大部分时间只点击 Suggested Actions。

### 4

世界拥有明确：

> Virtual Clock。

### 5

玩家不在场的 NPC / faction：

> 能通过 Event Queue 持续推进。

### 6

NPC：

> 不读取自己不知道的 World Truth。

### 7

游戏 reload 后：

> 世界一致。

### 8

Director 至少能够可靠处理：

```text
玩家过于顺利

玩家连续失败

玩家长期无进展

玩家濒临危险

游戏长时间低张力
```

### 9

Director：

> 不能修改 immutable facts。

### 10

Developer Mode：

> 可以看到 Director 决策原因。

---

# 59. 推荐开发顺序

## Phase A｜World Kernel Skeleton

实现：

```text
Entity
World State
Virtual Clock
Event Queue
Event Log
save/load
```

完全不用 LLM。

---

## Phase B｜Basic Play Loop

实现：

```text
Free Text
Action Parser
Rule Engine
World Mutation
Narrator
```

这时已经可以玩。

---

## Phase C｜NPC Knowledge

实现：

```text
World Truth
≠
NPC Knowledge
```

加入简单 NPC plan。

---

## Phase D｜Director V0

先只允许：

```text
advance_threat
surface_clue
offer_respite
increase_pressure
guide_affordances
```

五类 action。

---

## Phase E｜Director Telemetry

加入：

```text
tension
danger
progress
success/failure
frustration
player style
```

---

## Phase F｜Director Console

可视化：

> Director 到底在干什么。

这一阶段项目开始真正具有作品展示价值。

---

# 60. V0 最值得保护的差异化

这个项目未来很容易越做越像：

> 普通 AI D&D。

所以必须始终保护下面四个点。

### ① Simulation-first

世界不是 Prompt。

### ② Canonical State

事实由 Kernel 管。

### ③ Autonomous Time

世界不会因为玩家没看见就冻结。

### ④ Constrained Director

AI 不负责决定剧情。

AI 负责：

> **在不破坏世界规则和玩家 agency 的前提下管理体验。**

这四条就是项目 identity。

---

# 61. 后续可能扩展，但暂不进入 V0

未来可以考虑：

- 多人 Party
- 完整 CoC adapter
- D&D adapter
- Scenario SDK
- Scenario Marketplace
- 自动模组导入
- 自动 Scenario generation
- Voice
- 图片 / 地图
- Godot frontend
- 多 Agent faction simulation
- 长期 NPC memory
- 玩家历史画像
- AI 自动 playtest
- Director policy comparison
- 玩家自定义 Director 风格
- Horror / Mystery / Comedy 等 Director presets

但第一版一律不做。

---

# 62. 当前仍需要产品负责人确认的问题

以下问题暂时不应该由开发者自行决定。

## Q1｜V0 到底多像 CoC？

两种路线：

### A. 高度 CoC-like

属性、SAN、技能、d100、调查玩法都很像。

优点：

> 玩家马上懂。

缺点：

> 规则实现更复杂，也要更注意规则文本/IP边界。

### B. 只保留调查恐怖精神

自研非常轻的 d100 system。

我当前倾向：

> **B。**

---

## Q2｜玩家死亡应该多容易？

Director 是否应该明显保护角色？

我建议：

> **不保护 outcome，只保护 opportunity。**

角色真的可以死。

但濒死时：

> 世界应该更可能出现合理逃生机会。

是否同意？

---

## Q3｜Director 是否允许创造“小事实”？

例如：

> 玩家连续失败。

Director 想：

> “突然停电。”

如果 Scenario 原来没有写“停电事件”，是否允许它动态创建？

三种可能：

### Conservative

只能调用预写好的 intervention。

### Hybrid

关键事实必须预写；

环境性、小规模事件可以生成。

### Free

只要 validator 认为合理都可以生成。

我当前强烈倾向：

> **Hybrid。**

---

## Q4｜NPC 对话自由度多高？

NPC Agent 是否可以：

> 临场创造自己的个人细节？

例如：

> “我小时候住在伦敦。”

如果 Scenario 没定义。

建议：

> 允许低影响 personal flavor；
> 不允许生成会改变推理链的事实。

需要定义：

```text
hard canon
soft canon
flavor
```

---

## Q5｜Suggested Actions 是 3 个还是 4 个？

我偏向：

> 3 个主选项 + 永久 Free Action。

因为过多会像传统文字冒险菜单。

---

## Q6｜检定过程展示到什么程度？

例如：

```text
Spot Hidden: 42 / 65
Success
```

还是只叙事：

> 你发现墙角似乎有东西……

可以提供模式：

```text
Tabletop Mode
Cinematic Mode
```

但 V0 是否需要两个？

我倾向：

> 只做 Tabletop Mode，先保证系统透明。

---

## Q7｜Director 是否应该对玩家完全隐藏？

正式游戏：

> 隐藏。

Developer Mode：

> 完全显示。

我认为这是必需的。

---

## Q8｜第一版 Scenario 是否应固定结局？

可以：

### Branching Mystery

存在 3–5 个预设 major endings。

或者：

### Simulation Outcome

只定义世界规则，没有固定 endings。

我建议：

> **Hybrid。**

定义几个 world outcomes，但允许玩家产生意外结果。

---

## Q9｜首版需要地图吗？

我当前建议：

> 不做几何地图。

只做 Location Graph：

```text
Hospital Hall
     │
Office
     │
Basement
```

这样足以支持：

> movement + travel time。

---

## Q10｜Director 的“体验目标”应该是谁定义？

三条路线：

### Designer-defined

Scenario 作者规定目标 tension curve。

### Agent-defined

Director 自己决定。

### Hybrid

设计者提供：

```text
tone
pacing preferences
difficulty
```

Director 自主调度。

我建议：

> **Hybrid。**

---

# 63. 当前建议锁定的 V0 决策

如果没有特别反对，我建议第一版锁定：

```text
单人

CoC-inspired investigation horror

Python

SQLite

Pydantic

自研 World Kernel

virtual clock

priority event queue

event sourcing

structured NPC knowledge

NPC event-driven planning

3 suggested actions + free text

lightweight d100 rules

handcrafted scenario

Narrator Agent

Action Interpreter Agent

Director Agent

Director bounded tool space

session-local player model

Developer Director Console

不做多人

不做完整游戏引擎

不做程序化世界生成

不做实时 tick

不做几十 NPC 常驻 LLM
```

---

# 64. 项目最终希望证明的事情

不是：

> “LLM 可以玩 CoC。”

而是：

> **一个小型、确定性的 simulation runtime，加上受约束的 Director Agent，可以让 LLM TTRPG 获得比纯上下文角色扮演更可靠的世界一致性、更真实的时间流动以及更稳定的戏剧节奏。**

而用户看到的不是这些架构。

用户只应该感觉：

> **这个世界似乎真的在我看不到的地方继续活着，而且它很会讲故事。**
