# Visual World Kernel V1 设计文档

## 1. 项目背景

本项目希望构建一个类似 AI 酒馆 / TRPG / DND 交互体验的 Agent 世界。

与传统纯文本酒馆不同，本项目不将“世界”主要维护在 LLM 上下文中，而是维护一个独立、持续存在、结构化的 **World Runtime**。

LLM 主要负责：

- 理解玩家行为；
- 扮演角色；
- 生成自然语言；
- 进行导演决策；
- 提议世界行为。

真正的世界状态由 World Kernel 维护。

项目同时提供一个非常轻量的 RPGMaker 风格视觉层，使世界状态：

- 可观察；
- 可调试；
- 可展示；
- 可供玩家理解；
- 可被 Director Agent 使用。

因此，本项目中的地图不是传统游戏地图系统，而是：

> **结构化世界状态的一种视觉投影。**

---

# 2. 核心目标

V1 重点解决四个问题。

### 2.1 世界能够持续存在

NPC、地点、物品、时间、事件等状态不能依赖 LLM 当前上下文。

例如：

```text
Alice 当前位于 Church.Basement
Player 位于 Tavern
当前时间 Day 3 21:34
Ritual 将于 22:00 开始
```

即使 LLM 会话刷新，这些事实仍然存在。

---

### 2.2 世界状态能够可靠更新

玩家、NPC、Director、脚本都不能直接任意修改世界状态。

所有变化统一经过：

```text
Command
   ↓
Validation
   ↓
Event
   ↓
Reducer
   ↓
Runtime State
```

确保世界状态可验证、可追踪。

---

### 2.3 世界能够直接被视觉化

世界状态可以被自动渲染为类似 RPGMaker 的简单界面，例如：

```text
            [Old Church]
                Alice

                   │
                   │
[Tavern] ───── [Market]
 Player
                   │
                [Harbor]
```

视觉层主要承担：

- 空间理解；
- 状态观察；
- Debug；
- 玩家信息呈现。

---

### 2.4 同一个世界支持不同观察者

同一套 World State 可以生成不同 Projection：

```text
World State
     │
     ├── Dev View
     ├── Player View
     ├── NPC View
     └── Director View
```

因此不维护多套独立地图。

---

# 3. 非目标

V1 明确不做以下内容：

- 完整游戏引擎；
- 物理模拟；
- 实时碰撞；
- NavMesh；
- 高精度寻路；
- 60 FPS 后台世界模拟；
- 完整战斗系统；
- 完整经济系统；
- 生态模拟；
- GUI 世界编辑器；
- RPGMaker 级地图制作工具。

V1 核心目标是：

> **一个轻量、结构化、可观察、可更新的 Agent World Runtime。**

---

# 4. 总体架构

整体系统分为四层。

```text
┌────────────────────────────────────┐
│ ① World Definition                │
│                                    │
│ YAML / JSON / Markdown             │
│ Maps / Entities / Rules / Scenario │
└─────────────────┬──────────────────┘
                  │ load
                  ▼
┌────────────────────────────────────┐
│ ② World Runtime State             │
│                                    │
│ Clock                              │
│ Map State                          │
│ Entities                           │
│ Relations                          │
│ Knowledge                          │
│ Flags                              │
│ Event Queue                        │
│                                    │
│ 世界运行时唯一真相 SSOT              │
└─────────────────▲──────────────────┘
                  │
                  │ Reducer
┌─────────────────┴──────────────────┐
│ ③ World Kernel                    │
│                                    │
│ Command                            │
│ → Validation                       │
│ → Event                            │
│ → Reducer                          │
│ → State Update                     │
│                                    │
│ Event Queue / Event Log            │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ ④ Projection / Renderer           │
│                                    │
│ Dev / Player / NPC / Director      │
│ RPG Map / Timeline / Panels        │
└────────────────────────────────────┘
```

---

# 5. 核心设计原则

## 5.1 Runtime State 是唯一运行时真相

任何视觉界面都不是世界状态本身。

正确关系：

```text
World State
    ↓
Renderer
    ↓
Visual View
```

而不是：

```text
Visual View
    ↓
推断世界状态
```

例如：

```text
Alice.location = church
```

才是事实。

Alice 的 sprite 出现在教堂，只是这个事实的视觉表现。

---

## 5.2 地图和实体分离

地图描述：

> 世界空间是什么样。

Entity 描述：

> 世界中存在什么。

因此 Map 不维护：

```text
Tavern:
  NPCs:
    Alice
    Bob
```

Entity 自己维护：

```text
Alice.location = tavern
Bob.location = market
```

如果需要知道 Tavern 当前有哪些角色：

```text
SELECT entity
WHERE entity.location = tavern
```

得到即可。

---

## 5.3 Static Definition 与 Runtime State 分离

例如地图定义：

```yaml
id: church_basement_door

from: church
to: church.basement
```

表示：

> 世界中存在这条连接。

运行时状态：

```yaml
locked: true
destroyed: false
discovered_by_player: false
```

表示：

> 这条连接现在是什么状态。

即：

```text
Definition = 世界可能是什么
Runtime = 世界现在是什么
```

---

## 5.4 Visual 坐标不是逻辑坐标

地图可以保存：

```yaml
x: 12
y: 8
```

用于：

- UI 排版；
- sprite 显示；
- 地图布局。

但世界逻辑不应该依赖像素坐标。

Canonical location 应当是：

```text
church.basement
```

而不是：

```text
x = 421.7
y = 611.2
```

---

# 6. Authoring：结构化文档优先

V1 不开发 GUI 编辑器。

世界由 repo 中的结构化文件定义。

建议：

```text
world/
├── world.yaml
│
├── maps/
│   ├── grayhaven.yaml
│   ├── tavern.yaml
│   └── church.yaml
│
├── entities/
│   ├── alice.yaml
│   ├── bartender.yaml
│   └── cult_leader.yaml
│
├── items/
│   └── ritual_key.yaml
│
├── rules/
│   ├── movement.yaml
│   └── investigation.yaml
│
└── scenarios/
    └── chapter_01.yaml
```

这样非常适合 Vibe Coding。

例如开发者可以直接告诉 Coding Agent：

> 在 Tavern 东边增加 Market；Market 与 Church 相连；Alice 初始位于 Tavern；21:00 时 Alice 前往 Church。

Agent 直接修改 YAML。

随后：

```text
File change
   ↓
Schema Validation
   ↓
Reload World Definition
   ↓
Renderer Refresh
```

形成一种代码驱动的“所见即所得”。

---

# 7. World Data Model

V1 的 World Model 建议保持在以下几个核心模块。

```text
World
│
├── Clock
│
├── Map
│   ├── Locations
│   ├── Connections
│   └── Location State
│
├── Entities
│   ├── Actors
│   └── Objects
│
├── Relations
│
├── Knowledge
│
├── Flags
│
├── Event Queue
│
└── Event Log
```

---

# 8. Map

Map 负责空间结构。

## 8.1 Location

例如：

```yaml
id: tavern

name: Old Boar Tavern

visual:
  sprite: tavern_01
  x: 8
  y: 12
```

---

## 8.2 Connection

```yaml
id: tavern_market_road

from: tavern
to: market

type: road
```

逻辑层只需要知道：

```text
tavern ↔ market
```

视觉层自行决定如何画道路。

---

## 8.3 Hierarchy

地图允许：

```text
Grayhaven
│
├── Tavern
│   ├── Hall
│   ├── Kitchen
│   └── Room_201
│
├── Church
│   ├── Nave
│   ├── Office
│   └── Basement
│
└── Harbor
```

Entity location 可以直接引用：

```text
church.basement
```

---

# 9. Entity

Entity 表示世界中的对象。

V1 至少包括：

```text
Actor
Object
```

例如：

```yaml
id: alice

type: actor

profile:
  name: Alice
  sprite: alice_red

initial_state:
  location: tavern
  hp: 80
  mood: anxious
```

运行以后：

```json
{
  "id": "alice",
  "location": "church.basement",
  "hp": 42,
  "mood": "terrified"
}
```

Entity Runtime State 与 Entity Definition 分离。

---

# 10. Knowledge

Knowledge 是系统中的一级概念。

不能将：

- 世界事实；
- NPC 知识；
- NPC 误解；
- 玩家知识；

混在一起。

建议至少区分：

```text
World Truth

Actor Knowledge

Actor Belief
```

例如：

```text
World Truth
────────────────────
alice_brother = dead


Alice Belief
────────────────────
alice_brother = alive


Player Knowledge
────────────────────
alice_brother = unknown
```

这样才能支持真正的角色认知差异。

---

# 11. Relations

Relations 单独管理实体关系。

例如：

```yaml
subject: alice
target: player

trust: 20
affection: 5
fear: 0
```

也可以存在非人物关系：

```text
Alice → member_of → Church
Cult → hostile_to → Player
Key → opens → BasementDoor
```

V1 可以保持简单，后续再决定是否发展成完整 Graph Model。

---

# 12. Flags

Flags 用于表示简单离散世界状态。

例如：

```text
cult_awakened = true

bridge_destroyed = false

alice_trusts_player = true
```

适合表示不值得专门创建复杂 Entity 的世界事实。

---

# 13. 世界时间

World Kernel 维护一个 canonical clock。

例如：

```text
Day 3
21:34
```

V1 不使用实时 60 FPS Tick。

时间由行为推动。

例如：

```text
21:34

Player investigates basement

cost = 7 minutes

↓

21:41
```

随后 World Kernel 处理：

```text
所有 scheduled_time <= 21:41
```

的事件。

因此世界运行方式是：

> **Event-driven，而不是 Frame-driven。**

---

# 14. Command

所有世界参与者都只能通过 Command 尝试改变世界。

包括：

```text
Player
NPC Agent
Director
Script
System
```

例如：

```json
{
  "type": "move_entity",
  "actor": "alice",
  "destination": "church"
}
```

或者：

```json
{
  "type": "search",
  "actor": "player",
  "target": "tavern.backdoor"
}
```

Command 表示：

> 某个 Actor 希望世界发生什么。

Command 本身不是事实。

---

# 15. Validation

World Kernel 对 Command 进行验证。

例如：

```text
MOVE Alice → Church

检查：

Alice 是否存在？
Church 是否存在？
Alice 当前能否移动？
当前位置与 Church 是否连通？
连接是否被封锁？
```

失败：

```text
CommandRejected
```

成功：

产生 Event。

---

# 16. Event

Event 表示：

> 已经发生的事实。

例如：

```json
{
  "type": "entity_moved",
  "entity": "alice",
  "from": "tavern",
  "to": "church",
  "timestamp": "D3 21:41"
}
```

Command：

```text
Move Alice to Church
```

是请求。

Event：

```text
Alice moved from Tavern to Church
```

是事实。

二者必须区分。

---

# 17. Reducer

Reducer 根据 Event 更新 Runtime State。

例如：

```text
EntityMoved
    ↓

Alice.location:
tavern → church
```

Reducer 尽量保持：

- deterministic；
- 无 LLM；
- 可测试。

理想形式：

```text
new_state = reduce(old_state, event)
```

---

# 18. Event Queue

Event Queue 保存未来将发生的事件。

例如：

```text
22:00 RitualStarts

23:30 HeavyRainStarts

Day 4 08:00 MerchantArrives
```

World Clock 推进时：

```text
当前时间 >= scheduled_time
```

则处理对应事件。

---

# 19. Event Log

所有已发生事件追加到 Event Log。

例如：

```text
#0314 21:31 PlayerEnteredTavern

#0315 21:37 PlayerSearchedBackdoor

#0316 21:37 SecretPassageDiscovered

#0317 21:41 AliceMovedToChurch
```

Event Log 不允许随意修改过去记录。

---

# 20. Snapshot + Event Log

V1 不需要实现严格的完整 Event Sourcing，但建议采用轻量形式：

```text
Current Snapshot
+
Append-only Event Log
```

Snapshot 用来：

> 快速读取当前世界。

Event Log 用来：

> Debug / Replay / Explain。

数据库初期可直接使用 SQLite。

---

# 21. Replay

Event Log 允许以后实现世界重放。

例如：

```text
Snapshot #100
    +
Event #101
Event #102
...
Event #250

↓

World State #250
```

未来可以进一步支持：

```text
Snapshot #100
      │
      ├── Director A
      │
      └── Director B
```

从同一状态 fork 世界线，用于：

- Director 对比实验；
- Harness Evaluation；
- 回归测试；
- 剧情实验。

这不是 V1 必须实现的功能，但数据结构应避免阻塞它。

---

# 22. Projection

Projection 根据：

```text
World State
+
Viewer Context
```

生成某个观察者能够看到的世界。

统一接口概念：

```text
project(world_state, viewer)
```

---

# 23. Dev Projection

Dev View 显示所有信息：

- 全地图；
- 所有 NPC；
- 隐藏地点；
- 世界秘密；
- Event Queue；
- Event Log；
- NPC Knowledge；
- Director 决策。

它是最重要的 Debug View。

---

# 24. Player Projection

玩家只能看到：

- 已知地点；
- 当前可观察角色；
- 已发现物品；
- 玩家已知事件；
- 玩家认知中的世界。

例如：

```text
World Truth:
Secret Passage exists

Player Knowledge:
Secret Passage unknown
```

则地图不显示 Secret Passage。

当产生：

```text
SecretPassageDiscovered
```

之后：

```text
Player Projection
```

自动增加该地点。

---

# 25. NPC Projection

NPC View 根据：

```text
NPC Knowledge
+
NPC Belief
+
当前感知
```

生成其认知世界。

例如：

Alice 不知道玩家偷了钥匙：

```text
Dev View:
Player owns RitualKey

Alice View:
Unknown
```

这套机制同时可以为 NPC Agent 提供 Context。

---

# 26. Director Projection

Director 不一定需要全部原始数据。

Director View 应重点包含：

- 当前世界状态；
- 玩家位置；
- NPC 状态；
- 世界时间；
- Upcoming Events；
- Narrative Threads；
- 未发现线索；
- 当前冲突；
- 玩家近期行为；
- 世界节奏指标。

Director 根据这些信息决定是否进行叙事干预。

---

# 27. Director 不允许直接修改世界

Director 只能提出 Command / Intervention。

例如：

```json
{
  "type": "move_entity",
  "entity": "alice",
  "destination": "church",
  "reason": "advance cult storyline"
}
```

World Kernel 仍然负责：

```text
Validate
   ↓
Event
   ↓
Reducer
```

Director 不拥有特殊后门。

因此：

```text
Player
NPC
Director
Script
```

都遵守同一个世界协议。

---

# 28. Visual Renderer

Visual Renderer 不是游戏引擎。

V1 使用非常简单的：

- Tile；
- Sprite；
- Icon；
- Label；
- Status Badge；
- Connection Line。

例如：

```text
      Old Church
          👤 Alice
             ⚠

             │

Tavern ─── Market
 🙂
Player
```

其目标是：

> 让世界状态肉眼可理解。

而不是追求游戏画面质量。

---

# 29. Visual Asset

结构化 World Definition 可以指定：

```yaml
visual:
  sprite: alice_red
```

或者：

```yaml
visual:
  tile: church_ruin_02
```

Asset ID 与真实资源路径分离。

例如：

```text
alice_red
```

由 Renderer 解析到：

```text
/assets/npc/alice_red.png
```

以后可以替换素材而不影响 World State。

---

# 30. 前端同步机制

世界只有发生变化时才更新视图。

推荐：

```text
Command
   ↓
Event
   ↓
Reducer
   ↓
State Changed
   ↓
Projection Changed
   ↓
WebSocket / SSE
   ↓
Frontend Patch
```

例如：

```json
{
  "type": "entity.moved",
  "entity": "alice",
  "from": "tavern",
  "to": "church"
}
```

前端只需要更新 Alice 的位置。

不需要持续轮询世界。

---

# 31. 示例完整流程

玩家输入：

> 我检查酒馆后门。

系统：

```text
Player Input
    ↓
Action Interpreter
    ↓
Search(
  player,
  tavern.backdoor
)
```

World Kernel：

```text
Validation
```

检查：

- Player 是否在 Tavern；
- Backdoor 是否存在；
- 是否允许 Search。

验证成功。

产生：

```text
SearchCompleted

TimeAdvanced +7m

SecretPassageDiscovered
```

Reducer：

```text
Clock:
21:30 → 21:37

PlayerKnowledge:
+ secret_passage
```

Projection：

```text
Player View

Tavern
   │
   │ NEW
   ▼
Secret Passage
```

Event Log：

```text
21:37 Player searched Tavern Backdoor
21:37 Player discovered Secret Passage
```

整个过程不需要 LLM 直接修改数据库。

---

# 32. 推荐 V1 技术边界

第一版建议：

### Authoring

```text
YAML / JSON
```

必要时使用 Markdown 存较长的：

- 角色背景；
- 地点描述；
- Lore。

---

### Runtime

一个普通应用进程即可。

无需微服务。

---

### Database

```text
SQLite
```

保存：

- Runtime Snapshot；
- Event Log；
- Event Queue。

---

### Backend

具体语言和框架可以后续决定。

World Kernel 应尽量保持独立，不与 LLM Provider 强绑定。

---

### Frontend

普通 Web UI。

渲染：

- 地图；
- Entity Sprite；
- Timeline；
- Detail Panel。

无需真正游戏引擎。

---

# 33. V1 推荐界面

V1 可以只实现一个 Dev View。

布局参考：

```text
┌──────────────────────────────────────────────┐
│ WORLD                                      │
│ Day 3 · 21:34                             │
├───────────────────────────┬──────────────────┤
│                           │ Alice            │
│        MAP                │                  │
│                           │ Location: Church │
│ Tavern —— Market          │ HP: 42           │
│              │            │ Fear: 73         │
│            Church         │                  │
│              Alice        │ Knowledge        │
│                           │ ...              │
├───────────────────────────┴──────────────────┤
│ Timeline                                    │
│ 21:20 Player entered Tavern                 │
│ 21:31 Alice left Tavern                     │
│ 21:34 Director intervention                 │
├──────────────────────────────────────────────┤
│ Upcoming Events                             │
│ 22:00 Ritual Begins                         │
└──────────────────────────────────────────────┘
```

第一版甚至不需要玩家 UI。

先把：

> **World Runtime → Dev View**

闭环跑通。

---

# 34. 核心不变量

以下规则建议直接写进代码和测试。

### Invariant 1

世界运行时状态只有一个 SSOT。

---

### Invariant 2

Map 不重复存 Entity Location。

---

### Invariant 3

Projection 不拥有 canonical state。

---

### Invariant 4

任何 Actor 不得绕过 World Kernel 修改 Runtime State。

---

### Invariant 5

World State 的变化必须对应 Event。

---

### Invariant 6

Reducer 尽量 deterministic。

---

### Invariant 7

Visual 坐标不参与核心世界逻辑。

---

### Invariant 8

World Truth 与 Actor Knowledge / Belief 分离。

---

### Invariant 9

World Definition 与 Runtime State 分离。

---

### Invariant 10

V1 优先 Event-driven，不建立实时 simulation loop。

---

# 35. V1 最小功能范围

第一阶段只需要完成：

## World Definition

支持：

- Location；
- Connection；
- Actor；
- Object；
- initial state。

## Runtime

支持：

- Clock；
- Entity Location；
- Flags；
- Event Queue；
- Event Log。

## Actions

只实现少数核心 Command：

```text
Move
Wait
Inspect
Interact
SetFlag
```

## Events

对应：

```text
EntityMoved
TimeAdvanced
EntityInspected
InteractionOccurred
FlagChanged
```

## Projection

先实现：

```text
Dev Projection
Player Projection
```

## Visual

支持：

- 地图；
- NPC；
- 物品；
- 基础状态；
- Timeline。

---

# 36. 后续扩展方向

World Kernel 稳定以后，可以逐步加入：

### NPC

- Goal；
- Memory；
- Planning；
- Social Relation。

### World

- Weather；
- Combat；
- Economy；
- Dynamic Factions。

### Director

- Tension；
- Pacing；
- Narrative Thread；
- Intervention Budget。

### Projection

- NPC View；
- Director View；
- Knowledge Overlay；
- Relation Graph。

### Runtime

- Replay；
- Branch；
- Snapshot Fork；
- Scenario Evaluation。

---

# 37. 最终系统定位

本项目不应被定义为：

> 一个带地图的 AI 酒馆。

而应该定义为：

> **一个面向 Agent Narrative 的可视化 World Runtime。**

它通过：

```text
Structured World Definition
        +
Event-driven World Kernel
        +
Persistent Runtime State
        +
Knowledge-aware Projection
        +
Director Harness
```

让一个世界真正脱离 LLM 上下文独立存在。

其中 RPGMaker 风格视觉层不是世界本身，而是：

> **世界状态的低成本、可解释、所见即所得的视觉投影。**

最终形成：

```text
Structure
   ↓
World
   ↓
Events
   ↓
State
   ↓
Projection
   ↓
Narrative
```

而 Director AI 工作在这一整套稳定世界系统之上，而不是负责凭空维护世界本身。