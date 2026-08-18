# Living Tabletop V0 锁定设计

## 产品形态

- 本地单人 Web Demo，无登录、无多人、无部署依赖。
- 可从目录选择的多个数据化场景；当前包含原创验证场景《圣玛丽医院》和经典模组结构复现《科比特宅邸》。每个存档绑定自己的 `scenario_id`，加载和行动时由服务路由到对应引擎。
- 正式界面隐藏 Director；Developer Mode 显示决策原因、世界时钟、事件队列、NPC 位置、威胁和事件日志。
- 规则采用 CoC 7版 Quick-Start 兼容子集，显示检定值、候选骰、成功等级、SAN 与规则选择。

## 权威与写入边界

```text
Scenario Definition
        ↓
World Kernel ──→ Canonical State + append-only Event Log
        ↓
Player/NPC Knowledge Projections
        ↓
Narrator Text
```

玩家主动输入的文字一律交给 KP LLM 理解，不经过关键词、别名或局部文本匹配。LLM 可以在完整语义严格一致时选择一个预设动作，否则生成 `OpenActionPlan`，包括行动目标、耗时、检定方式、风险和可选的新地点。按钮携带明确 `action_id`，无需再次解释。LLM 只能返回结构化提议，所有副作用仍由 Kernel 执行并校验；模型不可用或输出无效时 fail closed，存档保持不变。

```text
任意玩家文本
  ├─ 精确命中预设动作 ───────────────┐
  └─ KP：OpenActionPlan ── CoC 判定 ─┼─→ Kernel 安全效果 ─→ Canonical State
                                      └─→ Narrator 可见叙事
```

因此“用户选择永远是对的”指任何选择都会被接纳为一次尝试，而不是玩家可以宣告成功或绕过硬事实。锁门、距离、时间、检定失败和途中事件仍然成立。

## 异步叙事与逐段阅读

世界结算不等待 Narrator。一次行动完成后，服务立即保存 Canonical State，并返回一个 `NarrativeSequence`：

```text
World/Rules 结算（毫秒级）──→ 保存 state_version N ──→ 立即返回作者 Beats
                                      │
                                      └─→ 后台 Narrator ──→ 仅向版本 N 追加 Generated Beats
```

- `NarrativeBeat` 只有表现文本、来源与可跳过标记，无世界副作用。
- “继续”只移动浏览器中的阅读游标，不发起世界行动，也不推进虚拟时间。
- “跳过描写”停止前端轮询，但不改变已经结算的结果。
- 玩家输入新行动时立即停止旧序列的播放；后台结果必须同时匹配 `sequence_id` 和 `state_version`，否则丢弃。
- 场景实体可在 `attributes.entry_beats` 中提供首访素材；关键事实仍由 Fact/Clue 与 Kernel 管理，不能只存在于可跳过文字里。
- 模型不可用时，序列以作者段落正常结束，游戏仍然完整可玩。

当前状态保存在 SQLite snapshot 中，事件单独追加保存。Snapshot 是加载入口，Event Log 用于审计、测试和确定性重建验证，而不是每次启动都强制全量重放。

## 行动与时间语义

1. 接收玩家意图；精确预设动作校验前置条件，开放行动由 KP 仲裁为自动、检定或物理上无法达成的尝试。
2. 记录 `action_started`。
3. 推进到行动预计结束时间，按 `(time, priority, id)` 执行途中事件。
4. 遇到 `interrupt_action=true` 的事件时停止推进，行动不结算并返回新的 Scene。
5. 未中断则执行技能/属性检定；适用时追加奖励/惩罚骰或对抗骰。
6. 普通失败可进入规则选择：接受失败、消耗幸运，或孤注一掷。等待选择期间不会提前应用成功/失败副作用。
7. 最终结果确定后，执行结构化效果、SAN 检定与伤害/重伤结算。
8. 更新 PlayerKnowledge、NPC Knowledge、威胁和 Experience Telemetry。
9. Director 在触发点从合法 intervention pool 选择机会，Kernel 再次验证；连续开放路线只触发世界内柔性提示，不强迫回归主线。
10. 立即返回作者 Narrative Beats；Narrator 在后台只接收可见结果并追加表现段落。

## CoC 7版基础规则层

- `roll <= skill`：成功。
- `roll <= skill / 2`：困难成功。
- `roll <= skill / 5`：极难成功。
- 01：大成功。
- 100，或技能低于 50 时的 96–100：大失败。
- 奖励/惩罚骰复用个位骰，选择更低/更高的十位组合；各最多两枚，正负互相抵消。
- 普通失败可按目标成功等级消耗差值幸运；大失败、孤注一掷和对抗检定不能这样改写。
- 非战斗、非对抗检定可孤注一掷；二次失败会应用场景专属后果，或推进主要威胁。
- SAN 使用成功损失/失败损失表达式；单次损失至少 5 点时进行 INT 检定，每日累计达到初始 SAN 的五分之一时触发不定期疯狂。
- 单次伤害达到最大 HP 一半会造成重伤并进行 CON 检定；单击达到最大 HP 才会立即死亡。单人 Demo 中 0 HP 会以“失去行动能力”结束当前调查。

完整范围与未实现项见 `docs/COC_RULES.md`。

## Canon 分层

- `hard_canon`：核心秘密、身份、已发生事件、公开线索；Director/NPC/LLM 永不可改。
- `soft_canon`：场景预设但尚未公开的低风险细节，可在合法效果中确定一次。
- `flavor`：不影响规则和推理链的对话色彩；可以临场生成，但不会写入 Fact，除非显式升级并通过校验。

## Director V0

确定性层计算 `tension / danger / progress / mystery / resource_pressure / success_streak / failure_streak / agency / frustration / relief_need / novelty / time_pressure`。Director 只在重大事件、场景切换、连续成功/失败、长期无进展、濒死或威胁跨阶段时运行。

首版允许五类干预：

- `advance_threat`
- `surface_clue`
- `offer_respite`
- `increase_pressure`
- `guide_affordances`

每次干预必须包含体验原因、世界内理由、受影响实体和预期效果。LLM 可以在合法候选之间建议，但最终由 Kernel 校验；离线时采用可预测的规则策略。

## 验收定义

- 固定种子自动通关能够获得足够证据、解决主要威胁并得到目标结局。
- 脚本玩家与五种策略玩家可离线运行；报告覆盖手写动作、不同终局、骰点结果和 Director 干预。
- 活跃状态无合法动作、合法动作被拒绝、时间倒退、事件序号断裂、公开投影错误和隐藏事实泄漏会让自动测试失败。
- 场景在加载时验证实体、事实、线索、动作条件、效果目标、威胁、事件、结局和移动边引用。
- 后台 NPC 与威胁事件在玩家不在场时继续推进。
- 中断事件可以打断耗时行动。
- NPC 不会凭空获得现场知识。
- immutable fact、非法移动和非法 Director intervention 被拒绝。
- 重载前后 Canonical State、队列、知识、Director 和 RNG draw count 一致。
- 记录过的结构化决策可以在不重新调用 LLM 的情况下重放。
- 回家过夜、模组外旅行和开放 CoC 检定均可执行并回放；开放旅行不得绕过锁门等移动条件。
