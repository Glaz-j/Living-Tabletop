# Living Tabletop

一个可运行的 **LLM-first、world-guarded AI TTRPG** 本地 Web Demo。LLM 负责理解玩家、完整组织对话并进行低风险的即兴创作；确定性的 World Kernel 管理虚拟时间、规则结算、硬事实和持久记忆，主要防止真正的世界冲突，而不是把检索结果当成台词白名单。规则层现已接入 CoC 7版 Quick-Start 兼容子集。

项目目前包含两个可选场景：原创框架示例《圣玛丽医院》，以及根据 Chaosium 经典入门模组《The Haunting》重新数据化的《科比特宅邸》。两者共用同一套内核、Director、存档和自由文本解释器。作者预设按钮和确定性规则可以离线运行；玩家主动输入的自由文本始终由 LLM 理解，不再使用关键词或别名模糊路由。

行动的世界结算与叙事表现已经分离：按钮和自由文本都会先形成 `PlayerIntentEnvelope`。自由输入由单一 `TurnComposer` 在一次前台调用中同时理解并生成完整演出；既有证据只是上下文，不是强制拒答条件。模型可以为已存在的人物、地点或物品补充地址、路线、外观、习惯等低风险细节，这些提案通过软事实冲突检查后，与台词在同一次 Kernel 事务中提交。CoC 检定仍由规则层完成，Composer 预先写好的成功/失败分支只展示实际发生的一支。模型不可用时保持输入和存档不变并明确报错；软事实或标点元数据失败时保留有用演出。

## 快速开始

```powershell
python -m pip install -e ".[dev]"
python -m living_tabletop.main
```

打开 <http://127.0.0.1:8000>。

可选的模型配置：

```powershell
Copy-Item .env.example .env
# 在 .env 中填写 LIVING_TABLETOP_API_KEY，并将 LIVING_TABLETOP_LLM_ENABLED 设为 true
```

应用不会把密钥或完整模型输入写入普通日志。存档中的开发观察记录会保留通过校验的结构化决策、调用角色、状态版本、耗时和 token 用量，用于确定性回放。
模型网关遇到连接超时、限流或 5xx 时会做一次短重试；仍失败才短暂打开对应提供方的熔断器。模型已经返回、但 Composer 的辅助 JSON 字段无效时会尽量提取并修复可见演出，同时把危险动作元数据降级为无状态行动；这不会被误记成网络断开，也不会触发 Ollama 熔断。只有没有可用文本或精确泄露隐藏硬事实时整轮不提交。

## Conversation-first Agent Runtime 与本地 Harness

自由输入的正常链路是 `ContextAssembler → TurnComposer → RuleEngine → WorldKernel → NarrativeSequence`。Composer 输出动作理解、1–5 个完整演出段、检定分支和可选软事实提案；Harness 请求严格 JSON Schema，但可见文本与状态元数据采用不同失败策略。玩家原文由服务器持有，模型对空格或标点的正规化不会改变授权；涉及位置变化的自由输入必须包含明确动身承诺，单纯问路、提及地点或“走到窗边但仍留在房间”都不能授权跨场景 MOVE。

NPC 问答不再先经过一条会阻塞演出的检索审批链。Composer 直接获得逐字近期对话、玩家相关已知事实，以及当前在场 NPC 未隐藏且与本轮输入相关的知识；无关事实不会全部塞给本地小模型。没有命中的地址、路线、营业时间、普通名声、外观或习惯可以合理即兴为 `soft_canon`，但谜底、身份、生死、亲属、所有权、伤害和规则数值仍由硬事实保护。`used_fact_ids` 必须来自玩家已知事实或当前说话 NPC 的授权知识，即使模型猜中隐藏 id 也不能披露。每轮 `TurnTrace` 保存输入、上下文摘要、Composer 输出、Kernel 事件、状态差异和 Outcome，且仅在开发者视图中出现。完整设计见 [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md)。

正式游戏默认使用 `auto` 路由：先调用本地 Ollama 的 `qwen3.5:9b-q4_K_M`，本地连接、生成或结构化解析失败时再切换远程 API。右上角“模型”面板可以在运行时切换自动、本地或远程模式，指定已发现的模型并执行真实生成测试；选择会保存在 `data/llm_preferences.json`，不会保存或暴露 API Key。

本地 Ollama 默认以 8192 token 上下文运行（`LIVING_TABLETOP_LOCAL_LLM_CONTEXT_WINDOW=8192`）。ContextAssembler 最多装配 24 条、约 8000 字符的逐字可见历史，并补充相关玩家事实、当前 NPC 相关知识、只读硬状态和可访问实体；可用按钮不会把预设对话台词塞进 Composer，避免旧剧情重播。Composer 不能看到未授权隐藏硬事实的值。旧版 Planner/Dialogue 管线只用于旧录制计划和测试夹具兼容；旧存档仍会幂等补齐新增静态事实、NPC 知识和 flag，不覆盖已经游玩的状态。

也可在不修改正式 `.env` 的情况下单独测试本地 Ollama Harness：

```powershell
python scripts/local_harness_smoke.py --model qwen3.5:9b-q4_K_M --case look_upstairs
python scripts/local_harness_smoke.py --model qwen3.5:9b-q4_K_M --case overnight_rest
```

本地或其他兼容网关还可使用 `LIVING_TABLETOP_LLM_TEMPERATURE` 与 `LIVING_TABLETOP_LLM_REASONING_EFFORT` 调整生成；未设置时保持网关默认行为。

## 测试与自动通关

```powershell
pytest --cov=living_tabletop --cov-report=term-missing
python scripts/playthrough.py
python scripts/self_play.py --scenario all
python scripts/rules_playtest.py
python scripts/benchmark_retrieval.py --iterations 250
python scripts/test_agent.py --scenario all --runs-per-persona 2 --turns 24 --output-dir artifacts/test-agents
```

`playthrough.py` 保留一条《圣玛丽医院》的固定通关路线。`self_play.py` 覆盖作者动作和结局路线；`rules_playtest.py` 覆盖接受失败、幸运改写、孤注一掷、SAN、对抗战和重伤。`test_agent.py` 则用 8 种人格依据当前公开场景动态生成互不重复的自由文本，不点击默认按钮；快速合约模式适合数百轮不变量 fuzzing，`--game-backend live` 可改用正式本地/API 模型。脚本返回非零退出码表示发现错误，详细说明见 [docs/TEST_AGENT.md](docs/TEST_AGENT.md)。

若要保留 JSON 与 Markdown 报告：

```powershell
python scripts/self_play.py --scenario all --output-dir artifacts/playtests
```

检索评测集位于 `evals/retrieval/the_haunting_v1.json`，固定基准报告位于 `artifacts/benchmarks/retrieval-benchmark.md`。详细说明见 [docs/PLAYTESTING.md](docs/PLAYTESTING.md)。

## 场景来源与版权边界

《科比特宅邸》保留《The Haunting》的调查地点、人物关系与核心因果，但所有中文叙事均为重新概述；项目不包含原始 PDF、手卡、地图、美术或规则书原文。规则代码依据 Chaosium 公开 Quick-Start 所述基础机制独立实现，不复制规则表格和文字。免费提供不等于开放许可；公开发布或商业使用这个改编场景前，需要另行确认许可。出处和实现边界见 [docs/SCENARIO_SOURCES.md](docs/SCENARIO_SOURCES.md)。

## 当前 Demo 边界

- 单人、CoC 7版基础规则子集、可选模组、3 个建议行动 + 永久开放行动；按钮只是快捷方式。
- 玩家可以离场、休息、前往模组外地点或永久偏离主线。规则引擎限制的是副作用写入，不是玩家能表达的意图；物理阻碍会让尝试失败，但不会把输入判成“非法”。
- Director 对连续偏离主线只使用世界内的小事件和线索机会进行柔性提醒，不会强制传送或锁死其他路线。
- `NarrativeSequence` 是不推进世界时间的表现层队列；作者段落离线可用，LLM 段落异步追加，并用 `state_version` 防止迟到覆盖。
- 支持成功等级、奖励/惩罚骰、对抗检定、幸运消耗、孤注一掷、SAN、临时/不定期疯狂、重伤、昏迷与骰点伤害。
- 状态表是当前真相；Event Log 是追加式审计与回放来源；Snapshot 用于快速加载。
- 玩家知识与 NPC 知识独立；普通 prose 先作为逐字可见历史保留。只有 TurnComposer 显式提出、通过冲突检查并由 Kernel 记录的 `soft_canon` 才会进入长期世界知识。
- Director 是确定性策略，只能制造机会、压力或 affordance，不能直接授予调查结论；自由输入本身不计作偏离主线。
- “完全回放”指用已记录的输入、模型结构化输出、骰子和事件重建同一世界状态，不承诺重新调用模型得到逐字相同文本。

规则范围与差异见 [docs/COC_RULES.md](docs/COC_RULES.md)，历史设计见 [docs/V0_DESIGN.md](docs/V0_DESIGN.md)，当前架构见 [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md)。
