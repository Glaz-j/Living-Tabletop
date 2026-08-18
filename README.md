# Living Tabletop V0

一个可运行的 **simulation-first AI TTRPG** 本地 Web Demo。确定性的 World Kernel 管理世界事实、虚拟时间、NPC 知识与事件；受约束的 Director 只能从合法机会池中调节节奏；KP 仲裁器接受玩家尝试的任意行动，LLM 负责理解意图和润色叙事，但不能直接修改世界。规则层现已接入 CoC 7版 Quick-Start 兼容子集。

项目目前包含两个可选场景：原创框架示例《圣玛丽医院》，以及根据 Chaosium 经典入门模组《The Haunting》重新数据化的《科比特宅邸》。两者共用同一套内核、Director、存档和自由文本解释器。作者预设按钮和确定性规则可以离线运行；玩家主动输入的自由文本始终由 LLM 理解，不再使用关键词或别名模糊路由。

行动的世界结算与叙事表现已经分离：按钮会先返回规则结果和作者预设的多段场景文字，玩家可用“继续”逐段阅读或直接跳过；Narrator 在后台补充描写。自由输入采用 LLM-first 结构化意图，模型不可用时保持输入和存档不变并明确报错；新行动会中断旧叙事，迟到的模型结果不能覆盖较新的世界状态。

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
模型网关遇到连接超时、限流或 5xx 时会做一次短重试；仍失败则使用确定性 KP 降级，并短暂打开熔断器，避免玩家行动卡死。

## 本地结构化 Harness

Keeper 与 Narrator 的模型输出会经过轻量原生 Harness：优先请求严格 JSON Schema，使用 Pydantic 校验结构及运行时约束；首次结果不合法时携带精确错误修复一次，仍失败则保持既有安全回退。Harness 不直接修改 World State，也不限制玩家输入。

正式游戏默认使用 `auto` 路由：先调用本地 Ollama 的 `qwen3.5:9b-q4_K_M`，本地连接、生成或结构化解析失败时再切换远程 API。右上角“模型”面板可以在运行时切换自动、本地或远程模式，指定已发现的模型并执行真实生成测试；选择会保存在 `data/llm_preferences.json`，不会保存或暴露 API Key。

本地 Ollama 默认以 8192 token 上下文运行（`LIVING_TABLETOP_LOCAL_LLM_CONTEXT_WINDOW=8192`）。系统会保存最近的玩家可见演出，并从最近最多 20 段、约 5000 字符中按当前话题检索相关演出、事实、NPC 知识、物品和候选动作；无关旧剧情不会整包塞进当前提示。作者事实、Keeper 临时演出、Narrator 软事实、NPC 台词与玩家意图具有不同可信层级。普通寒暄默认自动完成，异步演出会保持当前话题并拦截跨历史复读。旧存档没有该字段时，会从已保存的模型结构化输出中恢复并降级标注近期演出，因此无需重开游戏。

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
```

`playthrough.py` 保留一条《圣玛丽医院》的固定通关路线。`self_play.py` 是多路线测试实验室：它会运行研究优先、社交优先、冒险直闯、探索和随机策略，也会执行覆盖指定结局的脚本路线。`rules_playtest.py` 专门覆盖接受失败、幸运改写、孤注一掷成功/失败、SAN、奖励骰对抗战和重伤。脚本返回非零退出码表示发现错误。

若要保留 JSON 与 Markdown 报告：

```powershell
python scripts/self_play.py --scenario all --output-dir artifacts/playtests
```

详细说明见 [docs/PLAYTESTING.md](docs/PLAYTESTING.md)。

## 场景来源与版权边界

《科比特宅邸》保留《The Haunting》的调查地点、人物关系与核心因果，但所有中文叙事均为重新概述；项目不包含原始 PDF、手卡、地图、美术或规则书原文。规则代码依据 Chaosium 公开 Quick-Start 所述基础机制独立实现，不复制规则表格和文字。免费提供不等于开放许可；公开发布或商业使用这个改编场景前，需要另行确认许可。出处和实现边界见 [docs/SCENARIO_SOURCES.md](docs/SCENARIO_SOURCES.md)。

## V0 边界

- 单人、CoC 7版基础规则子集、可选模组、3 个建议行动 + 永久开放行动；按钮只是快捷方式。
- 玩家可以离场、休息、前往模组外地点或永久偏离主线。规则引擎限制的是副作用写入，不是玩家能表达的意图；物理阻碍会让尝试失败，但不会把输入判成“非法”。
- Director 对连续偏离主线只使用世界内的小事件和线索机会进行柔性提醒，不会强制传送或锁死其他路线。
- `NarrativeSequence` 是不推进世界时间的表现层队列；作者段落离线可用，LLM 段落异步追加，并用 `state_version` 防止迟到覆盖。
- 支持成功等级、奖励/惩罚骰、对抗检定、幸运消耗、孤注一掷、SAN、临时/不定期疯狂、重伤、昏迷与骰点伤害。
- 状态表是当前真相；Event Log 是追加式审计与回放来源；Snapshot 用于快速加载。
- 玩家知识与 NPC 知识独立，Narrator 只读取玩家可见投影。
- Director 可以生成低影响环境事实，但不能修改硬事实、骰子结果或已公开线索。
- “完全回放”指用已记录的输入、模型结构化输出、骰子和事件重建同一世界状态，不承诺重新调用模型得到逐字相同文本。

规则范围与差异见 [docs/COC_RULES.md](docs/COC_RULES.md)，详细架构见 [docs/V0_DESIGN.md](docs/V0_DESIGN.md)。
