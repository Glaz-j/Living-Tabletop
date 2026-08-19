# Dynamic Test Agent

Test Agent 是 Living Tabletop V2 的自由文本压力测试器。它与固定通关脚本互补：固定脚本验证作者路线和结局，Test Agent 验证玩家不按剧本说话时系统是否仍然自然、安全且连续。

## 人格

- 沉浸派演员：长对话、关系和情绪连续性；
- 调皮捣蛋鬼：边角互动和怪问题；
- 极限找茬者：问路、否定授权、复合问题；
- 精明谈判家：报价、条件和说话者归属；
- 离题漫游者：天气、生活和离开主线；
- 惜字如金者：省略主语和短指代；
- 混沌即兴者：动作与对话混合；
- 连续性审计员：引用上一轮原话、数字和对象。

每个输入从当前公开视图动态生成，不使用 `action_id`，并在整个报告范围内保证文本唯一。探针只是行为族，不是预设剧情路线。

## 运行

快速大规模不变量测试：

```powershell
python scripts/test_agent.py --scenario all --runs-per-persona 2 --turns 24 --game-backend synthetic --output-dir artifacts/test-agents
```

使用正式本地优先模型路由：

```powershell
python scripts/test_agent.py --scenario the_haunting_corbitt_house_v1 --runs-per-persona 1 --turns 2 --game-backend live
```

让本地/API 模型同时扮演测试玩家：

```powershell
python scripts/test_agent.py --scenario st_mary_hospital_v0 --runs-per-persona 1 --turns 2 --game-backend live --player-backend llm
```

`synthetic` 不是文风评测。它动态实现 Composer 合约，用于快速覆盖正式 `GameEngine → RuleEngine → WorldKernel → NarrativeSequence` 路径。`live` 才能证明真实模型的 JSON Schema 遵循、相关性、创作质量和延迟。

## 自动不变量

- 所有自由输入互不重复；
- 接受的行动必须推进状态版本；
- 纯问路和明确“不去”的表达不能移动玩家；
- 对话必须有可见演出；
- 问路必须给位置/路线，或明确说不知道；
- 连续性探针不能重播上一轮全文；
- NarrativeSequence 必须保留玩家逐字输入；
- 模型不可用与未预期引擎异常分别归类；
- 世界事件中断回答时不误报“没有回答”。

JSON 报告保存每一步的输入、地点前后、版本前后、演出、耗时与失败语料；Markdown 报告用于快速阅读。测试器发现问题后应先修实现或测试器模型，再使用同一 seed 重跑，不能简单删除断言。
