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

每个输入从当前公开视图动态生成，不使用 `action_id`。单局内不会重复同一句输入；不同随机种子的独立游戏允许复用相同句意，以验证相同请求在不同世界状态与事件序列中的结果。探针只是行为族，不是预设剧情路线。

## 运行

快速大规模不变量测试：

```powershell
python scripts/test_agent.py --scenario all --runs-per-persona 2 --turns 24 --game-backend synthetic --output-dir artifacts/test-agents
```

可恢复的 1024 局长跑：

```powershell
python scripts/test_agent.py --scenario all --total-runs 1024 --turns 32 --game-backend synthetic --output-dir .artifacts/test-agents/nightly-1024
```

`--total-runs` 是所有已选剧本与人格合计的精确局数，并采用轮转方式均衡分配。运行器默认 `--resume`：每局结束立即原子写入 `runs/*.json`，随后更新 `summary.json`、`summary.md` 与 `failures.jsonl`；进程中止后执行同一命令会跳过已完成的 run。更换局数、回合数或 backend 时必须使用新的输出目录，避免把不同实验混在一起。

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
- 取得、交还和仅查看物品必须与背包增减及 `item_acquired` / `item_removed` 事件一致；
- 背包不能有重复、缺失、失活或位置错误的实体；
- 每轮世界状态必须能够完整序列化并重新通过模型校验；
- Director 的内部世界解释不能泄漏到玩家演出；
- 模型不可用与未预期引擎异常分别归类；
- 世界事件中断回答时不误报“没有回答”。

JSON 检查点保存每一步的输入、地点前后、背包前后、事件类型、版本前后、演出、耗时与失败语料；Markdown 报告用于快速阅读。测试器发现问题后应先判断是产品缺陷还是合成 Composer 误报，修正后使用同一 seed 重跑，不能简单删除断言。
