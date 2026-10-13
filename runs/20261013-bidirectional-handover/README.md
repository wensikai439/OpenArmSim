# TAMP 左手到右手交接复测

日期：2026-10-13。为复核定向交接能力而新跑的单回合原生仿真，不与旧基线成功率合并。

- 动作：新场景由左手抓起瓶子，再由右手接住；供物手释放后，接收手独立持瓶。
- 结果：两个原生技能均完成；最终 `held={left:false,right:true}`，左手无接触；24.767仿真秒、约1.77墙钟秒。
- 范围：本地 MuJoCo v1、seed 0、oracle场景几何。未调用 LLM/API、cuRobo、ACT、π0.5或真机；不证明由自然语言模型能稳定选对工具，也不证明任意物体或真实迁移。

复现：

```sh
.venv/bin/python -B -m unittest sim.generalization_checks.GeneralizationChecks.test_left_pick_then_transfer_to_right_in_original_scene
```

完整阶段诊断、源码哈希及仿真清单见 [`result.json`](result.json)。
