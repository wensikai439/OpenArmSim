# TAMP 前移放置复测

日期：2026-10-13。为复核“把瓶子往前放”而新跑的单回合原生仿真，不与旧基线成功率合并。

- 初始瓶位：`[0.503, -0.173, 1.005] m`；机器人世界系 +X 定义为前方。
- 原目标：相对起点前移 `0.08 m`，锁定为 `[0.583, -0.173] m`。
- 结果：TAMP `pick → place` 均有原生回执；最终水平误差 `2.38 mm`；支撑、释放和验收均确认；543 控制步、18.1 仿真秒、约1.51墙钟秒。
- 范围：本地 MuJoCo v1、seed 0、单次已知瓶子/桌面场景。使用仿真真值；未调用 LLM/API、cuRobo、ACT、π0.5或真机。它只证明这个目标可执行，不证明任意位置泛化或 sim-to-real。

复现：

```sh
.venv/bin/python -B -m unittest sim.relocation_checks.RelocationChecks.test_parameterized_forward_relocation_has_real_grasp_and_independent_placement
```

完整阶段诊断、源码哈希及仿真清单见 [`result.json`](result.json)。
