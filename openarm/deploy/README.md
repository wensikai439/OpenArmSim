# 本地接入入口

先看项目 `docs/真机部署.md` 或压缩包顶层 `README-FIRST.md`。

- `config.json`：影子规划配置；运动固定禁用，标定证据仍待补。
- `observation.example.json`：实测快照的空模板；默认拒绝规划。
- `./run real preflight`：列出缺项。
- `./run real replay`：读取已有真实关节记录，不执行动作。
- `./run real demo`：合成快照的离线规划自检，不能算真机试验。
- `./run real shadow --observation 快照.json`：从实测快照计算下一阶段候选路径，不发送运动。

本包没有真机运动驱动或感知服务，也没有 `real execute` 命令。不要将参考控制器的构造器用于只读检查，它会使能电机。
