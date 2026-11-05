# OpenArm Harness

位置：`/home/sj/Desktop/wuji/lerobot_20260116/LLMHarness/`。

当前安装的是瓶身抓取 v3 和 NVIDIA ASPIRE 固定版本执行器。真实运动未接通；不调用 CAN、相机或现有 ACT/DP/π0.5 启动脚本。没有搬移或重复保存原有权重。

```text
LLMHarness/
├── run                 简短入口
├── README.md           本说明
├── AGENTS.md           后续工作的目录边界
├── DEPLOYMENT.json     来源、校验和与能力状态
├── OpenArmSim/         harness、skills、模型资产和独立 .venv
├── records/            安装、验收和原文件校验记录
└── .runtime/           仅本目录使用的缓存及临时文件
```

进入此目录即可运行：

```sh
./run verify
./run status
./run aspire recover
./run aspire recover --fault receiver_close_loss
./run real replay --episode 0
./run real demo
./run check
```

`verify` 检查包哈希与仿真结构，`recover` 在仿真中执行整个抓瓶交接任务，`demo` 只生成预抓取提案并在发送运动前停止。实验记录位于 `OpenArmSim/runs/`，安装/验收记录位于 `records/`。这些都不代表真机成功。

本目录的 `.venv` 使用现有 `openarm_fastwam` 的 Python 3.12 作为只读基础解释器，依赖安装在本目录内，不使用原环境的模型依赖。移除或迁移基础解释器后需要重建这个虚拟环境。

只允许在本 `LLMHarness/` 内新增、修改或删除本项目文件。相邻 `openarm_deploy/` 和 `src/` 只读；权重路径及三个模型启动入口记录在 `records/models.json`。不能把该路径记录当作已经连通策略执行。不要启动原版 systemd 服务、安装系统包或把 API 配置写入全局密钥环。目录入口暂未开放 API 配置、模型训练、网页服务或真机执行。

下一步是接入新鲜实测瓶篮位姿、左右手夹持证据及经过验收的轨迹执行器，再做硬件分段验证。仿真反馈恢复分支已经部署，真机恢复需要实测证据，不能回滚物理世界或直接播放仿真关节数组。
