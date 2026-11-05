# 外部依赖与历史入口

## 独立环境

Agent/控制台使用 Python >=3.11，依赖分别见 `PhyAgentOS-core/pyproject.toml` 和 `requirements-console.txt`。前端使用 Node 22 和 `console/web/package-lock.json`。本次验证复用了现有环境，未新建或更改生产环境。

仿真 Skill 的 provider 使用 Agent 环境；物理子进程使用 `$PIPER_SIM_ROOT/.venv/bin/python`，需要匹配的 MuJoCo / robosuite / RoboCasa 和 `piper_robocasa` 共享资产。它需要 air_handoff_task、pose_preflight、场景 JSON 和机器人资源等；这里只包含 PhyAgentOS 自有 RPC 和适配代码。RPent 也消费这套共享资产，并不拥有它。当前 Skill 不导入 RPent；旧 probe/debug 脚本例外。

真机适配依赖 `$PIPER_CONTROL_ROOT` 下已有的 robot-control：`tools/replay_web.py`、`replay_supervisor.py`、`replay_handoff.py`、`right_follow.py`、`dual_follow.py`、`piper_baseline` 与已登记的设备配置。相对 IK 从 `upstream/pyAgxArm/pyAgxArm/api/constants.py` 读取官方 PiPER MDH，使用 SciPy least_squares，不是 cuRobo。控制解释器由 `PIPER_CONTROL_PYTHON` 指定，默认 /usr/bin/python3；原部署为 Python 3.10、NumPy 1.21.5、SciPy 1.8.0。相机另需该环境原有的 ROS Humble、OpenCV 和视频依赖。禁止将 Agent 或训练依赖安装进控制环境。

相机启动沿用用户态 systemd 的 2 GiB 内存上限，显式传入项目与控制根目录；ROS setup 路径可用 `PIPER_ROS_SETUP` 配置。主机没有相应服务/设备时应报告缺失，不能以虚拟画面代替。

## 可配置路径

参照 `config/local.env.example`。不存在的默认 external/ 路径只是依赖位置，不会被自动下载、安装或连到原服务器。真机配置模板使用当前 `native-four-can-replay/v1` 且默认 offline。

`PAOS_ARCHIVE_ROOT` 可指向历史录像所在的 runs 目录。三组子目录为 `console-bottle-demo-20261024T093533`、`left-bottle-reset-then-black-box-20261024T021937`、`bottle-to-black-box-20261024T014103`。无需复制到仓库。

编号腕部 30 Hz 资产、编译模型、场景 source lock 不在本包中。`export_wrist_top30.sh` / `reset_wrist_top30.sh` 需要另行提供对应资产和场景。不能只把数据频率参数改为 30 就声称获得真实 30 Hz 仿真记录。

## 脚本分类

- 当前控制台依赖：`real_command_observe.py`、`real_command_stage.py`、`real_command_ik.py`。
- 当前仿真运行入口：`start_piper_sim.sh`、`reset_sim_scene.py`、`run_sim_expert.py`、`run_independent_sim_batch.py`。专家程序、输入场景及模型审阅服务需要按参数提供，不包含在框架源码中。
- 数据与训练工具：`prepare_*`、`export_*`、`align_act_domains.py`、`train_binary_*`、`infer_binary_act_checkpoint.py` 等。需要实际数据、登记文件、原模型项目和独立训练环境；部分历史默认值指向未附带的 runs。不会自动开始训练。
- 历史固定实验：`run_demo.sh`、`probe_rpent.py`、`debug_rpent_handoff.py`、`replay_rpent_handoff.py`、`run_rpent_console_task.py`、`run_*26*`、`run_fast_*`、`run_refresh_*`、`run_binary_expert.sh`、`run_mixed_act.sh` 等。它们保留原入口与实验条件，需要对应历史 bundle、expert.py、场景或数据；不作为干净克隆后开箱即用的命令。
- `checks/smoke_*` 涉及真实服务或原生接口，发布准备没有运行。`checks/test_real_runtime_*` 依赖外部控制源码，不能将缺失底座的检查改成假成功。

为迁移保留源码不代表已复测所有入口。所选测试范围和未验证项见 VALIDATION.md。
