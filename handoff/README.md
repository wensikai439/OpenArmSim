# OpenArmSim 同事交接包

这是 OpenArmSim 当前工作区的可运行源码快照，面向 Ubuntu/Linux 本地仿真与 harness 开发。压缩包解开后即可安装 Python 依赖并启动仿真页面；它不包含真机控制入口。

## 快速启动

建议使用 Ubuntu 22.04/24.04、Python 3.11。需要有可用的图形渲染环境；NVIDIA 主机优先使用 EGL。先把 ZIP 与 SHA256SUMS.txt 放在同一目录并运行 sha256sum -c SHA256SUMS.txt 验证传输，再解压。解压后在项目目录运行（若解压工具没有保留脚本权限，chmod 会补上）：

~~~sh
chmod +x run
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r sim/requirements.txt
./run start
~~~

浏览器打开 http://127.0.0.1:8767/ 。默认页面是多物体仿真工作台，服务仅监听本机并由用户级 systemd 管理。查看状态用`./run status`，停止用`./run stop`。若 EGL 初始化失败且机器正在运行桌面会话，可试：

~~~sh
MUJOCO_GL=glfw ./run objects --foreground
~~~

不需要 API 即可启动和手动检查仿真。要让 LLM 调用 harness，在页面的 API 配置中填入同事自己的兼容 API 地址、模型名和密钥；本包不含任何密钥，也不会替同事配置账户。

## 最小验收

首次安装后建议运行：

~~~sh
.venv/bin/python -B tools/verify_handoff.py
./run tamp --task pick_bottle
~~~

第一条检查真机参考图像与轨迹的哈希/时间对齐，并在 MuJoCo 中推进一个真实轨迹控制步；第二条运行瓶子抓取基线。页面的常用入口是「TAMP」技能和自由相机。任务完成以仿真器的物理评分为准；模型文字判断与仿真结果分开记录。

## 包含内容

- 当前 MuJoCo 环境、TAMP/harness、页面前端、物体场景和运行工具。
- OpenArm v1 模型及所需 cell 资源；上游文件保留在 openarm_mujoco/，其许可证随包提供。
- 三条已导入的真机轨迹、对应采集时钟和挑选的相机参考帧，供仿真回放/诊断。
- reference/source/inputs/ 下只读导出的 OpenArm URDF、控制器、相机、记录器及 IK 源码快照；source manifest 保留内容哈希。
- sim/profile.json 与 calibrate.json 中当前仿真的候选参数及其未标定说明。
- PACKAGE-MANIFEST.json，列出包内其余每个文件的大小和 SHA-256，可供同事检查传输是否完整。

本包清理了源数据里的个人绝对路径和远端主机名；相对文件名与导出文件哈希保留，未改动文件仍保留原始来源哈希。真机源文件是团队内部工作资料，按公司内部权限使用，不要公开上传。生成此包只读取本机已有的导出副本，没有连接或修改 SSH 电脑。

## 当前能做什么，以及尚未接入的部分

TAMP 可在当前注册的场景/技能范围内使用仿真状态规划末端位姿与执行抓取、交接、放置；它不是任意物体的通用视觉抓取器。MuJoCo 是物理仿真，模型 API 可由同事自行配置。系统只运行本机仿真。

cuRobo 需要单独准备兼容的 CUDA/cuRobo 运行环境；本包不带 GPU 环境。如果服务不可用，先用 TAMP 基线。ACT 检查点、LeRobot 环境和个人路径配置未打包：现有 ACT 完整仿真任务尚未通过，直接附上权重会形成无法独立启动的假象。π0.5 OpenArm 检查点也未随包提供。要接入 ACT/π0.5，需另行准备匹配 OpenArm 16 维状态/动作与三路相机契约的本地模型服务，并按 harness/policy.example.json 配置。

这些限制同样适用于真机迁移：已回放轨迹、源代码或仿真任务通过都不等于独立实物标定完成，也不构成 sim-to-real 成功证据。瓶子/篮子几何、桌面/基座、相机内外参、夹爪映射和控制器参数仍有候选值，报告标签以 transfer_ready=false 为准。

## 文件与数据边界

包内没有历史 runs/ 大目录、训练数据集、ACT/π0.5 权重、API 配置、聊天记忆数据库、SQLite 状态、虚拟环境、缓存或 systemd 持久服务。完整开发回归含有指向特定历史 runs/ 报告的检查，因历史实验未随包分发，请用上面的便携验收步骤检查交付包。也移除了原始 SSH 数据导入脚本和个人路径。解压后产生的 .venv 与新实验记录属于同事自己的本地数据。

这是源码快照，不含 Git 提交历史。OpenArm 上游模型许可证在 openarm_mujoco/LICENSE；真机导出源码和轨迹仅供组织内部按授权使用。开始改动前建议同事先把此目录纳入团队私有 Git，再用提交记录管理后续版本。
