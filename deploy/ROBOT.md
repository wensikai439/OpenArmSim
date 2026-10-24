# OpenArm 真机 Harness

这个目录只保留真机 agent、API 配置、凝练的瓶身技能和 NVIDIA ASPIRE 官方执行器。没有 MuJoCo、仿真模型、仿真页面、数据集或仿真训练环境。

```text
LLMHarness/
├── run                 启动入口
├── config.json         真机服务地址与技能校验和
├── harness/            API agent 与真机工具客户端
├── skills/             抓下半瓶身→左手交接→入篮程序及说明
├── third_party/aspire/  固定版本官方执行器、注册表与许可证
├── .private/           同一 API 的私有配置，仅所有者可读
├── .venv/              Python + NumPy，无仿真依赖
├── .runtime/           本目录缓存和临时文件
└── records/            接入状态、调用记录和原文件校验
```

```sh
./run status
./run verify
./run api-test
./run chat "看一下现在能不能开始抓瓶子，先不要运动"
```

API 使用与本地技能实验相同的地址、模型和密钥。密钥不进入源码、打包文件、Git、命令行或调用日志，仅保存在本目录权限600的 `.private/api.json`；父目录权限700。这里不写系统密钥环。

## 当前的缺项

凝练程序的14个动作/观测工具，目前仍需接入真正的瓶篮位姿、夹持证据、运动规划和执行服务。原有 ACT/DP/π0.5 的权重与启动代码保持原位、只读；路径记录不代表已经接到此技能。安装与 API 连通不等于能直接抓瓶，不能发送仿真关节轨迹代替适配。

`config.json` 的 `robot_service` 在真实服务验收前保持null。缺少服务时，agent会报告缺项，`./run skill --execute` 也返回blocked、0运动命令，不假装动作已完成。没有自动启动任何相机、CAN或机械臂。

真实服务只接受显式配置的本机HTTP地址：`GET /status` 应返回契约 `openarm-body-v3/1`、`source=real_robot`、标定已验收和14个工具；`GET /observation` 应提供 `source=measured_perception`、基座坐标系、200ms内时间戳、实测物体位姿/夹持证据及技能需要的状态。`POST /tools/工具名` 需要实现原技能的参数、前置条件和回执，真实动作由确定性控制器执行。这个API服务目前没有实现，不能把接口客户端说成驱动。

实现和验证服务之后，操作员在场时才能使用 `./run chat "抓瓶子并交接入篮" --execute` 或 `./run skill --execute`。本次部署没有执行这两条命令。LLM只选工具，冻结程序由官方ASPIRE执行器调用14个真实工具；没有通过大模型直接发关节、CAN或高频指令。程序结束也不自动等于真实成功，还需要独立的释放、持瓶和篮底支撑证据。

所有后续文件、日志、缓存和环境只能写在本 `LLMHarness/` 内。其他远程目录、原模型配置、驱动、环境和权重都不能修改。没有系统服务、自动启动或定时任务。

来源：[NVIDIA ASPIRE](https://github.com/NVlabs/ASPIRE)，固定执行器提交 `fa00f0ef2bc1d91d594b9aa502c22133c9f538f7`。这是OpenArm自定义接入客户端，不是NVIDIA官方OpenArm驱动。
