# PhyAgentOS 控制台

启动、配置和依赖见 [框架说明](../README.md)。此目录保留任务对话、页内模型切换、三路观察、调用记录、停止、录像展示与管理页面。

- 服务端：`server.py`、`management.py`、`providers.py`。
- 真机命令：`real_commands.py`，经现有 Skill / AgentTask / Forge 绑定下发；模型不直接发送 CAN。
- `real_demo.py` 为历史逐段 demo，依赖外部 run 中的计划与 dispatcher，不是干净克隆后可运行的机器人示例。
- `real_preparation.py` 为旧离线策略准备代码，保留源码不表示当前四 CAN 链路支持策略执行。
- 前端：`web/`，使用 npm lockfile 构建。

历史录像目录通过 `PAOS_ARCHIVE_ROOT` 指定；未配置时显示实际存在的文件数。默认保留三组录像入口，不向仓库提交视频。新的运行输出保存在本框架 runs/，模型配置和工作区保存在 PAOS_CONFIG 指定的实例目录。
