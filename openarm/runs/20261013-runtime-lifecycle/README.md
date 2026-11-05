# 8767工作台生命周期与启动诊断

日期：2026-10-13。范围：本机仿真服务、健康检查和模型worker启动状态。不调用LLM/VLA推理、不训练、不访问公司SSH主机、不连接真机。

## 结果

- 主工作台统一由`./run start`（默认`./run`）管理，监听`127.0.0.1:8767`；`./run status`和`./run stop`只管理`openarm-sim.service`及其ACT/cuRobo依赖。旧8765只保留`./run serve`显式入口。本轮未创建定时器或登录自启动链接。
- 健康接口增加工作台ID、启动时刻、引擎、任务活动状态、build/harness profile、策略配置路径、源码指纹，以及ACT/cuRobo实际连接状态；不返回模型密钥。
- 代码或仿真配置在进程启动后发生变化时，`/health`标记需要重启；`/api/tools`、`/api/chat`、场景变更及重置/回放拒绝新调用并返回HTTP 409。`stop`仍可用。
- 模型worker的systemd失败后不自动反复初始化。显式启动工作台时最多尝试一次，由状态和journal保留失败原因。

## 验收证据

| 检查 | 结果 |
|---|---|
| Shell/Python语法 | `run`通过`sh -n`；四个入口Python文件可编译 |
| 源指纹成本 | 260个运行源文件，完整SHA256检查约0.023秒 |
| 本机相机帧 | `/api/frame?camera=overview`返回有效JPEG，960×540，35,506字节 |
| 普通启停 | `./run stop`后`./run start`重新就绪，8767唯一监听 |
| 源变更门 | 临时给`harness/skills.py`追加一个换行后，工具POST返回409；原字节及mtime恢复后，`list_skills`返回200、13项 |
| 异常恢复 | 服务tick 0、无job时对OpenArmSim cgroup发SIGKILL；systemd约2秒内恢复，health就绪、新epoch、tick 0、job为空 |
| worker就绪 | ACT与cuRobo都没有监听；日志明确为`CUDA unavailable`。两者现在各尝试一次后停止，不伪报可用 |

## GPU诊断（P0.1未完成）

当前内核为`7.0.0-38-generic`。已安装`linux-modules-nvidia-595-open-7.0.0-34-generic`，当前内核目录没有NVIDIA模块，`modinfo nvidia`返回`Module nvidia not found`，`nvidia-smi`无法连接驱动。APT只读演算找到匹配`7.0.0-38`包；再次预演显示将升级20个包、安装2个包并移除旧内核模块包。实际包状态未改变；已授权安装命令通过本机图形认证框等待用户输入。用户此前要求暂不重启，因此安装后也不会自动重启；加载模块和GPU验收留待之后维护窗口。

## 源码身份

初始备份在本目录`before/`；服务入口五个文件按SHA256记录在`before_sha256.json`。ACT/cuRobo单元旧版本从2026-10-11同事ZIP恢复保存。当前源码在尚未建立父项目Git的工作区中；health报告的当前运行源指纹为`63b961363ec0f9e920cde7a6fa772bc5c92778132ed266897217306f92f31e9d`（260个文件）。
