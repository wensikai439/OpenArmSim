# OpenArm Lab 同事交接包

这是可解压运行的 OpenArm 仿真与 harness 源码快照。使用 Ubuntu 22.04/24.04、Python 3.11；不需要连接真机。包内含 OpenArm MuJoCo 模型、当前 TAMP/harness、网页工作台和少量只读真机参考数据，不含 Git 历史。

## 安装与启动

将 ZIP 与 `SHA256SUMS-20261013.txt` 放在同一目录，先运行 `sha256sum -c SHA256SUMS-20261013.txt`，再解压。进入解压出的 `OpenArmSim` 目录：

```sh
chmod +x run
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r sim/requirements.txt
./run start
```

浏览器访问 http://127.0.0.1:8767/ 。`./run start` 会把当前解压位置写入用户级 systemd 服务配置；以后用 `./run status` 查看、`./run logs` 查日志、`./run stop` 停止。若只想前台运行，可用 `./run objects --foreground`，按 Ctrl+C 退出。EGL 初始化失败时，在有桌面会话的机器上试 `MUJOCO_GL=glfw ./run objects --foreground`。

页面不配置 API 也可启动和查看仿真。需要模型 API 时，在页面里填入团队自己的兼容 API 配置；此包不含密钥。不要把密钥提交进 Git 或交接包。

## 离线验收

首次启动前可运行以下只读物理 smoke 和固定 TAMP 案例：

```sh
.venv/bin/python -B tools/verify_handoff.py
./run tamp --task pick_bottle
```

`verify_handoff.py` 会核对包内文件清单与哈希、检查采集帧/时钟数据，并只在 MuJoCo 中推进一个参考状态。第二条命令是固定场景的仿真抓取案例。两者都不调用模型 API、不访问远程主机，也不代表任意物体泛化或真机迁移通过。输出记录保存在本地 `runs/`。

## 能力范围

- 页面包括对话、场景相机、阶段和关键帧；LLM 可按注册契约选择当前支持的 TAMP 技能。
- TAMP 以仿真场景几何作为 oracle，技能只覆盖代码声明的机器人、物体、姿态和起始条件；它不是通用视觉抓取器。
- ACT 与 π0.5 权重、训练数据及独立推理环境没有打包。cuRobo 也需要单独准备兼容 CUDA 环境。后端不可用时会显示未就绪，不会把失败静默切换成其他算法。
- 包含的三段真机参考轨迹和相机帧用于对照与回归；它们不是独立机械参数标定，也不能证明 sim-to-real。

交接包清单 `PACKAGE-MANIFEST.json` 逐文件记录大小与 SHA-256。代码和上游模型资产各自保留来源版本；导出的内部参考资料仅供团队按权限使用，请勿公开上传。包中不含训练权重、个人 API 配置、数据库、虚拟环境、缓存或历史大型实验目录。
