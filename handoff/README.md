# 最新接入包 · 2026-10-24

`OpenArmSkill-20261024-v2.zip`（约34.8MB）与同名 `.sha256` 是本轮交付。解压后先看 `OpenArmSim/README-FIRST.md`。

新版在6个新瓶位、12条件的成对测试中由6/12提高到12/12；365项回归及3项包校验测试通过。已用全新隔离依赖环境完成解压、仿真skill、回放、影子规划、拒绝过期/缺失接口和离线再打包检查。当前包没有真实三维感知服务或真机执行适配器，运动默认禁用；本轮没有连接、上传或修改真机电脑。

本轮ZIP SHA-256：`8932708c97f063f964131562f63c0ce4d940b2e6052380b0b4bf8ff0093cfd71`。

以下保留历史交付说明。

# OpenArm 技能交接包

当前技能包：`OpenArmSkill-20261024.zip`，校验文件：`OpenArmSkill-20261024.sha256`。

请先阅读项目 [真机部署说明](../docs/真机部署.md)。新包包含抓瓶交接v2、harness、MuJoCo模型和固定版本ASPIRE运行时；不含真实硬件驱动、密钥或模型权重。它可用于本地仿真复现及真机适配开发，不能解压后直接操作OpenArm。

重新打包：在项目目录运行 `./run aspire package`。已有同名文件时拒绝覆盖，可用 `--output handoff/其他名称.zip` 生成新版本。

---

# 旧版同事包说明（2026-10-13）

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

浏览器访问 http://127.0.0.1:8767/ 。`./run start` 会把当前解压位置写入用户级 systemd 服务配置；以后用 `./run status` 查看、`./run logs` 查日志、`./run stop` 停止。若只想前台运行，可用 `./run objects --foreground`，按 Ctrl+C 退出；可用 `OPENARM_PORT=18765 ./run objects --foreground` 启动独立开发实例。EGL 初始化失败时，在有桌面会话的机器上试 `MUJOCO_GL=glfw ./run objects --foreground`。

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
