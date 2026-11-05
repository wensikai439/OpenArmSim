# PhyAgentOS 本地适配

这是一套独立的 PhyAgentOS Agent / Forge / 控制台实现，位于 `松灵机械臂/phyagentos/`。它与现有 Harness、ASPIRE 并列，未替换它们的服务、前端或机器人接口。**尚未适配 OpenArm 机器人；PiPER 仿真与控制底座均为外部依赖。**

## 包含哪些功能

- 官方 Core 的任务、工具、验证与经验流程，以及部署中实际使用的本地修复。
- 命令对话、当前页模型切换、三路观察、调用记录、停止、录像展示与原有七项管理入口。
- `piper-dual-sim` 仿真 Skill，以及 `piper-real-runtime` 真机接口适配。
- 专家程序运行、数据转换与 ACT 实验辅助脚本。历史实验的程序、场景、数据和模型不在本包中，见 [依赖说明](docs/DEPENDENCIES.md)。

代码不包含密钥、会话数据库、录像、数据集、模型、共享仿真资产或环境。录像页在未配置外部目录时没有视频文件。工具调用成功不等于物体任务完成。

## 安装与启动控制台

以下在 Linux、Python >= 3.11、Node.js 22 环境执行。为本框架使用具名项目环境；不要向采集/控制环境安装这些依赖。已有对应环境时直接复用，不重复创建。

```bash
cd /path/to/OpenArmSim/松灵机械臂/phyagentos
# 仅首次安装且没有可复用的项目环境时执行：
python3.11 -m venv .venv
.venv/bin/python -m pip install -e './PhyAgentOS-core[dev]' -r requirements-console.txt
bash scripts/build_console.sh

cp -n config/local.env.example config/local.env
mkdir -p .runtime
cp -n config/agent.example.json .runtime/config.json
cp -n config/real_runtime.example.json config/real_runtime.json
```

编辑 `.runtime/config.json`，填写实际模型名、提供商地址、Key 和本副本工作区的绝对路径；按需在 `config/local.env` 中设置外部依赖路径。上述文件均被 Git 忽略。使用已有环境时设置 `PAOS_PYTHON`。`PAOS_CONFIG` 默认指向本副本 `.runtime/config.json`，控制台和官方 CLI 使用同一配置，避免覆盖另一实例的数据。

```bash
source scripts/env.sh
bash scripts/run_console.sh
```

默认访问 `http://127.0.0.1:19081/`；可设置 `PAOS_CONSOLE_PORT` 使用其他空闲端口。启动控制台不会完成机器人接入，不能与另一个占用相同 Skill Gateway/RPC 端口的实例同时运行。外部模型调用需要可用模型和凭据；三路图像分析需要支持图像的模型。

## 安装 Skill

先配置对应外部依赖，并确保 Dora 0.4.1 可从 PATH 找到。Gateway 1.1.1 由官方制品解析器获取，制品哈希保留在 skill.yaml 中；本仓库不附带二进制。

```bash
source scripts/env.sh
# 每个归档顶层直接包含 skill.yaml、SKILL.md、profiles/。
# 归档是生成物，不提交到 Git。
tar -czf bundles/piper-dual-sim.tar.gz -C bundles/piper-dual-sim skill.yaml SKILL.md profiles archive-manifest.json
paos skill install bundles/piper-dual-sim.tar.gz --local --yes
# 完成共享仿真资产及其环境配置后，才手动启动：
# paos skill start piper-dual-sim --profile joint

# 真机 Skill 使用同样的打包/安装方式；安装不授权运动。
tar -czf bundles/piper-real-runtime.tar.gz -C bundles/piper-real-runtime skill.yaml SKILL.md profiles archive-manifest.json
paos skill install bundles/piper-real-runtime.tar.gz --local --yes
```

真机配置默认 `offline`、`release=null`。现有动作仍经原生控制链路、任务绑定和单次放行执行；本包不提供一个新的控制底座。移到其他机器人后需要单独适配，不能直接复用 PiPER 的关节或相机约定。

## 当前能力边界

System 1 指通用 LLM/VLM 控制，System 2 指动作模型控制。System 1 的观察与相对动作有历史实测；近期物体任务由 Codex 逐段规划，不等于前端模型已独立自主完成整项任务。System 2 和混合模式尚未在当前四 CAN 接口验收。ACT 的历史离线推理不等于本接口支持策略执行。

历史抓瓶、双臂交接和入桶有成功记录；毛巾只完成第一折尝试，完整叠毛巾未完成。这里不分发这些历史录像或运行证据，也不将它们记为本次发布测试。

## 检查与来源

[本次检查](docs/VALIDATION.md) · [版本与修改](docs/PROVENANCE.md) · [外部依赖与历史脚本](docs/DEPENDENCIES.md)

Core 保留原 MIT 许可证；Forge 和 RPC 文件保留原 Apache-2.0 许可证与 NOTICE。目标仓库其他代码的许可不因本次导入发生变化。
