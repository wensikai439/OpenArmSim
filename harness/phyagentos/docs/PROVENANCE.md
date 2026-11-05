# 版本与修改

来源索引为 [SOURCE.json](../SOURCE.json)。Core 来自 PhyAgentOS/PhyAgentOS-core 的 1.1.0 基线，提交 `99314a454d1c64cbff5c938d6f121276c10a4dc2`。部署目录不是 Git checkout，使用已验证归档与冻结基线进行文件比对。

导入前已有 17 个修改文件与 5 个新增测试，涵盖 Skill 激活/任务冻结绑定、Forge 工作流 overlay、状态轮询、验证诊断、自定义提供商和经验反思修复。本次导出再为 config/loader.py 增加可选 PAOS_CONFIG，CLI 显式配置仍优先。未把这些修改宣称为上游发布内容。

Skill 导出版本添加 `+openarmsim.1`，区分与原部署版本的路径适配。Gateway 1.1.1 与 Dora 0.4.1 保持；Forge 供应商文件、许可证和 NOTICE 保留。源代码与原部署一致性清单由上传审计保存，本仓库仅记录可发布的信息。

导出修改只发生在独立副本：替换主机路径、增加配置示例、使录像根目录可配置、相机服务显式继承必要路径、修正包内校验清单、标记缺少录像的集成检查。现有页面功能与原生动作约束保留；没有在原部署实施这些更改。

## Core 与上游基线差异

- `PhyAgentOS/agent/experience/activation.py`：modified
- `PhyAgentOS/agent/experience/analyzer.py`：modified
- `PhyAgentOS/agent/experience/coordinator.py`：modified
- `PhyAgentOS/agent/experience/evolution.py`：modified
- `PhyAgentOS/agent/session_verifier.py`：modified
- `PhyAgentOS/agent/tools/agent.py`：modified
- `PhyAgentOS/agent/tools/forge_tool_api.py`：modified
- `PhyAgentOS/cli/commands.py`：modified
- `PhyAgentOS/config/loader.py`：modified
- `PhyAgentOS/forge/task.py`：modified
- `PhyAgentOS/forge/tool_client.py`：modified
- `PhyAgentOS/providers/custom_provider.py`：modified
- `PhyAgentOS/skill_runtime/integration.py`：modified
- `PhyAgentOS/skill_runtime/manager.py`：modified
- `PhyAgentOS/verification/engine.py`：modified
- `PhyAgentOS/verification/request_builder.py`：modified
- `PhyAgentOS/verification/service.py`：modified
- `tests/test_agent_evolution.py`：modified
- `tests/test_forge_action_admission.py`：added
- `tests/test_forge_workflow_overlay.py`：added
- `tests/test_skill_runtime_status_poll.py`：added
- `tests/test_verification_diagnostics_transport.py`：added
- `tests/test_verification_engine_diagnostics.py`：added

第三方许可见 Core/LICENSE 及各 Skill profiles 目录中的 LICENSE/NOTICE 文件。此次不改变目标仓库其他部分的许可。
