# 本次导出验证（2026-11-05）

- 选定离线回归：51 项通过，另 18 项 subtests 通过。覆盖控制台边界与模型切换、路径边界、配置隔离，以及现有 Forge 准入、工作流、状态轮询、验证诊断修复。
- 干净副本没有录像，2 项录像集成测试明确跳过；另将 PAOS_ARCHIVE_ROOT 指向原有外部录像目录，以只读方式运行这 2 项检查，全部通过，包括真实视频 HTTP Range 和路径越界拒绝。没有复制录像进仓库。
- 前端 TypeScript / Vite 构建通过；使用与 lockfile 相符的原有依赖树，没有升级依赖。node_modules 和 dist 不上传。
- 两个 Skill 通过官方 ArchiveValidator 的文件哈希/大小校验及 SkillManifest 解析。第一次验证使用的临时目录名不符合官方“目录名必须匹配 Skill 名称”的要求；修正验证目录后通过，未放宽解析器。
- 配置缺失时，启动脚本在启动 Uvicorn 前返回明确错误；未用真实 Key 启动另一个控制台。
- Python AST 与 Shell 语法检查通过；未在本次导出中启动机器人、相机、仿真、训练或模型调用。

复跑选定检查（项目根目录，已配置开发环境）：

```bash
source scripts/env.sh
python -m pytest -q -rs tests   PhyAgentOS-core/tests/test_forge_action_admission.py   PhyAgentOS-core/tests/test_forge_workflow_overlay.py   PhyAgentOS-core/tests/test_skill_runtime_status_poll.py   PhyAgentOS-core/tests/test_verification_diagnostics_transport.py   PhyAgentOS-core/tests/test_verification_engine_diagnostics.py
bash scripts/build_console.sh
```

检查借用了既有项目的解释器与前端依赖；没有验证全新机器从零安装，也没有验证目标 OpenArm 接口、外部模型服务、全部历史脚本或真机任务。原部署代码未因本次导出而更改。完整原始测试日志留在上传审计目录，仓库只保存这份不含私有运行信息的摘要。

`git diff --check` 报告上游快照中 5 个文件已有的空白格式问题；逐文件确认与冻结上游基线字节一致，保留原样，没有将其记为零告警。
