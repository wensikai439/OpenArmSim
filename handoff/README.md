# 真机接入包 v4

当前使用 `OpenArmRobot-20261024-v4.zip` 和同名 `.sha256`。包内源码约47KB，来源提交 `5d41e3a`。SHA256：`423aa4153b23f9b8376d3ac738e153deb13ac7a85c85a9fc6e40f600e9ca3c65`。

公司只在 `/home/sj/Desktop/wuji/lerobot_20260116/LLMHarness/` 中部署API agent、技能程序/契约和官方ASPIRE执行器。包不含仿真、模型资产、权重、密钥或虚拟环境。其他公司目录只读，原ACT、DP、π0.5不动。真实感知和动作工具服务仍缺，不能把部署完成当成已能真机抓瓶。

纯真机v4已更新，原API配置保留，17项包完整性与175项原部署文件检查通过，实际运动为0。回执在公司端 `records/receipt-v4.json`，本地 `runs/20261024-robust/deployment.json`。

本地本轮完成159次原生实验、1次默认CLI整任务和381项平台回归。历史32条件回归由30/32修复为32/32；另8个新位置×2条件为16/16，同组对照也是16/16，不能称总体留出成功率已提高。数据与可视化见 `runs/20261024-robust/`。

使用说明见 `docs/真机部署.md`，客户端契约见 `deploy/ROBOT.md`，首次真机准备见 `deploy/试运行.md`。重打包命令为 `.venv/bin/python -B -m tools.package_robot`；已有文件不覆盖。ASPIRE固定版本和逐文件校验由包内 `third_party/aspire/ORIGIN.json`、`MANIFEST.json` 记录。

旧仿真交接ZIP保留作本地历史快照；公司端完整仿真部署已撤回并清理，禁止再上传。旧纯真机v3源码备份在公司端 `records/versions/v3.zip`，不含API密钥。
