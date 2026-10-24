# OpenArm Harness

真机接入包：API agent、冻结瓶身技能、NVIDIA ASPIRE 官方执行器、测量与控制适配代码。公司电脑不包含仿真平台、MuJoCo、网页、模型权重或数据集。

```text
LLMHarness/
├── run          唯一命令入口
├── README.md
├── AGENTS.md    远程文件与设备操作范围
├── app/         agent、服务、测量/控制适配、skill、官方执行器
├── config/      robot.json、hardware.json、calibration.json
└── records/     接入记录、版本备份、校验清单、每次操作日志
```

私有 API 保存在 `.private/api.json`，权限600；环境和临时文件位于 `.venv/`、`.runtime/`。默认没有后台服务、定时任务或设备连接。

## 现在可用

```sh
./run status
./run verify
./run preflight
./run service
```

`service` 只打开本机8772端口的无设备检查服务，Ctrl+C结束；不会打开相机、CAN或电机。它与agent的状态协议已经对接，但会如实报告物理接口未验收。`preflight`只读检查原部署源码哈希、同一ACT初始目标和接入证据。

API保持原地址/模型/密钥，无需重新填写。`./run chat "检查准备情况，先不要运动"`可让同一模型查询接入情况；`./run api-test`仅测API。**当前不能直接抓瓶**：缺少实测相机外参、夹持阈值以及经实机验收的14工具运动提供者，`--execute`仍被阻止。

## 已准备的链路

- `app/devices.py`：独立、延迟打开的RGB-D读取器，不导入旧相机模块的设备枚举。原OpenArm控制器、CAN反馈和初始化源码按哈希只读引用。初始化复用ACT第71条数据目标、400步插值和5秒目标保持，增加实际到位检查与逐指令记录。本轮没有执行初始化。
- `app/evidence.py`：URDF具名关节FK、物体相对位姿变换、编码器/力矩/物体在夹爪内位置/持续稳定窗口联合判定；抓取区域包含夹垫完整几何范围及定位误差。闭爪不等于持瓶。
- `app/devices.py:marker_poses`：可选注册标记RGB-D定位基线，使用ArUco、PnP重投影和对齐深度复核。**不假设现有瓶子上有标记**。实际标记尺寸、相机外参仍未配置。可更换感知前端，但实测坐标系、误差和时间戳契约必须相同。
- `app/motion.py`：双臂30Hz持续保持，LLM/API等待不停止另一只手的指令。每个轨迹点等待伺服回执，不跳点追赶；超时、关节跟踪异常即锁定停止。执行前需要受检规划器复核双臂完整路径和携物碰撞。**实际规划器尚未接入验收，没有直线插值冒充避障的后备路线。**
- `app/service.py`：状态/新鲜观测/工具回执接口、会话鉴权、单任务占用、重复请求拒绝和未知执行结果停止。没有自动加载未经验收的运动提供者。

这是一组可测试的适配代码和无设备服务，不能写成“三维定位和14个真实工具已经联调成功”。旧VR手柄标定不是相机外参，回位时间结束也不是已到位。`config/calibration.json`的未知项保留为null。修改控制配置后必须重新验收并更新发布清单，不能通过删掉校验绕过。

## 记录和后续验收

每次任务独立写入 `records/时间-类型-编号/`：`session.json`记录任务与版本，`events.jsonl`记录Unix/单调时间、API工具、观测、请求、驱动返回、关节反馈、故障和停止。运动请求在下发前写入并刷盘，日志有连续哈希链；`./run audit-log records/具体目录`检查完整性。未来获准的采集会话还需记录录像及其时钟映射；本轮没有拍摄或开启设备。

下一次现场依次完成：定位/时间同步→夹爪空抓与夹瓶阈值→同ACT初始位置实际到位→空手路径→右手下半瓶身抓取与保持→交接→左手保持与入篮。每段有独立实测回执后，才开放整段API任务。详见 [试运行](app/试运行.md)。所有文件、缓存和日志只能写在本目录内。

来源：[NVIDIA ASPIRE](https://github.com/NVlabs/ASPIRE)，固定提交 `fa00f0ef2bc1d91d594b9aa502c22133c9f538f7`；测量接口参考 [RealSense对齐示例](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py) 和 [OpenCV标记文档](https://docs.opencv.org/5.0/tutorials/objdetect/aruco_detection/aruco_detection.html)。OpenArm适配器由本项目实现，未宣称NVIDIA官方真机驱动。
