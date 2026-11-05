---
name: piper-dual-sim
description: 独立共享 PiPER 双臂仿真；真实 Forge Query / Action / Session、关节专家记录与接触证据。
---
# 共享 PiPER 双臂仿真

本地适配 Skill 0.4.2+openarmsim.1，Core 基线 1.1.0 / Gateway 1.1.1 / Dora 0.4.1。主要物理资产由 `PIPER_SIM_ROOT` 指定， 是独立共享仿真，RPent和PhyAgentOS分别消费。本 Skill 自带仅服务本实例的 RPC 适配，不导入 RPent 工程或包。profile=osc 沿用原 OSC_POSE；profile=joint 在 PhyAgentOS 自有子进程使用官方 JointPositionController（absolute rad），专家位姿经有界 IK 生成实际关节目标并保存逐控制步记录。共享源码保持只读。joint 通过已选择场景加载可追溯派生资产；piper_dual_wrist_top_30hz_v1 将两腕相机移至法兰上侧及近端、实际控制/记录30Hz（100×1/3000s）。其余机器人、执行器、碰撞/接触、辅助及原生任务阈值保持；观察 control_hz/physics_dt_s 与 scene_initialization 核实当前版本。

## 当前任务夹爪模式

joint profile采用二值任务命令OPEN/CLOSE：OPEN为100mm；CLOSE朝0.1mm数值闭合端点每控制步最多推进2mm，遇真实双指瓶接触后保持当步servo目标。夹住物体的实测开度与全合任务意图分别记录，不要求穿透物体达到零开度；不接受任意中间开度作为任务命令。未抓住物体的set_gripper close可达到空夹爪端点，grasp_bottle仍必须证明双指接触才完成。共享模型的每指±10N执行器力限、碰撞和原10mm安全保护保持；接触辅助保持原生明确标签，不能宣称无辅助抓取。

observe的gripper_command_mode= binary_endpoints；binary_gripper_commands分别提供task_mode（0全合/1全开）、native_command、servo_width_target_m及measured_opening_m。binary_gripper_contract_path指向该episode合同；joint_commands.jsonl每帧gripper_task_modes是实际任务意图，command_native/ctrl是实际消费目标，不应相互冒充。profile=osc保留旧原生接触关闭模式。实际3586步完整原生交接通过，官方语义验收须另看对应Task enforce/verdict；未使用mock或修改数据库。

## 观测与持续 Session

`piper.observe` {} 取得原生 cam_right、overview 两相机、实例/episode/帧绑定、左右实测 tcp_world_m / tcp_rotation_rad、physics_step、sim_time_s、原生任务状态与接触信息。
`joint_positions_rad` 按 `joint_names` 显式顺序读取 MuJoCo data.qpos 和原生 ws.qindexes；`gripper_feedback.finger_joint_qpos_m` 是命名 slide 关节的米值，`joint_opening_m` 是两手指关节位移绝对值之和，不是从图像推算的夹爪间隙。`gripper_command` 仍仅为指令。
Planner 等待时物理暂停；新图像接收不意味着物理推进。

`piper.observe_stream` {interval_ms?:500..5000,max_duration_s?:1..60} 是真正的 Forge Session：持续捕获原生双相机和状态，记录独立 invocation 的 JSONL manifest。默认 500ms、30s；须通过 forge_tool_start_session / session_status / session_result / stop_session 使用同一 invocation。停止后 result 提供 samples、manifest_path、manifest_sha256 和首末实测状态/末帧引用。Session 只采集，不推进物理；原生同步动作执行期间采样暂停，不宣称实时控制频率。停止此 Session 不停止动作。必须停止或等其真实终态后再 finalize task。

## 动作前只读路径检查

`piper.check_pose` {side:"left"|"right",target_world_m:[x,y,z],target_rotation_rad?:[rx,ry,rz]} 是 Query，与单次 move_arm 使用同样的参数和当前真实起点。它调用原生 check_pose_path 的关节限位 IK/FK 路径检查，在临时 MuJoCo Data 中计算；回执包含 verification.accepted、拒绝原因、实际起点与状态SHA，并核对 live qpos/qvel、时间与控制步数均未改变。Query 的 succeeded 仅表示读取完成；必须检查 verification.accepted。

复杂抓取/交接的每个新移动候选先 Query 检查，只有 accepted=true 才提交相同参数的 move_arm。accepted=false 时没有提交动作，可根据拒绝原因修改姿态/位置或缩短该段后重新检查；suggested_sample_pose 与 observe 的 grasp/staging priors 均只是假设候选，仍须检查。不得将被拒目标直接当动作试探。一次查询只检查当前起点到该目标，执行改变起点后要重新检查下一段；不保证实际动力学、夹爪/瓶接触或业务成功，动作后的原生 verified 与最终真实证据仍为验收依据。

本 profile 将官方既有 tools.retention_sec 设为3600秒（原默认300），保留动作终态供长耗时模型轮询；仍须及时读取同一invocation的终态与result，不能依赖保留时长重放未知效果。

工具完整列表为 observe/check_pose 两个 Query、下列六种 Action 与 observe_stream Session。相机帧由 observe/官方证据采集提供，不存在额外的图像读取/显示 Forge 工具；不得猜测工具名。瓶子水平侧抓应基于当前 tabletop_approach 的清空高度、姿态过渡和侧向接近几何分段推导，不能把静态 pose prior 当作从初态直达的轨迹。

## 六种原生独立动作

- `piper.set_gripper` {side:"left"|"right",open_or_close:"open"|"close"}。
- `piper.move_arm` {side:"left"|"right",target_world_m:[x,y,z],target_rotation_rad?:[rx,ry,rz]}。世界系米/rad；省略 rotation 保持当前姿态。最大单步位移 0.20m；原生 workspace、碰撞及 pose_preflight 拒绝保留真实失败，不改 qpos 或碰撞制造通过。先用最新 observe 目标几何和姿态，不能从旧 Task 复用绝对坐标。
- `piper.grasp_bottle` {side}：当前姿态闭爪；不自动移动或旋转。自由直立桌面瓶子须先建立水平侧抓姿态。真实双指当前接触是辅助抓持/空中接收的必要条件。
- `piper.release_bottle` {side}：关闭该臂辅助约束、实际张爪并让重力作用；必须检查释放后的真实状态。
- `piper.return_arm_to_initial` {side}：原生限长路径回到本 episode 初态；瓶子不得被持有。
- `piper.place_bottle_at_origin` {side}：已持瓶的一臂搬回本 episode 原位置、物理释放并验收桌面稳定落定；不能把它称为桶内放置成功。

允许按用户任务组合抓持、搬运、空中交接与释放。必须遵循最新 `measured_grasp_candidates`、`tabletop_approach`、`primitive_geometry`、finger_contact_now 以及每次原生 verification。接收臂双指真实接触后闭爪准备接收，供体张爪时原生再次检查接触才转移辅助约束；不得绕过接触或使用固定宏伪造任务成功。禁止把六原语与原生固定 stage 宏混用；独立动作开始后固定 stage 不能继续。

## 任务与验收

本轮 activate_skill(name="piper-dual-sim",role="primary") 的 activation_id 创建 Task；读实时 context；所有 Query/Action/Session 绑定 task_id。先 observe 记录初态，逐个动作等待同一 invocation status/result 到已知终态，再 observe 并 finalize。能力执行状态与语义 Task verdict 是独立证据；未知动作效果不重复提交。

成功判据按用户目标定义。`native_result.primitive_completed/verified` 只表示单项原语。原生 task_success、air_handoff_verified、碰撞/接触、瓶子位置/姿态/速度必须保留原值，不能把张爪/小位移当作抓瓶、空中交接或桶内落定成功。原生辅助抓持须明确标注 assisted。照片不能替代毫米级状态测量；用绑定 observe 数值和 native verification。证据 sources=image/cam_right,image/overview，kinds=rgb_image,robot_state。

失败修复仅在状态已知且 episode 未 failed 时，根据原生拒绝理由读取新观测，在同 Task begin_revision 调整尚未完成的计划。原生 failed 或 reset_required=true 的 episode 不能自动继续。经验/M4 必须属于本 Skill 和适用工作流；历史 move-arm-by-ee 单臂经验不适用于本 Skill。

## 合作式取消与边界

Action 的 cancellable=true 表示可请求在原生控制 tick 边界停止；不是立即硬停或暂停。forge_tool_cancel_action 的 accepted 仅为请求入队。必须继续查同一 invocation，只有原生包装提供匹配执行标识、明确取消步数的实际终止回执，才能记录 cancelled。动作完成可能先于请求；此时保留真实成功/失败结果。已取消 episode 保留部分轨迹与原生 failed=true，reset_required=true，不得恢复未完成动作；须正常复位新 episode。

Runtime 复位仅用户显式管理 stop/start，LLM 不自动复位。禁止真机、ROS/CAN/SDK、shell/HTTP 绕过 Forge、改数据库或手动写 qpos。

Session Dora 异步事件sink按官方Gateway v1.1.0 vendored SDK dora.py:336-339回移同一条件：Action与Session均使用response-first事件屏障。仅此条件变更；本项目Python3.11兼容vendor其余不变。

Dora双向Tool输入queue_size=128；官方Dora0.4.1默认每输入仅1，突发status/stop/result必须保留各独立协议消息。

本 Skill 初始工作流指导继承旧适配 Skill 的已晋升 revision 2，作为有来源的迁移基线；旧 Task/Lessons 归属不变，不声明新 Skill 已完成自然晋升或 M4。

<!-- paos:learned-workflow:start -->

## Learned Workflow (revision 2)

### Trigger

Simulator-only dual-arm workflow: grasp a bottle with the donor arm, carry it to an airborne handoff pose, close the receiver gripper to establish multi-contact transfer, release the donor gripper, then settle the bottle into the receiver-side bucket and verify the composite transfer record and final placement metrics.

### Preconditions

1. Primary activated Skill is the live dual-arm simulation Skill; the task explicitly requires grasping, handoff, and placement behavior
2. A fresh activation binds a new task with verification mode set to enforce, and all queries/actions are scoped to that task_id
3. Live capabilities are discovered through forge_tool_context before action selection
4. Robot-state observation, gripper command, Cartesian arm-move, grasp, and release capabilities are available as registered Forge tools
5. Initial per-arm TCP position, rotation, physics_step, and sim_time_s are measured and saved from the same task before any motion
6. The bottle, donor arm, receiver arm, and target bucket are present in the simulator scene and reachable within the Skill's documented workspace bounds

### Workflow

1. Activate the primary dual-arm simulation Skill and use the returned activation binding to create a new independent task with verification mode set to enforce
2. Discover live robot capabilities via forge_tool_context and bind every Query and Action to the created task_id
3. Invoke the robot-state observation capability with empty arguments to read and save per-side tcp_world_m, tcp_rotation_rad, physics_step, and sim_time_s
4. Open both grippers, then command the donor arm through a sequence of Cartesian arm-moves toward the bottle while checking pose after each move
5. Invoke the grasp capability on the donor side and confirm the bottle is held via observation
6. Command the donor arm through a sequence of Cartesian arm-moves to lift and carry the bottle to an airborne handoff pose above the receiver-side bucket, checking pose after each move
7. Command the receiver arm through a sequence of Cartesian arm-moves to a pose where its gripper can contact the bottle, checking pose after each move
8. Close the receiver gripper to establish at least two receiver contact geoms on the bottle while the donor still holds it
9. Invoke the release capability on the donor side and confirm the composable_transfers entry records donor=right, receiver=left, at least two receiver contact geoms, qpos_change=0, table_contact=false, and bottle base z above the required threshold
10. Command the receiver arm through a sequence of Cartesian arm-moves to lower the bottle into the bucket and settle it
11. Invoke the robot-state observation capability again and save final per-side tcp_world_m, tcp_rotation_rad, physics_step, and sim_time_s
12. Before finalizing, cross-check that every primitive explicitly named in the success criteria appears in the task's invocation lineage with primitive_completed=true and verified=true
13. Finalize the task and report task_id, invocation identifiers, transfer record fields, final placement metrics, physics/sim progress, and the native verdict

### Verification Checkpoints

1. Each native primitive reaches a terminal state with primitive_completed=true and verified=true
2. The composable_transfers entry at the donor-gripper open event shows donor=right, receiver=left, at least two receiver contact geoms, qpos_change=0, table_contact=false, and bottle base z above the required threshold
3. Final observation reports bucket_bottom_contact=true, bucket_xy_error_m below the required tolerance, bottle_speed_norm below the required threshold, both assist_active=false, failed=false, collision_stop=null, and numerical_warnings all zero
4. physics_step and sim_time_s both increase relative to the initial observation
5. Both camera evidence streams are present in the before/after observations
6. Every primitive named in the success criteria is present in the invocation lineage with primitive_completed=true and verified=true
7. Evaluate the exact composable_transfers entry and the terminal observe metrics rather than macro-only task_success/air_handoff_verified flags, which may remain false even when the composite contract is met

### Recovery

1. If a native action is rejected or failed, stop and finalize preserving the failure instead of replaying or substituting another path
2. If an invocation status is unknown or not yet terminal, wait for the same invocation's status/result rather than issuing a duplicate command
3. If the transfer record is missing or shows fewer than the required receiver contact geoms, re-observe the live state and adjust the receiver gripper closure or handoff pose before releasing the donor
4. If final placement metrics are unmet, re-observe and continue corrective receiver-arm motion rather than finalizing on per-primitive success alone
5. If measured bucket_xy_error_m exceeds tolerance, do not claim success; re-observe and treat the result as failed or incomplete
6. If required evidence is missing, record the missing evidence and continue only through registered Forge tools; do not edit storage or restart services

### Applicability Boundaries

1. Applies only to the activated simulator dual-arm Skill and the registered Forge tool surface, not to real hardware
2. Applies only to assisted airborne bottle handoff and bucket settling; does not cover unassisted single-arm transfer or non-bottle objects
3. Does not authorize historical single-arm relative-motion workflows or tools for this dual-arm task
4. Execution must stay within registered Forge tools; direct SDK/HTTP/shell execution, database modification, file editing, and service restarts are out of scope
5. Requires the handoff pose, receiver reach, and bucket location to be within the simulator Skill's valid workspace and native primitive capabilities
6. Should not be used when the user explicitly requires a different primary Skill or a non-simulator execution environment

Discover live robot capabilities with `forge_tool_context` and execute only through registered Forge tools. Treat Gateway completion as an execution fact and use the task verification verdict for success.

<!-- paos:learned-workflow:end -->
