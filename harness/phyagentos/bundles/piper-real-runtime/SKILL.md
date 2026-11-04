# Current four-CAN connection (0.2.3+openarmsim.1)

The connection now selects native-four-can-replay/v1, offline by default. The old UnifiedRuntime task sender uses the former can_left/can_right follower mapping, which conflicts with the current registered teaching-arm roles. Do not use that old hardware path for the current robot.

All execution on this route reuses robot-control tools/replay_web.py Worker -> managed replay_supervisor.py -> replay_handoff.py. The original owner holds the shared Runtime lock, resolves all four registered USB identities, performs role/enable/alignment handling, receives T/R/P/Q and saves its evidence. No CAN/ROS sender or new control page is implemented here. A worker death or uncertain start is never redispatched.

Use capabilities first. The current supported task type is four_can_replay. The operator first registers the exact native trajectory through runtime_bridge.py prepare-replay ID /absolute/server/trajectory.json. This runs the original native --preview validator and freezes the bytes. prepare_task accepts program={"registered_replay_id":"ID"}; raw joint/TCP targets are not accepted from the Agent. This is a reference to a reviewed native expert program, not permission to execute it.

run_task still requires the exact prepared artifact SHA256 and a fresh, one-use onsite operator release in config/real_runtime.json. The Agent must not create the release or operator confirmation. Until the operator is onsite, keep mode=offline and release=null. This route does not expose TCP stepping, station programs, policies, model debugging or camera control. Existing native control-console facilities remain responsible for those workflows.

Worker state is process/task status. It is not a fresh robot joint observation; no cached values are promoted to fresh feedback. Native cursor completion is only program completion. business_success/physical_success and physical_stop stay unknown without actual evidence. Cancellation uses the same native owner and its Q/SIGTERM cleanup; force-termination or missing final evidence is reported as unknown. A vanished owner without a final receipt must be reconciled before another dispatch.

For an explicitly authorized bounded relative commissioning test, a registered native trajectory may use execution_mode=follower-relative-j1. The native task captures independent follower encoder feedback at execution start; right J1 offsets stay within 0.5 degrees and 0.25 degrees/s, return to that same origin, and no teaching-arm or gripper command is sent. Both followers must already be enabled and on CAN control. This is a raw joint-relative validation, not calibrated TCP control. Four-arm replay remains the default for existing trajectories. Live Forge preflight now dispatches to this same native transport instead of rejecting it through the legacy UDS branch.

For the recorded bottle task, `follower-relative-bottle` reuses that same follower-only path with both grippers. Prepare the complete `pick&placev2` capture using the native `prepare_capture_replay.py --relative-bottle` command. It retains all recorded joint increments and native gripper widths/forces, retimes to at most 10 degrees/s and 15 mm/s, and stays within 100 seconds. The current physical start pose must match the demonstrated start before a human-confirmed `start_reference` is bound to fresh follower feedback. The initial gate checks both followers before any target/mode/gripper send; a missing reference or more than 0.5 degrees of intervening motion rejects. It performs no teaching-arm alignment. Relative control does not compensate for a different physical start or moved objects. Completion of this program still does not establish grasp, handover or placement success; retain actual camera evidence.

## Legacy UDS contracts (preserved; not the current four-CAN hardware adapter)

# PiPER Runtime programs and basic actions

This task Skill uses an existing operator-owned robot-control Runtime through runtime-task/v1. It never creates a CAN/ROS executor, starts hardware or cameras, initializes arms, attaches adapters, changes field configurations, or grants field approval. ACT/DP and towel adaptation remain paused.

1. Query piper_real.capabilities. Offline results describe software only. Query piper_real.state for the Runtime's joints, gripper and TCP with source timestamps. piper_real.observe additionally requires the existing three-camera observation service; cameras are not required for state or program/basic action dispatch.
2. Use piper_real.prepare_task to write an immutable local candidate. It does no native I/O and field_approved remains false. Keep the returned prepared_id and SHA256; never overwrite an existing candidate.
3. A human operator must already have enabled the appropriate existing Runtime executor and released the exact artifact in the local connection configuration. Use piper_real.run_task only for that prepared_id. Do not create field releases or set field_verified/field_review_id from model reasoning. Field approval is distinct from offline validation.
4. Track the Forge Action and piper_real.recover using its receipt_reference. Pending manual checkpoints are included in status; only the field operator answers through the native task-bound confirmation channel. An operator-only runtime_bridge.py confirm CLI uses the original receipt/client binding. No confirmation-answer tool is exposed to the Agent.
5. Use the Forge Action cancel operation for a running action. piper_real.cancel_task can request owned cancellation by the original receipt after a provider restart. Recovery and cancellation preserve the original Runtime instance and client session. Never resubmit an uncertain task or cancel another client's task. Native terminal status plus execution_rights=available is required before another dispatch. Cancellation acknowledgement is not physical stop; the operator owns emergency stop.

## Native program contracts

- single_arm_program uses independent-pick-place/v1: request, arm and steps. Each step is {skill,args}; permitted skills are observe, move_to_station, set_gripper, return_home, confirm_object and wait. Station names are home, pick_approach, pick_contact, pick_lifted, place_approach, place_release and retreat. Step args are: observe {}, move_to_station {arm,station}, set_gripper {arm,mode:open|grasp}, return_home {arm,profile:station_only|complete}, confirm_object {arm,checkpoint:after_lift|after_release|after_retreat}, wait {duration_s:0..10}. Station coordinates, gripper widths, transitions and parked-arm conditions come from the operator's native configuration. No raw Cartesian/joint targets are accepted. Native field_review_id and field_verified must already exist in the operator release. Human checkpoints remain task/step/arm/checkpoint-bound. The bridge waits within the native 120-second program budget rather than treating the checker's 0.1-second checkpoint estimate as a human response deadline.
- bimanual_program uses bimanual-demo/v1 and its native validator: registered center/pose_a/pose_b, both-gripper open/closed, observe/wait and demo_home. An unconfigured native executor must reject.

## Basic action contract

Use task_type=visual_servo_step and a single step object:

- move_right_tcp_relative: dx_m, dy_m, dz_m, optional frame=right_base, step_level_m, timeout_s and poll_s. Native translation norm maximum is 0.010 m; orientation stays with the Runtime. Camera/world coordinates and execution-only fields are rejected.
- set_gripper: side=right, position_m and unit=m. Native hardware bounds remain authoritative.
- bounded_joint_step: candidate_id=right_joint1_plus/minus through right_joint6_plus/minus. Only advertise/use this when the native capability is explicitly enabled. The Runtime sets step size/limits; the bridge reads fresh native right-arm feedback immediately before submission and supplies expected_q_rad/expected_feedback_timestamp. Never invent or prefill those readings in a candidate.

Native completion, callbacks and protocol fixtures do not prove grasp or placement. Preserve task results, timestamps, pending confirmations, result_refs and release_evidence. business_success and physical_success remain null without real evidence. An unknown physical stop remains unknown.

Server instructions and current field blockers: ../../docs/DEPENDENCIES.md (host-specific preparation and field records are not distributed)

The Forge invocation deadline is 150 seconds to cover the native 120-second single-arm budget plus status/cancellation handling; this does not extend native hardware limits.
