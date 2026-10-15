"""Short local commands: ./run check; ./run replay --episode 0."""
import argparse
import json
import unittest
from pathlib import Path

from .environment import OpenArmEnv
from .recording import Recorder
from .replay import ReferenceReplay


def replay(args, *, root=None):
    env = OpenArmEnv(images=False)
    recorder = source = None
    status, error = "reference_exhausted", None
    try:
        env.reset(seed=args.seed, options={"episode": args.episode})
        source = ReferenceReplay(env, args.episode, args.clock)
        recorder = Recorder(env, "replay", record_images=args.images, seed=args.seed, root=root,
                            details={"episode": args.episode, "mode": "commanded-dynamics", "real_time": False,
                                     "replay_clock": source.contract})
        before = recorder.observation(env)
        limit = min(args.steps or source.intervals, source.intervals)
        for index in range(limit):
            action, (_, _, terminal, truncated, info), diagnostic = source.advance()
            after = recorder.observation(env)
            recorder.append(before, action, after, info, terminated=terminal, truncated=truncated,
                            diagnostics=diagnostic)
            before = after
            if terminal or truncated:
                status = "success" if info["evaluation"]["success"] else "invalid" if info["evaluation"]["invalid"] else "truncated"
                break
        if args.steps and limit < source.intervals and status == "reference_exhausted":
            status = "partial_diagnostic"
    except KeyboardInterrupt:
        status = "cancelled"
    except Exception as exc:
        status, error = "error", str(exc)
    finally:
        if recorder:
            metrics = source.metrics() if source else {}
            recorder.finish(status, env.info(), error=error, metrics=metrics)
            print(json.dumps({"run": str(recorder.path), "status": status, "steps": recorder.count,
                              "evaluation": env.evaluation, "error": error}, ensure_ascii=False, indent=2))
        env.close()
    return 1 if error else 0


def main():
    parser = argparse.ArgumentParser(description="OpenArm 本机仿真与实验记录")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="检查物理、接口与记录契约，不连接模型或真机")
    tamp = sub.add_parser('tamp', help='本机 TAMP 基线，使用明确标注的仿真场景信息')
    from .tamp import TampPlanner
    tamp.add_argument('--task', choices=TampPlanner.TASKS, default='pick_bottle')
    sub.add_parser("calibration", help="只读查看标定证据与缺项")
    sub.add_parser("align", help="用已审计的原始图像生成关节投影与动力学三列对照，不调整参数")
    sub.add_parser("wrist", help="腕部相机候选拟合与原生遮挡验证；不应用参数")
    sub.add_parser("gripper", help="历史 v2 模型的夹爪坐标与增益对照，不代表当前 v1 模型")
    sub.add_parser("timing", help="三条记录动作的延迟对照；不调用模型或真机")
    sub.add_parser("calibrate", help="辨识右臂早期响应，用独立 episode 检验；不覆盖参数")
    sub.add_parser("policy-check", help="只读核验本地 OpenArm π0.5 检查点及伴随文件")
    policy_server = sub.add_parser("policy-serve", help="启动仅本机可用的官方 LeRobot π0.5 推理服务")
    policy_server.add_argument("--port", type=int, default=8001)
    policy_server.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    exporter = sub.add_parser("export", help="用官方 LeRobot v3 写入并逐帧读回验证")
    exporter.add_argument("run")
    exporter.add_argument("--purpose", choices=("imitation", "diagnostic"), default="imitation")
    trainer = sub.add_parser("train", help="运行有预算上限的官方 ACT 训练流程")
    trainer.add_argument("dataset")
    trainer.add_argument("--recipe", choices=("act_smoke", "act_baseline"), default="act_smoke")
    trainer.add_argument("--steps", type=int, default=5)
    trainer.add_argument("--seed", type=int, default=42)
    command = sub.add_parser("replay", help="独立于页面回放真机指令")
    command.add_argument("--episode", type=int, choices=(0, 1, 2), default=0)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--steps", type=int, default=0, help="只运行指定控制步数；0 为完整轨迹")
    command.add_argument("--images", action="store_true", help="每步保存三路 PNG 图像，可能较慢")
    command.add_argument("--clock", choices=("recorded", "uniform30"), default="recorded",
                         help="默认使用已审计的记录时间；uniform30 仅用于时间压缩对照")
    args = parser.parse_args()
    if args.command == 'tamp':
        import time
        from .service import SimulationService
        service=SimulationService()
        try:
            service.call('tamp', task=args.task, observation_id=service.snapshot()['observation_id'], realtime=False)
            while service.snapshot().get('job'):
                time.sleep(.1)
            result=service.snapshot()['last_run']
            print(json.dumps(result,ensure_ascii=False,indent=2))
            raise SystemExit(0 if result['status'] in ('primitive_completed','success') else 1)
        finally:
            service.close()
    if args.command == "replay" and args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.command == "policy-check":
        from .pi05_server import audit
        try:
            print(json.dumps(audit(), ensure_ascii=False, indent=2))
        except (OSError, ValueError) as exc:
            print(json.dumps({"status": "unavailable", "error": str(exc)}, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        return
    if args.command == "policy-serve":
        from .learning import run_module
        raise SystemExit(run_module("sim.pi05_server", ["--port", args.port, "--device", args.device]))
    if args.command == "calibration":
        from .calibration import report
        print(json.dumps(report(), ensure_ascii=False, indent=2))
        return
    if args.command == "align":
        from .alignment import build
        build()
        return
    if args.command == "gripper":
        from .gripper_study import build
        build()
        return
    if args.command in {"calibrate", "timing", "wrist"}:
        from .learning import LearningJobs
        jobs = LearningJobs()
        job = jobs.start(args.command)
        print(json.dumps(job, ensure_ascii=False), flush=True)
        try:
            while jobs.thread.is_alive():
                jobs.thread.join(1)
        except KeyboardInterrupt:
            jobs.close()
        result = jobs.status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["status"] == "completed" else 1)
    if args.command == "export":
        from .learning import run_module
        raise SystemExit(run_module("sim.dataset", [args.run, "--purpose", args.purpose]))
    if args.command == "train":
        from .learning import run_module
        raise SystemExit(run_module("sim.train", [args.dataset, "--recipe", args.recipe,
                          "--steps", str(args.steps), "--seed", str(args.seed)]))
    if args.command == "check":
        tests = unittest.defaultTestLoader.loadTestsFromNames(["sim.checks", "harness.checks", "harness.model_checks", "harness.workspace_checks", "harness.stage_checks", "harness.compact_checks", "sim.tamp_checks", "sim.calibration_checks", "sim.policy_checks", "sim.scene_library_checks", "harness.policy_trace_checks", "harness.policy_route_checks", "sim.contact_checks", "sim.timing_checks", "sim.alignment_checks", "sim.wrist_checks", "sim.view_checks", "sim.backend_checks", "sim.generalization_checks", "sim.placement_checks", "sim.controller_checks", "sim.gripper_checks", "harness.skill_checks", "harness.skill_registry_checks", "harness.execution_receipt_checks", "harness.runtime_composition_checks", "harness.settled_hold_checks", "harness.planner_stack_checks", "harness.task_placement_checks", "harness.memory_checks", "harness.knowledge_checks", "harness.goal_checks"])
        tests.addTests(unittest.defaultTestLoader.loadTestsFromName('sim.model_checks'))
        tests.addTests(unittest.defaultTestLoader.loadTestsFromName('sim.relocation_checks'))
        tests.addTests(unittest.defaultTestLoader.loadTestsFromName('sim.grasp_feedback_checks'))
        tests.addTests(unittest.defaultTestLoader.loadTestsFromName('harness.relay_checks'))
        tests.addTests(unittest.defaultTestLoader.loadTestsFromName('tools.platform_checks'))
        raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful())
    raise SystemExit(replay(args))


if __name__ == "__main__":
    main()
