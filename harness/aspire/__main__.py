"""Short local entry point: ./run aspire test --x .50 --y -.17."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description='OpenArm 抓瓶技能与 ASPIRE 实验')
    commands=parser.add_subparsers(dest='command',required=True)
    configure=commands.add_parser('configure',help='单独保存实验 API，不覆盖主页面配置')
    configure.add_argument('--base-url',required=True)
    configure.add_argument('--model',required=True)
    train=commands.add_parser('train',help='新建有预算限制的 ASPIRE 技能发现实验')
    train.add_argument('--output',default='runs/'+time.strftime('%Y%m%d-%H%M%S')+'-aspire')
    test=commands.add_parser('test',help='独立仿真测试已验证的抓瓶交接技能，不连接真机')
    test.add_argument('--x',type=float,default=.503)
    test.add_argument('--y',type=float,default=-.173)
    test.add_argument('--seed',type=int,default=0)
    recover=commands.add_parser('recover',help='独立仿真运行反馈恢复技能；不访问真机、不调用 API')
    recover.add_argument('--x',type=float,default=.50)
    recover.add_argument('--y',type=float,default=-.21)
    recover.add_argument('--seed',type=int,default=0)
    recover.add_argument('--fault',choices=('none','right_close_loss','receiver_close_loss','stale_grasp'),default='none')
    recover.add_argument('--pose-bias',type=float,default=.05,help='诊断用感知高度偏差，.04 至 .07 米')
    prepare=commands.add_parser('prepare',help='只检查本地真机接口准备情况，不连接或移动真机')
    prepare.add_argument('--config',type=Path,default=ROOT/'skills/bottle_relay/robot.example.json')
    package=commands.add_parser('package',help='生成不含密钥和权重的仿真/技能交接包')
    package.add_argument('--output',type=Path,default=ROOT/'handoff/OpenArmSkill-20261024-v3.zip')
    args=parser.parse_args()
    if args.command=='prepare':
        from tools.prepare_robot import main as prepare_robot
        return prepare_robot(['--config',str(args.config)])
    if args.command=='package':
        from tools.package_skill import main as package_skill
        return package_skill(['--output',str(args.output)])
    if args.command=='configure':
        from harness.llm import ModelSettings
        from harness.vault import SecretStore
        settings=ModelSettings(store=SecretStore(namespace='openarm-aspire'))
        settings.configure({'base_url':args.base_url,'model':args.model,'api_key':getpass.getpass('API Key：'),
                            'vision':True})
        print('实验 API 已保存到系统密钥环。')
        return 0
    if args.command=='train':
        from .campaign import run
        output=(ROOT/args.output).resolve()
        if not output.is_relative_to(ROOT/'runs'):parser.error('实验输出必须放在 runs 内')
        run(output)
        return 0
    if args.command=='recover':
        folder=ROOT/'skills/bottle_recovery';program=folder/'program.py'
        if not (folder/'manifest.json').exists():parser.error('恢复技能尚未验收')
        manifest=json.loads((folder/'manifest.json').read_text())
        if not manifest.get('admitted') or hashlib.sha256(program.read_bytes()).hexdigest()!=manifest['program_sha256']:
            parser.error('技能未验收或程序已改变；请重新验证')
        for relative,expected in manifest.get('runtime_sources',{}).items():
            source=(ROOT/relative).resolve()
            if (not source.is_relative_to(ROOT) or not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest()!=expected):
                parser.error('技能运行时与验收版本不同：'+relative)
        from .recovery_campaign import run_trial
        case={'seed':args.seed,'bottle_xy_m':[args.x,args.y],'fault':args.fault,'pose_bias_m':args.pose_bias}
        output=ROOT/'runs'/(time.strftime('%Y%m%d-%H%M%S')+'-bottle-recovery')
        world_type=None
        if manifest.get('runtime_backend')=='placement':
            from .placement import PlacementWorld
            world_type=PlacementWorld;case.update(motion_profile='precision',grasp_region='body')
        elif manifest.get('runtime_backend')=='body':
            from .body import BodyWorld
            world_type=BodyWorld;case.update(motion_profile='precision',grasp_region='body')
        elif manifest.get('motion_profile')=='precision':
            from .precision import PrecisionWorld
            world_type=PrecisionWorld;case['motion_profile']='precision'
        result=run_trial(case,output,program,'frozen',world_type=world_type)
        print('完整记录：'+str(output))
        return 0 if result['task_success'] else 1
    folder=ROOT/'skills/bottle_relay'
    program=folder/'program.py'
    if not program.exists() or not (folder/'manifest.json').exists():
        parser.error('抓瓶技能尚未通过复验并入库，请先查看实验结果')
    manifest=json.loads((folder/'manifest.json').read_text())
    if hashlib.sha256(program.read_bytes()).hexdigest()!=manifest['program_sha256']:
        parser.error('技能程序已改变，原验收结果不再适用；请重新验证')
    output=ROOT/'runs'/(time.strftime('%Y%m%d-%H%M%S')+'-bottle-skill')
    from .runtime import run,validate_case
    case=validate_case({'seed':args.seed,'bottle_xy_m':[args.x,args.y]})
    result=run(program,case,output)
    print('完整记录：'+str(output))
    return result


if __name__=='__main__':raise SystemExit(main())
