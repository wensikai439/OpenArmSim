"""Local deployment-contract preflight; never contacts or commands hardware."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def inspect(config_path):
    config_path=Path(config_path).resolve()
    config=json.loads(config_path.read_text())
    folder=ROOT/'skills/bottle_relay'
    manifest=json.loads((folder/'manifest.json').read_text())
    actual=hashlib.sha256((folder/'program.py').read_bytes()).hexdigest()
    if actual!=manifest['program_sha256']:raise ValueError('admitted skill hash mismatch')
    if config.get('schema')!='openarm-real-contract/1':raise ValueError('unknown real adapter contract')
    gaps=[]
    if config.get('hardware_enabled') is not False:gaps.append('交接配置必须默认禁用真机运动')
    adapter=config.get('adapter',{})
    if not adapter.get('implementation'):gaps.append('尚无真实 OpenArm 工具适配器实现')
    if adapter.get('tested_skill_sha256')!=actual:gaps.append('真机适配器尚未与当前 skill 哈希配对验收')
    if config.get('observation',{}).get('source')!='measured_perception':
        gaps.append('真实相机位姿与夹持检测尚未接入；不能使用仿真真值')
    expected=[f'{arm}_joint{i}' for arm in ('left','right') for i in range(1,8)]
    if config.get('joint_names')!=expected:gaps.append('14个机械臂关节的顺序尚未确认')
    for arm in ('left','right'):
        mapping=config.get('grippers',{}).get(arm,{})
        if any(type(mapping.get(k)) not in (float,int) for k in ('open_command','close_command','command_min','command_max')):
            gaps.append(arm+'夹爪命令/反馈映射缺失')
    if config.get('observation',{}).get('frame')!='robot_base':gaps.append('需明确机器人基座坐标系')
    if config.get('control',{}).get('watchdog_verified') is not True:gaps.append('反馈超时、急停及控制器看门狗未验收')
    verified=[]
    for name in ('kinematics','cameras','controller','bottle_basket','paired_trials'):
        item=config.get('evidence',{}).get(name,{})
        relative=item.get('path');digest=item.get('sha256')
        if not relative or not digest:
            gaps.append(name+'证据未提供');continue
        path=(config_path.parent/relative).resolve()
        if not path.is_relative_to(config_path.parent) or not path.is_file():
            gaps.append(name+'证据路径无效');continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            gaps.append(name+'证据哈希不符');continue
        verified.append(name)
    return {'schema':'openarm-preflight/1','skill_sha256':actual,
        'contract_complete':not gaps,'hardware_enabled':False,'hardware_access':False,
        'transfer_ready':False,'missing':gaps,'evidence_files_verified':verified,
        'scope':'only local hashes and declared interface completeness; cannot certify physical calibration or execute a real robot'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'skills/bottle_relay/robot.example.json')
    args=p.parse_args(argv);result=inspect(args.config)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['contract_complete'] else 2


if __name__=='__main__':raise SystemExit(main())
