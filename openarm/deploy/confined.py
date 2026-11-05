"""Company-host entry point: directory-local, offline and hardware-disabled.

This launcher prevents accidental use of the application's user-systemd and
global-keyring entry points. It is not a security sandbox for arbitrary code.
"""
import json
import os
from pathlib import Path
import sys

APP=Path(__file__).resolve().parents[1]
ROOT=APP.parent


def environment(root):
    runtime=root/'.runtime'
    locations={
        'TMPDIR':runtime/'tmp','TMP':runtime/'tmp','TEMP':runtime/'tmp',
        'XDG_CACHE_HOME':runtime/'cache','XDG_CONFIG_HOME':runtime/'config',
        'XDG_DATA_HOME':runtime/'data','XDG_STATE_HOME':runtime/'state',
        'PIP_CACHE_DIR':runtime/'cache/pip','UV_CACHE_DIR':runtime/'cache/uv',
        'MPLCONFIGDIR':runtime/'config/matplotlib','TORCH_HOME':runtime/'cache/torch',
        'HF_HOME':runtime/'cache/huggingface','NUMBA_CACHE_DIR':runtime/'cache/numba',
        '__GL_SHADER_DISK_CACHE_PATH':runtime/'cache/nvidia',
        'CUDA_CACHE_PATH':runtime/'cache/cuda','TRITON_CACHE_DIR':runtime/'cache/triton',
        'PYTHONUSERBASE':runtime/'python-user',
    }
    root=root.resolve()
    for key,path in locations.items():
        if not path.resolve().is_relative_to(root):
            raise ValueError('runtime path escapes deployment directory')
        path.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(path)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
                      PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1',
                      MUJOCO_GL=os.environ.get('MUJOCO_GL','egl'))
    # No inherited checkout, provider, model or backend search paths.
    os.environ.pop('PYTHONPATH',None)
    os.environ['OPENARM_ASPIRE_SOURCE']=str(APP/'third_party/aspire')


def command(args):
    if not args or args==['help']:
        return None
    if args==['status']:
        return ['-m','harness.real','preflight']
    if args==['verify']:
        return ['tools/verify_handoff.py']
    if args[0]=='real' and len(args)>1 and args[1] in ('preflight','replay','demo','shadow'):
        return ['-m','harness.real',*args[1:]]
    if args[0]=='aspire' and len(args)>1 and args[1] in ('recover','test'):
        return ['-m','harness.aspire',*args[1:]]
    if args==['check']:
        return ['-m','unittest','harness.real_checks','harness.aspire.runtime_checks',
                'harness.aspire.recovery_checks','tools.package_checks','deploy.confined_checks']
    raise ValueError('该接入部署只开放 verify/status/check、real preflight/replay/demo/shadow、aspire recover/test；未开放真机运动、系统服务或全局 API 密钥配置。')


def main():
    args=sys.argv[1:]
    try:
        selected=command(args)
        record=json.loads((ROOT/'DEPLOYMENT.json').read_text())
        if record.get('root')!=str(ROOT) or record.get('hardware_enabled') is not False:
            raise ValueError('invalid deployment root or motion flag')
        environment(ROOT)
    except (ValueError,OSError) as exc:
        print(str(exc),file=sys.stderr);return 2
    if selected is None:
        print('OpenArm Harness · v3 接入部署（真机运动未开放）\n'
              './run verify              包完整性与仿真结构检查\n'
              './run status              查看接入缺项\n'
              './run aspire recover      仿真运行瓶身抓取→交接→入篮\n'
              './run real replay         查看已导出的真机关节记录\n'
              './run real demo           合成快照影子规划，零运动命令\n'
              './run check               离线回归检查')
        return 0
    os.chdir(APP)
    os.execv(sys.executable,[sys.executable,'-B',*selected])


if __name__=='__main__':raise SystemExit(main())
