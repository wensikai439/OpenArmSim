"""Offline towel-task preparation. Reads metadata/config only; never opens Runtime sockets."""
from __future__ import annotations
import os

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(str(Path(__file__).resolve().parents[1]))
CONTROL = Path(os.environ.get('PIPER_CONTROL_ROOT', str(ROOT/'external/robot-control')))
DATASET = Path(os.environ.get('PIPER_TOWEL_DATASET', str(ROOT/'external/fold_towel')))
CAMERAS = ['cam_left', 'cam_middle', 'cam_right']
ORDER = [*(f'left_joint{i}' for i in range(1, 7)), 'left_gripper',
         *(f'right_joint{i}' for i in range(1, 7)), 'right_gripper']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_document(path):
    raw = Path(path).read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError('配置文件超过 1 MiB')
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError('配置必须是 JSON 对象')
    return raw, document


def dataset_report():
    path = DATASET / 'meta/info.json'
    raw, info = read_document(path)
    features = info.get('features', {})
    checks = [
        {'name': '数据时间步为 30 Hz', 'ok': info.get('fps') == 30},
        {'name': '双臂状态与动作均为 14 维，顺序为左臂、左爪、右臂、右爪',
         'ok': all(features.get(k, {}).get('shape') == [14]
                   and features.get(k, {}).get('names') == ORDER
                   for k in ['observation.state', 'action'])},
        {'name': '左、中、右三相机均为 640×480 RGB / 30 fps',
         'ok': all(features.get('observation.images.' + c, {}).get('shape') == [480, 640, 3]
                   and features.get('observation.images.' + c, {}).get('info', {}).get('video.fps') == 30
                   for c in CAMERAS)},
        {'name': '存在已转换回合', 'ok': isinstance(info.get('total_episodes'), int)
         and info['total_episodes'] > 0},
    ]
    return {'path': str(DATASET), 'fps': info.get('fps'), 'episodes': info.get('total_episodes'),
            'state_dim': features.get('observation.state', {}).get('shape'),
            'camera_names': CAMERAS, 'checks': checks, 'metadata_ok': all(x['ok'] for x in checks),
            'metadata_path': str(path), 'metadata_sha256': sha(raw),
            'metadata_modified_unix': path.stat().st_mtime,
            'scope': 'existing_converted_metadata_only',
            'current_collection_finalized': False, 'per_frame_integrity_checked': False}


def native_policy_entry():
    path = CONTROL / 'piper_baseline/unified_runtime.py'
    raw = path.read_bytes()
    for node in ast.walk(ast.parse(raw)):
        if isinstance(node, ast.FunctionDef) and node.name == 'start_act_task':
            arguments = node.args.args + node.args.kwonlyargs
            bounded = any(x.arg == 'duration_s' for x in arguments)
            return {'source': str(path), 'source_sha256': sha(raw), 'task_type': 'act_approved',
                    'seconds': None if bounded else 1.0, 'bounded_duration_supported': bounded,
                    'default_seconds': 1.0, 'maximum_seconds': 60.0 if bounded else 1.0,
                    'continuous': False, 'scope': 'source_interface_only_not_running_runtime'}
    return {'task_type': 'act_approved', 'seconds': None, 'source': str(path),
            'bounded_duration_supported': False}


def status():
    _, cfg = read_document(ROOT / 'config/real_runtime.json')
    return {'task': '毛巾任务', 'stage': 'software_preparation',
            'native_io_performed': False, 'hardware_execution_enabled': False,
            'bridge': {'mode': cfg.get('mode'), 'field_release_present': bool(cfg.get('release')),
                       'prepared_task_types': ['bimanual_program', 'act_approved'],
                       'policy_binding_implemented': True, 'policy_instance_prepared': False},
            'dataset': dataset_report(), 'native_policy_entry': native_policy_entry(),
            'blockers': [
                '正在采集的数据尚未冻结；当前只核对已有 fold_towel 转换数据的元信息。',
                '尚未指定本轮毛巾任务训练出的模型及部署配置。',
                '毛巾策略任务桥已实现；尚未指定模型，因此未生成可执行的毛巾任务制品。',
                '有限时长接口按源码检查；正在运行的原 Runtime 未重启，不能据此声称已加载新接口。',
                '模型真实推理、现场相机和停止效果尚未验收。'],
            'physical_task_success': None}


def inspect_model_config(path_text):
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError('请填写服务器上的毛巾 ACT 部署配置 JSON 路径')
    path = Path(path_text.strip())
    if not path.is_absolute():
        raise ValueError('请使用服务器绝对路径')
    raw, config = read_document(path)
    errors = []
    if config.get('model_id') != 'act' or config.get('adapter') != 'act':
        errors.append('本次入口只核对已有 ACT 适配器。')
    if Path(config.get('dataset') or '/missing').resolve() != DATASET.resolve():
        errors.append('配置数据源不是 fold_towel，不能将抓放模型登记成毛巾模型。')
    data = dataset_report()
    if not data['metadata_ok']:
        errors.append('毛巾数据元信息与部署契约不匹配。')
    if not errors:
        checkpoint = Path(config.get('checkpoint') or '/missing')
        if not checkpoint.is_absolute():
            errors.append('checkpoint 需要绝对路径。')
        else:
            try:
                _, policy = read_document(checkpoint / 'config.json')
                inputs, outputs = policy.get('input_features', {}), policy.get('output_features', {})
                if (policy.get('type') != 'act' or policy.get('chunk_size') != 32
                        or inputs.get('observation.state', {}).get('shape') != [14]
                        or outputs.get('action', {}).get('shape') != [14]
                        or any(inputs.get('observation.images.' + c, {}).get('shape') != [3, 480, 640]
                               for c in CAMERAS)):
                    errors.append('ACT checkpoint 的类型、chunk、14D 或三路图像契约不匹配。')
                weights = checkpoint / 'model.safetensors'
                if not weights.is_file() or weights.stat().st_size == 0:
                    errors.append('checkpoint 缺少非空 model.safetensors。')
            except (OSError, ValueError) as exc:
                errors.append('checkpoint 配置不可读：' + str(exc))
    native = {'checked': False}
    if not errors:
        code = ('import json,sys,hashlib; from piper_baseline.model_config import resolve_config; '
                'c=resolve_config(sys.argv[1]); '
                'print(json.dumps({"config":{k:c[k] for k in '
                '["model_id","fps","chunk_size","n_action_steps","camera_names"]},'
                '"resolved_config_sha256":hashlib.sha256(json.dumps(c,sort_keys=True,'
                'ensure_ascii=False,default=str).encode()).hexdigest(),'
                '"files":{k:c[k] for k in ["checkpoint","training_lock","action_smoothing"]}}))')
        result = subprocess.run([str(CONTROL / '.conda/ros-humble/bin/python'), '-B', '-c', code, str(path)],
                                cwd=CONTROL, text=True, capture_output=True, timeout=10)
        native = {'checked': True, 'exit_code': result.returncode}
        if result.returncode:
            errors.append('原生纯配置检查拒绝：' + result.stderr[-1200:])
        else:
            native.update(json.loads(result.stdout))
    model_files = {}
    if not errors:
        files = native.pop('files')
        checkpoint = Path(files['checkpoint'])
        for name, file in {'checkpoint_config': checkpoint / 'config.json',
                           'checkpoint_weights': checkpoint / 'model.safetensors',
                           'training_lock': Path(files['training_lock']),
                           'action_smoothing': Path(files['action_smoothing'])}.items():
            with file.open('rb') as stream:
                fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
            model_files[name] = {'path': str(file), 'sha256': fingerprint}
    return {'ok': not errors, 'errors': errors, 'source_path': str(path), 'source_sha256': sha(raw),
            'checked_at': datetime.now(timezone.utc).isoformat(), 'native_checker': native,
            'model_files': model_files,
            'dataset_metadata_sha256': data['metadata_sha256'], 'native_io_performed': False,
            'policy_inference_verified': False, 'training_task_provenance_verified': False,
            'runtime_configuration_modified': False, 'hardware_execution_enabled': False,
            'note': '仅核对配置与元信息；权重训练来源、真实推理和完整任务调用仍需验收。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config', help='Inspect an explicit existing ACT config without loading weights')
    args = parser.parse_args()
    result = inspect_model_config(args.model_config) if args.model_config else status()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('ok') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
