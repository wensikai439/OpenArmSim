"""Read-only semantic goal verification: absolute and relative forms can agree."""
import copy
import hashlib
import json
from pathlib import Path

from sim.environment import ROOT
from sim.recording import dump

OUT = ROOT/'runs/20261009-objects'


def score(record, protocol):
    plan = record['terminal']['agent']['plan']; goal = plan.get('placement_goal', {})
    initial = next(row for row in protocol['initial']['diagnostics']['scene_directory']['objects']
        if row['object_id'] == protocol['object_id'])['origin_m'][:2]
    target = [initial[0]+protocol['forward_m'], initial[1]]
    def same(left, right):
        return len(left)==len(right)==2 and all(abs(a-b)<=1e-6 for a,b in zip(left,right))
    checks = copy.deepcopy(record['checks'])
    checks['locked_original_object_goal'] = (goal.get('object_id') == protocol['object_id']
        and goal.get('surface') == 'table' and same(goal.get('target_xy_m', []), target)
        and same(goal.get('anchor', {}).get('object_xy_m', []), initial))
    receipts = record['receipts']
    all_events = [e for t in record['turns'] for e in t['trace'] if e['kind']=='tool']
    selection = [e for e in all_events if e['name']=='select_manipulation_object' and e['status']=='completed']
    checks['one_requested_object_selection'] = (len(selection)==1
        and selection[0]['detail']['arguments']['object_id']==protocol['object_id']
        and selection[0]['detail']['result']['physical_state_preserved'] is True)
    actions = [e['name'] for e in all_events if e['name'] in {'pick','place','handover','relocate','move_relative','move_to_pose','run_policy','set_gripper'} and e['status']=='completed']
    checks['exact_requested_native_tools'] = actions==protocol['expected_native_tools']
    checks['correct_native_object_receipts'] = len(receipts)==2 and all(r['feedback'].get('object_id')==protocol['object_id']
        and r['feedback'].get('primitive_success') is True for r in receipts)
    checks['correct_selected_arm'] = all(r['arguments'].get('arm')==protocol['arm'] for r in receipts)
    checks['all_stages_done'] = bool(plan.get('steps')) and all(s['status']=='done' for s in plan['steps'])
    checks['all_turns_completed'] = bool(record['turns']) and all(t['status']=='completed' for t in record['turns'])
    checks['physical_placement_confirmed'] = plan.get('goal_assessment',{}).get('confirmed') is True
    checks['declared_waiting_and_budget'] = plan.get('timing_mode')==protocol['timing_mode'] and plan.get('time_budget_s')==protocol['time_budget_s']
    checks['no_prepare_or_reset'] = not any(e['name']=='prepare_simulation' and e['status']=='completed' for e in all_events)
    checks['run_manifests_name_actual_object'] = bool(record.get('manifests')) and all(
        m['details'].get('object_selection',{}).get('object_id')==protocol['object_id']
        and m['task']=='Manipulate selected registered object: '+protocol['object_id'] for m in record['manifests'])
    sources = ['native_two_finger_contact_and_lift', 'native_support_release_and_stability']
    checks['native_receipt_sources'] = len(receipts)==2 and all(r['feedback'].get('verification_source')==source
        for r,source in zip(receipts,sources))
    placement = receipts[-1]['feedback'].get('placement', {}) if receipts else {}
    checks['original_object_placement_gate'] = (placement.get('object_id')==protocol['object_id']
        and placement.get('confirmed') is True and placement.get('stable_seconds',0)>=1.
        and same(placement.get('xy_m',[]),target) and placement.get('tolerance_m',1)<=.010
        and receipts[-1]['feedback'].get('position_error_m',1)<=.010)
    events = [e for t in record['turns'] for e in t['trace'] if e['kind']=='tool'
        and e['name'] in protocol['expected_native_tools'] and e['status']=='completed']
    checks['execution_id_bindings'] = len(events)==len(receipts)==2 and all(
        event['detail']['result'].get('execution_id')==receipt['execution']['id'] for event,receipt in zip(events,receipts))
    return {'passed':all(checks.values()),'checks':checks,'expected_target_xy_m':target,
        'actual_plan_target_xy_m':goal.get('target_xy_m'),
        'scope':'exact predeclared selected-object forward-placement task; no generalization or transfer claim'}


def main():
    source = OUT/'workbench.json'; target = OUT/'workbench-verified.json'
    if target.exists(): raise ValueError('preserve prior scoring revision')
    record = json.loads(source.read_text()); protocol = json.loads((OUT/'workbench.protocol.json').read_text())
    result = {**score(record,protocol), 'original_input':source.name,'original_collector_verdict':record['passed'],
        'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'correction':'the model used the equivalent absolute target; compare original selected-object anchor plus requested offset, not presence of offset_xy_m',
        'physical_thresholds_changed':False,'new_API_calls':0,'new_physics_actions':0,
        'hardware_access':False,'transfer_ready':False}
    dump(target,result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
