"""Short method index, explicit applicability and separate evidence archive.

Only the isolated multi-object app installs this adapter. Historical coordinates,
images and raw feedback never enter the default index. Detail/receipt gates and
user forgetting stay owned by Workspace.
"""
import copy
import json

from .object_gpu import ObjectGPUWorkspace, consolidate_gpu

PROTOCOLS = {
    'object_motion': '用新观测与当前夹持状态重新规划，核验整条路径和实际末端反馈。历史坐标不能复用；末端到位不等于物体任务完成。',
    'object_manipulation': '锁定本次物体与原目标；抓取查持续双指接触和抬升，交接查接收手独立持物，放置查支撑、松手、撤离和连续稳定。另查非目标物体是否被碰动。历史经验不证明开放物体或真机迁移。',
}
DISPLACEMENT = '非目标物体被碰动后停止本任务并保留证据。改进路径净空，再另开新测试；不能改原目标、关掉检查或把重置后的成功算成本次恢复。'


def consolidate(receipts, forgotten=()):
    rows = []; scopes = {}
    for receipt in receipts:
        item = copy.deepcopy(receipt)
        if item['feedback'].get('execution_binding', {}).get('valid') is False: continue
        if (item['evidence_kind'] == 'simulator_oracle' and item['feedback'].get('failure_code') == 'invalid'
            and str(item['feedback'].get('message', '')).startswith('non-target object displaced: ')):
            item['feedback']['failure_code'] = 'non_target_object_displaced'
        intent = item['arguments'].get('next_handover') if item['tool'] == 'pick' else None
        if intent:
            receiver = intent.get('to_arm')
            if receiver not in ('left', 'right'): continue
            scoped = str(item.get('context')) + '|next_receiver=' + receiver
            scopes[scoped] = item.get('context'); item['context'] = scoped
        rows.append(item)
    lessons = consolidate_gpu(rows, forgotten)
    originals = {item['id']: item for item in receipts}
    for lesson in lessons:
        if lesson['context'] in scopes: lesson['context'] = scopes[lesson['context']]
        source = next((originals[s['receipt_id']] for s in lesson.get('sources', []) if s['receipt_id'] in originals), None)
        args = source['arguments'] if source else {}
        conditions = {key: copy.deepcopy(lesson[key]) for key in ('object_id', 'payload_state', 'backend', 'surface')
            if lesson.get(key) is not None}
        conditions.update({key: args[key] for key in ('arm', 'from_arm', 'to_arm') if args.get(key) in ('left', 'right')})
        if args.get('next_handover'):
            conditions['next_handover'] = copy.deepcopy(args['next_handover'])
            lesson['title'] = lesson['title'].replace('抓取', '交接前抓取', 1)
        lesson['conditions'] = conditions
        tool = lesson['tool']; code = lesson.get('failure_code')
        if code in ('goal_collision', 'ik_unreachable'):
            lesson['summary'] = ('保留锁定的原目标；换接近姿态或路线后重新规划，并检查非目标物体是否移动。无可行解就报告失败，不能改目标或关闭检查来通过。')
        elif lesson['outcome'] == 'success' and tool != 'relocate':
            lesson['protocol_id'] = 'object_motion' if tool in ('move_relative', 'move_to_pose') else 'object_manipulation'
            statement = {'pick': '已核验双指接触与实际抬升。', 'handover': '已核验接收手接触、供物手释放与独立持物。',
                'place': '已核验原目标、支撑、释放与稳定。', 'move_relative': '已核验本次末端位姿。',
                'move_to_pose': '已核验本次末端位姿。', 'set_gripper': '已核验编码器开合稳定。',
                'run_policy': '仅对应匹配的模型与动作契约。'}.get(tool, lesson['summary'])
            errors = lesson.get('measured_error_mm')
            if errors: statement += f' 记录误差 {errors[0]:g}–{errors[1]:g} mm。'
            lesson['summary'] = statement
    return lessons


def index_entry(lesson):
    # The base index intentionally omits sources. Add typed applicability only,
    # never a raw receipt, camera path, transient pose or model conjecture.
    from .knowledge import index_entry as current
    base = getattr(current, '_object_memory_original', current)
    item = base(lesson)
    for key in ('object_id', 'payload_state', 'backend', 'surface', 'conditions'):
        if key in lesson: item[key] = copy.deepcopy(lesson[key])
    return item


class ObjectMemoryWorkspace(ObjectGPUWorkspace):
    index_byte_limit = 8192

    def consolidate_memory(self, receipts, forgotten=()):
        return consolidate(receipts, forgotten)

    def memory(self, role, epoch, *, public=False, query='', context=None, include_oracle=False, skill=None, compact=False):
        value = super().memory(role, epoch, public=public, query=query, context=context,
            include_oracle=include_oracle, skill=skill, compact=compact)
        if public or not compact: return value
        # The index scope is explicit task bookkeeping, never historical object
        # localization. Filter BEFORE top-k so recent unrelated objects cannot
        # crowd out an older compatible method. Full user archive stays intact.
        selected = getattr(self, 'index_object_id', None)
        with self.lock:
            sql = 'SELECT * FROM experiences WHERE role=?'; params = [role]
            if context is not None: sql += ' AND context=?'; params.append(self._context(context))
            if skill: sql += ' AND tool=?'; params.append(skill)
            rows = [dict(row) for row in self.db.execute(sql + ' ORDER BY created DESC', params)]
            forgotten = {row['id'] for row in self.db.execute('SELECT id FROM forgotten_knowledge WHERE role=?', (role,))}
        for row in rows:
            for key in ('arguments', 'feedback', 'evidence'): row[key] = json.loads(row[key])
            row['compatible'] = True
        lessons = self.consolidate_memory(rows, forgotten)
        if selected:
            lessons = [item for item in lessons if item.get('object_id') in (None, selected)]
        candidates = self._rank(lessons, query, 8) if query else lessons[:8]
        from .knowledge import PROTOCOLS as protocols
        index = []; rules = {}
        for lesson in candidates:
            item = index_entry(lesson); new_rules = dict(rules)
            if lesson.get('protocol_id') in protocols: new_rules[lesson['protocol_id']] = protocols[lesson['protocol_id']]
            size = len(json.dumps({'index': index + [item], 'protocols': new_rules}, ensure_ascii=False).encode('utf-8'))
            if size > self.index_byte_limit: continue
            index.append(item); rules = new_rules
        value['knowledge'] = {'index': index, 'protocols': rules}
        value['retrieval'].update(object_id=selected, format='task object index; expand stable knowledge IDs on demand',
            matched_methods=len(lessons), index_byte_limit=self.index_byte_limit,
            index_bytes=len(json.dumps(value['knowledge'], ensure_ascii=False).encode('utf-8')),
            selected_outcomes={outcome: sum(item['outcome'] == outcome for item in index)
                for outcome in ('success', 'failure', 'cancelled', 'rejected', 'unverified')})
        return value

    @staticmethod
    def _outcome(feedback, evidence_kind):
        if feedback.get('execution_binding', {}).get('valid') is False: return 'failure'
        return ObjectGPUWorkspace._outcome(feedback, evidence_kind)

    def record_feedback(self, ident, role, epoch, goal, tool, arguments, feedback, context=None,
        evidence_kind='sensor_execution', evidence=None):
        value = copy.deepcopy(feedback)
        if tool == 'handover' and value.get('primitive_success'):
            expected = {key: arguments.get(key) for key in ('object_id', 'from_arm', 'to_arm')}
            valid = (expected['object_id'] in ('bottle', 'block', 'can') and expected['from_arm'] in ('left', 'right')
                and expected['to_arm'] in ('left', 'right') and expected['from_arm'] != expected['to_arm']
                and all(value.get(key) == wanted for key, wanted in expected.items())
                and value.get('held_by') == expected['to_arm'])
            value['execution_binding'] = {'valid': valid, 'expected': expected,
                'source': 'trusted workspace checks against this execution tool arguments; native feedback preserved'}
        return super().record_feedback(ident, role, epoch, goal, tool, arguments, value, context, evidence_kind, evidence)

    @staticmethod
    def _step_done(step, feedback, plan):
        if feedback.get('execution_binding', {}).get('valid') is False: return False
        if step['checkpoint'] == 'handover_confirmed':
            goal = plan.get('placement_goal')
            if goal and feedback.get('object_id') != goal['object_id']: return False
            if (feedback.get('from_arm') not in ('left', 'right') or feedback.get('to_arm') not in ('left', 'right')
                or feedback['from_arm'] == feedback['to_arm'] or feedback.get('held_by') != feedback['to_arm']): return False
        return ObjectGPUWorkspace._step_done(step, feedback, plan)

    def knowledge_detail(self, role, epoch, ident, *, context=None, include_oracle=False):
        value = super().knowledge_detail(role, epoch, ident, context=context, include_oracle=include_oracle)
        memory = self.memory(role, epoch, context=context, include_oracle=include_oracle)
        lesson = next(item for item in memory['knowledge']['lessons'] if item['id'] == ident)
        return {**value, 'conditions': copy.deepcopy(lesson.get('conditions', {}))}


def install_isolated():
    from . import knowledge
    if knowledge.index_entry is not index_entry:
        index_entry._object_memory_original = knowledge.index_entry
    knowledge.index_entry = index_entry
    knowledge.PROTOCOLS.update(PROTOCOLS)
    knowledge.RECOVERY['non_target_object_displaced'] = DISPLACEMENT
