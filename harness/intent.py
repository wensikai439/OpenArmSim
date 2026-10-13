"""Explicit user arm requirements, distinct from model plans and long-term memory.

This deliberately bounded text reader is NOT a general task-language parser.
It binds only named registered objects and explicit positive role phrases.
Unrecognized, quoted or ambiguous text remains unbound. Native receipts, never
plan titles, advance requirements. Installed only in the development harness.
"""
import copy
import hashlib
import json
import re

ALIASES = {
    'bottle': ('瓶子', '瓶', 'bottle'), 'block': ('长方体', '方块', 'block'),
    'can_short': ('短圆柱', 'can_short'), 'can': ('圆柱', 'can'),
    'book': ('书本', '书', 'book'), 'cup': ('杯子', '杯', 'cup'),
}
ARMS = r'(?:左手|右手|left\s+(?:hand|arm)|right\s+(?:hand|arm))'
PICK = r'(?:抓起|抓取|抓住|夹起|拿起|取起|夹住|pick\s+up|grasp|lift)'
PLACE = r'(?:放到|放入|放回|放置|放好|放在|放稳|放(?=[一二三四五六七八九十\d])|place|put)'
RELAY = r'(?:交给|交接到|交接给|交到|递给|递到|handover\s+to|hand\s+(?:it\s+)?to|transfer\s+to)'
BOUNDARY = r'[，,。.;；\n]'
SOURCES = {'pick': 'native_two_finger_contact_and_lift',
    'handover': 'native_receiver_contact_then_donor_release',
    'place': 'native_support_release_and_stability'}


def arm(text):
    return 'left' if text.startswith(('左', 'left')) else 'right'


def objects(text):
    """Longest alias wins at overlapping positions; can_short is not can."""
    hits = []
    for ident, names in ALIASES.items():
        for name in names:
            pattern = re.escape(name) if not name.isascii() else r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])'
            for m in re.finditer(pattern, text, re.I): hits.append((m.start(), m.end(), ident))
    accepted = []
    for start, end, ident in sorted(hits, key=lambda h: (-(h[1]-h[0]), h[0])):
        if not any(start < old[1] and old[0] < end for old in accepted): accepted.append((start, end, ident))
    return sorted(accepted)


def extract(prompt):
    if not isinstance(prompt, str): raise ValueError('original user prompt required')
    contract = {'schema': 'explicit-user-roles/1',
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'scope': 'recognized explicit positive arm roles only; not complete natural-language understanding',
        'requirements': [], 'protected_objects': [], 'unbound_clauses': []}
    if re.search(r'如果|假如|若是|例如|比如|举例|\b(?:if|unless|otherwise)\b', prompt, re.I):
        contract['unbound_clauses'] = [{'range':[0,len(prompt)], 'reason':'conditional or hypothetical task not supported by this bounded reader'}]
        return contract
    mentions = {o[2] for o in objects(prompt)}
    quoted = [(m.start(), m.end()) for m in re.finditer(r'“[^”]*”|「[^」]*」|"[^"\n]*"|‘[^’]*’', prompt)]
    last_pick = {}; last_object = None
    for match in re.finditer(r'[^，,。.;；\n]+', prompt):
        text = match.group(); offset = match.start()
        if any(a < match.end() and offset < b for a, b in quoted):
            contract['unbound_clauses'].append({'range': [offset, match.end()], 'reason': 'quoted text'})
            continue
        local = {o[2] for o in objects(text)}
        if len(local) == 1 and re.search(r'不动|不要移动|保持原位|leave\s+.*(?:alone|unchanged)', text, re.I):
            contract['protected_objects'].extend(local)
            continue
        if re.search(r'不要|不得|不需要|无需|不能|不是|别用|\b(?:do not|don.t|never|not)\b', text, re.I):
            contract['unbound_clauses'].append({'range': [offset, match.end()], 'reason': 'negative or contrastive clause'})
            continue
        events = []
        # No crossing another arm mention, sentence boundary, or a transfer.
        bridge = r'(?:(?!' + ARMS + '|' + RELAY + r').){0,32}?'
        for m in re.finditer(r'(?P<arm>' + ARMS + ')' + bridge + '(?P<verb>' + PICK + ')', text, re.I):
            if text[max(0, m.start()-1):m.start()] == '从': continue  # source hand != acting hand
            ident = acted_object(text, m.start('verb'), m.end('verb'), 'pick', last_object, mentions)
            if ident is None:
                contract['unbound_clauses'].append({'range':[offset+m.start(), offset+m.end()], 'reason':'pick object ambiguous'})
                continue
            item = {'skill': 'pick', 'object_id': ident, 'arm': arm(m['arm'].lower())}
            events.append((m.start(), m.end(), item)); last_pick[ident] = item['arm']; last_object = ident
        for m in re.finditer(r'(?P<arm>' + ARMS + ')' + bridge + '(?P<verb>' + PLACE + ')', text, re.I):
            ident = acted_object(text, m.start('verb'), m.end('verb'), 'place', last_object, mentions)
            if ident is None:
                contract['unbound_clauses'].append({'range':[offset+m.start(), offset+m.end()], 'reason':'placed object ambiguous'})
                continue
            events.append((m.start(), m.end(), {'skill': 'place', 'object_id': ident, 'arm': arm(m['arm'].lower())}))
        for m in re.finditer(RELAY + r'\s*(?P<to>' + ARMS + ')', text, re.I):
            before = text[:m.start()]
            ident = acted_object(text, m.start(), m.end(), 'handover', last_object, mentions)
            if ident is None: continue
            actors = list(re.finditer(ARMS, before, re.I))
            donor = arm(actors[-1].group().lower()) if actors else last_pick.get(ident)
            receiver = arm(m['to'].lower())
            if donor and donor != receiver:
                start = actors[-1].start() if actors else m.start()
                events.append((m.start(), m.end(), {'skill': 'handover', 'object_id': ident,
                    'from_arm': donor, 'to_arm': receiver, '_quote_start': start}))
        for start, end, item in sorted(events, key=lambda e: e[0]):
            quote_start = item.pop('_quote_start', start)
            item.update(source_range=[offset+quote_start, offset+end], source_quote=prompt[offset+quote_start:offset+end])
            contract['requirements'].append(item)
    contract['protected_objects'] = sorted(set(contract['protected_objects']))
    return contract


def first_object(text):
    hits = objects(text)
    if not hits: return None
    if len(hits)>1 and re.search(r'和|与|及|\band\b', text[hits[0][1]:hits[1][0]], re.I): return None
    return hits[0][2]


def acted_object(text, start, end, kind, previous, mentions):
    """Bounded operand grammar; destination names are not manipulation items."""
    before = text[:start]; after = text[end:]
    marker = list(re.finditer(r'把|将', before))
    if marker:
        operand = before[marker[-1].end():]
        if objects(operand): return first_object(operand)
    if kind == 'pick':
        segment = re.split(RELAY+'|'+PLACE, after, maxsplit=1, flags=re.I)[0]
        if objects(segment): return first_object(segment)
    if kind == 'place' and re.fullmatch(r'place|put', text[start:end], re.I):
        operand = re.split(r'\b(?:on|in|into|onto|to)\b', after, maxsplit=1, flags=re.I)[0]
        if objects(operand): return first_object(operand)
    # "瓶子由右手放在圆柱原位": only pre-verb nouns can name the
    # carried item. For Chinese 放在/到/入, post-verb nouns are destinations.
    preceding = objects(before)
    if len({h[2] for h in preceding}) == 1: return preceding[0][2]
    if preceding: return None
    if previous: return previous  # adjacent transfer/place after a named pick
    if len(mentions)==1: return next(iter(mentions))
    return None


def same(required, action):
    return all(action.get(k) == v for k, v in required.items()
        if k in ('skill', 'object_id', 'arm', 'from_arm', 'to_arm'))


def native_events(record):
    """A bound completed oracle receipt and its actual verified substeps only."""
    f = record['feedback']; args = record['arguments']; name = record['skill']
    if record.get('evidence_kind') != 'simulator_oracle': return []
    rows = f.get('substeps', []) if name == 'relocate' else [
        {'skill': name, 'parameters': args, 'feedback': f, 'status': f.get('status')}]
    events = []
    for index, row in enumerate(rows):
        kind = row.get('skill'); feedback = row.get('feedback') or {}; parameters = row.get('parameters') or {}
        if kind not in SOURCES or row.get('status') != 'completed' or feedback.get('primitive_success') is not True:
            continue
        if feedback.get('verification_source') != SOURCES[kind]: continue
        if kind == 'place' and feedback.get('placement', {}).get('confirmed') is not True: continue
        ident = feedback.get('object_id')
        if not ident or parameters.get('object_id', ident) != ident: continue
        event = {'skill': kind, 'object_id': ident}
        if kind == 'handover':
            for key in ('from_arm', 'to_arm'):
                if feedback.get(key) != parameters.get(key): break
                event[key] = parameters[key]
            if len(event) != 4 or feedback.get('held_by') != event['to_arm']: continue
        else:
            side = parameters.get('arm')
            if side not in ('left', 'right') or feedback.get('arm', side) != side: continue
            if kind == 'pick' and feedback.get('held_by') != side: continue
            event['arm'] = side
        event.update(execution_id=record['execution_id'], substep_index=index if name == 'relocate' else None)
        events.append(event)
    return events


def progress(contract, records):
    requirements = contract['requirements']; completed = []
    for record in records:
        for event in native_events(record):
            if len(completed) < len(requirements) and same(requirements[len(completed)], event):
                completed.append(event)
    return {'verified_requirements': len(completed), 'total_requirements': len(requirements),
        'verified_events': completed, 'remaining': copy.deepcopy(requirements[len(completed):])}


def check_action(contract, state, action):
    ident = action.get('object_id')
    if ident in contract['protected_objects']:
        raise ValueError('原始任务要求 '+ident+' 保持原位；本次提议未启动运动')
    remaining = state['remaining']
    if not remaining: return
    next_for_object = next((r for r in remaining if r['object_id'] == ident), None)
    if next_for_object is None: return  # unrelated preparation is not silently prohibited
    if remaining[0]['object_id'] != ident:
        raise ValueError('原始任务顺序尚未完成 '+remaining[0]['object_id']+' 的要求；本次提议未启动运动')
    required = next_for_object; kind = action['skill']
    valid = True
    if kind == required['skill']: valid = same(required, action)
    elif kind == 'pick':
        expected = required.get('from_arm') if required['skill'] == 'handover' else required.get('arm')
        valid = expected is None or action.get('arm') == expected
    elif kind == 'handover' and required['skill'] == 'pick': valid = False
    elif kind == 'handover' and required['skill'] == 'place': valid = action.get('to_arm') == required.get('arm')
    # A temporary set-down/regrasp can be a legitimate recovery. It does not
    # satisfy the still-pending transfer, and finish cannot skip that transfer.
    if not valid:
        fields = {k:v for k,v in required.items() if k in ('skill', 'object_id', 'arm', 'from_arm', 'to_arm')}
        raise ValueError('原始用户要求尚未完成 '+json.dumps(fields, ensure_ascii=False)+
            '；不能换手/换交接方向来替代。可以按新观测更换路径或抓取姿态；本次提议未启动运动')


def proposed_events(name, args):
    if name in ('pick', 'handover', 'place'):
        return [{'skill': name, **{k:args[k] for k in ('object_id', 'arm', 'from_arm', 'to_arm') if k in args}}]
    if name == 'relocate':
        first = args.get('arm'); last = args.get('via_arm', first); ident = args.get('object_id')
        values = [{'skill':'pick', 'object_id':ident, 'arm':first}]
        if first != last: values.append({'skill':'handover', 'object_id':ident, 'from_arm':first, 'to_arm':last})
        return values + [{'skill':'place', 'object_id':ident, 'arm':last}]
    if name in ('run_policy', 'run_pi05_vla'):
        return [{'skill':'pick', 'object_id':'bottle', 'arm':'right'},
            {'skill':'handover', 'object_id':'bottle', 'from_arm':'right', 'to_arm':'left'}]
    if name == 'run_tamp':
        stages = {'reach_bottle':0, 'pick_bottle':1, 'handover_bottle':2, 'place_bottle':3, 'handover_to_basket':3}
        values = [{'skill':'pick', 'object_id':'bottle', 'arm':'right'},
            {'skill':'handover', 'object_id':'bottle', 'from_arm':'right', 'to_arm':'left'},
            {'skill':'place', 'object_id':'bottle', 'arm':'left'}]
        if args.get('task') in ('handover_bottle', 'place_bottle'): return [values[stages[args['task']]-1]]
        return values[:stages.get(args.get('task'), 0)]
    return []


def install_isolated(base):
    class IntentHarness(base):
        def _intent(self, context):
            root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id'))
            session = self.sessions.get(root, {})
            if (context or {}).get('type') != 'llm' or not isinstance(session.get('goal'), str): return None
            if 'user_role_contract' not in session: session['user_role_contract'] = extract(session['goal'])
            return root, session['user_role_contract']

        def _role_records(self, root):
            plan = self.workspace.active_plan or {}
            if plan.get('turn_id') != root: return []
            records = []
            with self.workspace.lock:
                for bound in plan.get('execution_tools', []):
                    row = self.workspace.db.execute('SELECT * FROM experiences WHERE id=? AND role="execute"', (bound['id'],)).fetchone()
                    if row is None or row['tool'] != bound['skill']: continue
                    evidence = json.loads(row['evidence'])
                    after = evidence.get('after', {}).get('observation_id', '')
                    if not after or after.split(':')[0] != row['epoch']: continue
                    records.append({'execution_id': row['id'], 'skill': row['tool'],
                        'arguments': json.loads(row['arguments']), 'feedback': json.loads(row['feedback']),
                        'evidence_kind': row['evidence_kind']})
            return records

        def _invoke(self, name, args, role, context=None):
            pair = self._intent(context) if role == 'execute' else None
            if pair:
                root, contract = pair; records = self._role_records(root); state = progress(contract, records)
                preview = copy.deepcopy(state)
                for event in proposed_events(name, args):
                    check_action(contract, preview, event)
                    if preview['remaining'] and same(preview['remaining'][0], event): preview['remaining'].pop(0)
                if name == 'relocate' and args.get('allow_handover_recovery') and contract['requirements']:
                    raise ValueError('原始任务有明确手部要求；连续工具的自动换手恢复不能保证该要求。请取消 allow_handover_recovery，或用单项技能按原要求恢复；未启动运动')
                if name == 'finish' and args.get('verdict') == 'success' and state['remaining']:
                    # A learned policy's termination lacks role receipts. Keep
                    # its visual report separate, as the existing VLA contract
                    # requires; capability compatibility is never task proof.
                    opaque_only = records and all(r['skill'] in ('run_policy', 'run_pi05_vla', 'run_tamp') for r in records)
                    if not opaque_only:
                        raise ValueError('原始用户手部要求还没有本任务的原生成功回执：'+
                            json.dumps(state['remaining'], ensure_ascii=False)+'；模型改写阶段或不同方向的成功不能代替。请按实际失败记录结束 failure/uncertain，或继续原要求')
            result = super()._invoke(name, args, role, context)
            if pair and name in ('observe', 'set_plan', 'get_run_status', 'finish'):
                detail = {'contract': copy.deepcopy(contract), **progress(contract, self._role_records(root)),
                    'route_proposals_are_not_completion_evidence': True,
                    'learned_policy_role_verification': 'UNKNOWN without independent role evidence',
                    'instruction': 'Preserve these original user requirements through recovery. Model plans choose paths; they cannot redefine these roles. Unbound language is not certified.'}
                result['user_requirements'] = detail
                if name in ('set_plan', 'finish') and (self.workspace.active_plan or {}).get('turn_id') == root:
                    with self.workspace.lock:
                        self.workspace.active_plan['user_requirements'] = detail
                        self.workspace.save_plan()
            return result
    return IntentHarness
