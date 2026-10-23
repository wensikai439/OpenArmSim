---
name: bottle-relay
description: Validated OpenArm simulator bottle relay using geometry-oracle TAMP tools.
---

# 抓瓶、交接与入篮

冻结最终测试：20/20；相同工具基线：20/20。扩展16位置：生成程序15/16，参数化工具15/16，固定轨迹0/16。本轮没有证明技能凝练提高成功率。当前只覆盖相同直立瓶子的位置变化，使用仿真几何。未验证真机迁移、视觉定位或失败恢复。

以下是 Haiku 从开发证据凝练的说明。独立上下文成功复写代表方法能够传达；未发生的故障分支只属于程序保护，不属于已经学会或实测过的恢复技能。原文中的单次坐标和步数不得用作新场景的动作目标；文末“earlier version”没有独立版本对照证据。

# Skill: Right-to-left bottle relay into basket (OpenArm, simulated geometry)

## Applicability
- Task: bottle upright on table at a varying position; RIGHT arm picks, hands to LEFT arm, LEFT arm places into `basket`.
- Uses only declared tools: `get_observation`, `pick`, `handover`, `place`. No stored coordinates; position comes from observation at run time.
- Evidence: 5 development cases (seeds 51-55), bottle_xy between roughly (0.47, -0.14) and (0.54, -0.21). All 5 reached protocol success with program_ok. This is a small sample from one seed range. It says nothing about other seed ranges, bottle poses, or disturbances.

## Program structure (gated by observation)
1. Read observation; abort with ValueError if `upright` is not True.
2. `pick(arm='right', hold_seconds=2, next_handover={'to_arm':'left'})`. Check `.ok`, then re-observe and require `'right'` in `held_by`.
3. `handover(from_arm='right', to_arm='left', hold_seconds=2)`. Check `.ok`, then re-observe and require `'left'` in `held_by`.
4. `place(arm='left', receptacle_id='basket')`. Check `.ok`, then re-observe.
5. Set `result` to a compact report. Returning normally is not task success. Success must come from the protocol checks, which are external to this program.

## Failure signatures and checks
- Pre-check failure: `upright` is False at start. Raise before any motion. This is a physical/precondition state, not an API error.
- `pick` not ok: read `diagnostics['failure_code']` and `current_holder`. Raise with that code. Do not retry blindly.
- Holder mismatch after pick: `held_by` lacks `'right'` even though `.ok` is true. Treat as a contact/grasp failure, not success.
- Handover not ok or receiver mismatch: `'left'` absent from `held_by` after handover. Do not proceed to place.
- `place` not ok: read `failure_code`. Typical causes are a lost hold or an unsupported basket drop.
- Final check: `held_by` should be empty and `bottle_center_m` should be near the basket after place. Observed final centers were near y≈0.36, z≈1.02 across all 5 cases.

## Recovery
- The development program has no in-program recovery. Every branch raises. Any repair should be a new program version with a changed approach, not a silent retry.
- Motion budget: at most 12 motion calls and 120 simulated seconds. The 5 successful runs used 3 motion calls and about 48 simulated seconds, so there is headroom for one re-pick attempt in principle. That is untested.
- `move_relative` (norm ≤ 0.15 m) is available for small corrections, but no development run used it. Its effect is unverified.

## Observed timing (development, not a guarantee)
- pick: 395-397 steps, about 13.2 sim s. handover: 694-716 steps, about 36-37 sim s. place: 327-352 steps, about 48 sim s cumulative.

## Limitations
- Sample size is 5 cases in one seed range. No failure cases were observed, so failure signatures above are predicted from the program's checks, not from demonstrated failures.
- Only seeds 51-55 and bottle_xy within the sampled box were tested. Behavior outside that region is unknown.
- The 5 development runs were stable. Recovery paths are not demonstrated.
- Do not describe this as vision-only or real-robot transfer. The program uses declared simulator geometry through `get_observation`.
- Planner capabilities used (pick, handover, place with receptacle) are inherited from the existing toolset, not introduced by this skill.

## Code sketch
```python
obs = get_observation()
report = {'initial_held_by': obs.get('held_by'), 'initial_bottle_center_m': obs.get('bottle_center_m'), 'initial_upright': obs.get('upright')}
if obs.get('upright') is not True:
    raise ValueError('bottle not upright at start')

p = pick(arm='right', hold_seconds=2, next_handover={'to_arm': 'left'})
report['pick_ok'] = p.ok
if not p.ok:
    raise ValueError('pick failed: ' + str(p.diagnostics.get('failure_code')))
if 'right' not in (get_observation().get('held_by') or []):
    raise ValueError('right does not hold after pick')

h = handover(from_arm='right', to_arm='left', hold_seconds=2)
report['handover_ok'] = h.ok
if not h.ok:
    raise ValueError('handover failed')
if 'left' not in (get_observation().get('held_by') or []):
    raise ValueError('left does not hold after handover')

pl = place(arm='left', receptacle_id='basket')
report['place_ok'] = pl.ok
if not pl.ok:
    raise ValueError('place failed')

obs = get_observation()
report['final_held_by'] = obs.get('held_by')
report['final_bottle_center_m'] = obs.get('bottle_center_m')
result = report
```

## Status
- Demonstrated: 5/5 development cases reached protocol success in one seed range (51-55).
- Not demonstrated: failure recovery, other seeds, bottle poses outside the sampled box, any `move_relative` use, or improvement over the earlier version (no paired evaluation of v1 vs. this version is provided; v1 had 5 passes with no detailed results).
