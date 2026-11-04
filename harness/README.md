# Harness

## Additional framework: PhyAgentOS

[`phyagentos/`](phyagentos/README.md) contains the independently packaged
PhyAgentOS Core/Forge, console and PiPER Skill adapters. It does not replace
this Harness or ASPIRE, and is not yet wired to OpenArm simulation or hardware.
Its external assets, control runtime, credentials and recordings are not included.
See its README for setup and the distinction between validated capabilities and
unfinished action-model integration. Existing entry points remain unchanged.

LLM API → typed skill gateway → TAMP / learned-policy adapter → native MuJoCo
→ synchronized cameras and robot feedback → verification → next LLM decision.

The UI keeps conversation on the left, video above stages on the right. The
optional **技能** panel shows backend readiness and exposes a bounded Cartesian
motion check. The **记忆** panel shows durable user preferences and evidence-backed methods or failure remedies. Ephemeral scene beliefs and raw traces belong to the current task and experiment archives, not the long-term UI.

The optional bottle workbench now defaults NEW geometric plans to
`measured_delay` and a bounded120-second budget when the model omits them;
the accepted receipt names the defaulted fields. Explicit paused diagnostics, continuations, locked budgets and
the independent VLA timing contract remain unchanged. A motion cannot silently
use `move_relative` while its pending stage declares `move_to_pose`; the model
can update its plan while retaining completed stages and the original goal.
Table-goal grasp hints are bound to the current root task, never a stale plan
or manual call. Memory groups grasp purpose, requested hold duration and audit
method without historical target coordinates. Early continuous-audit failures
retain their method and incomplete status, never become final-window successes.
Native and actual API evidence: `runs/20261009-bottle/index.html`.

The optional task queue in `harness/task_queue.py` extends the existing
`set_plan` with ordered `object_goals` (2–3 registered objects) and per-stage
`object_id`. All final targets anchor before motion, with one original clock
and budget. Declaration reuses fresh native inspection and may select its
first object as bookkeeping; it never starts robot motion. After current
placement has a trusted receipt, release, original-target support and stability,
`select_manipulation_object` advances only to the next declared object. It
preserves qpos/qvel/ctrl/time, targets, stage receipts and the original budget.
`TaskScene` passively monitors all registered objects on the original physics
substeps; task acceptance requires every placed object still meets its goal,
plus non-requested objects remain undisturbed. Native task goals cannot be
replaced. Model deliberation is not a running motor stage. Single-object tools
and the fixed demo remain available; this is not general symbolic TAMP.

The [declared occupied-space test](../runs/20261009-tasks/index.html) completed
five native tools across two objects. Native and independent real API passed;
the browser's physical goals passed but its model submitted visual uncertainty,
so the original full-stage check remains failed (9/10 checks, 5/6 stages).
All failed attempts, source versions and original120s measured-delay budgets
are preserved. Current memory projection has3 box methods/2973 UTF-8 bytes,
versus5 cylinder methods/4731 bytes; neither includes raw receipts or old poses.

2026-10-09 multi-object development app (`./run objects`, port8767): `pick` and
directed `handover` use registered bottle/box/cylinder geometry; box left→right
and cylinder right→left have declared native relay/10s hold evidence. Cylinder
`pick` accepts `next_handover.to_arm` to rank donor grasps. A locked table goal
supplies scratch placement lookahead without changing the object target;
`next_placement_transit` is only an unexecuted cuRobo pose hypothesis.
The original bottle macro stays separate, and multi-object learned policies
remain disabled without a matching contract. Native relay paths add12mm
clearance from non-target free objects;3mm displacement invalidates the task.
The harness rejects subsequent motion before dispatch and tells the model to
finish failure/uncertain. These development thresholds are not physical accuracy.

`resolve_placement_transit` in `harness/object_transport.py` computes an
UNEXECUTED TCP target above this task's original table placement goal, from the
fresh selected-object pose and actual TCP grasp relation. A same-epoch,
object/direction-bound successful relay receipt can suggest the upright yaw;
its old world pose is never reused. Empty/dual holds, wrong objects or roots,
changed goals, stale epochs and object/TCP drift reject before motion. The model
must pass the exact `motion_request` to `move_to_pose`, wait for native GPU
feedback, then `place`. Current full native box/cylinder cases passed; actual
API and browser box compositions passed at13/14 calls and0.64/1.00mm placement
error with the original120s measured-delay contract. They are development
cases, not arbitrary-object or physical-transfer rates.

`harness/object_memory.py` installs a process-local method index with typed
object/payload/arm/direction/intent conditions. Common rules occur once; plain
and relay-conditioned grasps have distinct titles. Native displacement failures
become one specific lesson; raw receipts remain archived and gated. Forgetting
suppresses the stable knowledge ID without deleting experiment evidence.
Default indices carry neither historical poses nor camera links. See the
[relay evidence](../runs/20261009-handover/index.html) for all failures and scope.
After explicit task-object selection, the compact decision index filters that
identity before ranking/top-k; raw archive and the public method panel remain
available. At most8 entries and8192 UTF-8 bytes of index/common rules enter this
projection. Method details stay on-demand; raw experiments do not become
current perception or erase previous failures.


2026-10-08: `pick(arm=left|right)` and directed `handover(from_arm, to_arm)` are available for the declared bottle. The reverse operator searches shared reachability, re-grounds the actual post-transit bottle pose/axis, confirms receiver contact before releasing the donor, and verifies receiver-only hold. The fixed right-to-left-to-basket macro remains separate. This is candidate oracle geometry, not arbitrary-object perception.

`prepare_simulation(mode=preserve|reset)` is a bounded new-task preflight. Preserve keeps all physical state, actuator commands and contact dwell; it clears prior task milestones and starts a new 40-second window. Invalid or recorded-clock state rejects preserve. The LLM cannot reset after starting a motion in the same goal. After preparation it must re-observe and publish a new plan. `grasp_confirmed`, `handover_confirmed` and `placement_confirmed` bind actual skill receipts, independently of legacy right-to-left stage events.

Memory uses one small, pose-free index, not an episode transcript: preferences, validated procedures and actionable failure remedies. A single successful tiny move and model verdicts are not promoted. Coordinates do not split one method into dozens of entries; arm, direction, backend, cause, obstacle family and compatible contract remain distinct. Knowledge details and experimental before/after receipts are separate on-demand reads. Raw oracle evidence still requires explicit `include_oracle` and taints the epoch for learned-policy control; the default procedure index contains no poses, raw trace or camera URLs. Forgetting a derived method leaves experiment evidence available for auditing.

References: [Claude Code memory](https://code.claude.com/docs/en/memory), [OpenAI memory and compaction](https://developers.openai.com/cookbook/examples/agents_sdk/building_reliable_agents_memory_compaction), [Pigey](https://lianegalanti.github.io/Pigey/). These are design references, not a claim of full replication.

Current backend routing: `move_to_pose` / `move_relative` accept an optional
`planner: tamp | curobo` (default `tamp`). Object skills remain TAMP-specific;
cuRobo never substitutes native IK on failure. cuRobo supports empty-gripper
Cartesian moves, verifies software FK, checks the whole proposed discrete path
with native collision geometry before yielding actuator targets, and records
its exact source revision and geometry hash. This is not physical calibration.
ACT is connected with strict local weight/I/O verification; task transfer has
not passed. The UI states this separately from inference readiness.

Memory borrows the compact-index / on-demand-detail separation from
[Claude Code](https://code.claude.com/docs/en/memory) and outcome banks from
[RoboHarness](https://github.com/LZY-1021/RoboHarness/blob/main/src/soma_memory.py).
Compatible duplicate receipts form one lesson with a repeat count, newest
representative feedback and original evidence pointers. Raw rows remain intact.
SQL deduplicates before each bank's retrieval limit; generic failure codes also
retain their distinct error cause. Scope, arguments, backend and provenance
remain part of the key. This is deterministic structured retrieval, not a
claimed vector-memory implementation or measured generalization benefit.

The backend tabs use the accessibility pattern in
[Radix Tabs](https://www.radix-ui.com/primitives/docs/components/tabs): roving
focus, selected state and Arrow/Home/End navigation, implemented in the existing
native frontend. No frontend framework or package migration was introduced.

## Skills

`GET /api/skills` returns exact capability contracts. LLM tools are `observe`,
`list_skills`, `inspect_manipulation`, `set_plan`, `move_to_pose`, `move_relative`, `pick`, `handover`,
`place`, `run_policy`, `get_run_status`, `stop`, `finish`, `recall_memory`, and
`remember_scene`. Legacy `run_tamp`/`run_pi05_vla` remain direct developer APIs,
not tools declared to the LLM.

- `move_to_pose`: left/right TCP, world-frame metres, quaternion **xyzw**.
- `move_relative`: world or current tool axes, vector norm at most 15 cm.
- `pick`: declared upright `bottle`, either arm, supported on table or basket floor.
- `handover`: bottle, either direction; donor hold and lift required.
- `place`: held bottle, either arm, table target (`target_xy_m` or `offset_xy_m`) or basket cavity. Requires selected arm alone holding, native support, released hands and continuous one-second stability. Table placements additionally require upright posture. Fixed legacy macro retained.
- `run_policy`: `pi05` or `act` with the declared atomic handover capability;
  matching checkpoint, 16-D action/state and three-camera contracts are required.
  A configured endpoint is verified on connection. No fallback to another model.

Native TAMP uses measured robot state, IK and collision search. Geometry comes
from simulator truth and is labelled **oracle**. Motion commands preserve the
other arm and grippers, account for a held bottle, and reach physics through
actuator targets. Planning uses scratch MjData; live objects are never welded,
teleported or assigned qpos. Candidate TCP offset `[0,.006,.075]` is not a measured
physical calibration. Unsupported objects, poses, starts and controllers fail
explicitly. cuRobo is connected for empty-gripper Cartesian moves only.
The ACT client accepts a compatible declared endpoint; no ACT checkpoint is
silently promoted from a smoke training run.

## Execution loop

The LLM publishes a task-specific plan and starts exactly one skill. Completion
is matched by epoch and execution ID; it records feedback and resumes the same
goal with fresh views. The model may select another skill, recover or submit a
camera verdict. Limits are 8 motion starts and 32 model requests per task, with
per-request and planner deadlines. An identical failed request from unchanged
measured robot state is blocked. Cancellation records the result and suppresses
further automatic decisions. Restart never resumes motion.

Full-task simulator scoring remains independent of the model camera verdict.
`primitive_success` and execution exit are not a complete-goal score. An epoch
that has exposed oracle feedback cannot run a learned policy until reset.
Text-only models may route TAMP but must report camera results as `uncertain`.
The terminal visual-review turn is an isolated blind assessment: it receives
the current synchronized head/left/right images and a whitelist of robot
proprioception, with no earlier chat, durable memory, run-status or simulator
oracle tools. It can only `observe` or `finish`; simulator scoring remains
private to the evaluator. This prevents oracle leakage during visual review,
but does not provide object detection, camera-to-world calibration, or visual
closed-loop manipulation. Ambiguous images must remain `uncertain`.
Replanning preserves the root execution lineage and completed stages. Stage IDs
are resolved when a model renumbers a recovery plan. Each logical stage binds to
its actual execution receipt; retries replace a failed attempt, and a successful
first move cannot complete a later repeated move. `occurrence` labels the logical
repetition, rather than counting failed attempts. Physics is paused between skills during
LLM decisions; API wall time is logged. These checks are not real-time transfer
tests. Existing policy delay diagnostics remain a separate timing protocol.

For new experiments, `tools/collect_run.py` is the read-only result collector.
Pass the root turn ID and a new folder under `runs/`; it supports both absolute
and relative motion, native execution receipts, model API waits, and failed or
successful runs. It validates the local database schema before writing, records
token use and timing, and links raw transitions/manifests instead of copying
large arrays. It never submits, cancels, resets, or restarts a task:

```bash
python3 -B tools/collect_run.py --turn-id <root-turn-id> --output runs/YYYYMMDD-topic
```

The default source is the 8767 workbench and its read-only `harness/state/objects.sqlite3` journal.
Provider cost remains `unknown` unless that provider returns pricing metadata.

## Memory

SQLite (`state/agent.sqlite3`) preserves user data through additive migrations.
Memory is supplied as labelled reference data, not inserted into system rules.
Model input uses decision summaries with outcomes, failure diagnostics and
evidence references; full feedback and robot-state transitions stay in SQLite
and the experiment archive, outside the default memory panel. Memory input size and provider usage/cache counters are
recorded in the call trace. Camera views are not altered by this summarization.
Compatibility hashes the physical/control/camera resources and policy config;
test files, UI and memory implementation edits do not themselves invalidate
experiences. Older context formats remain historical, without automatic
promotion. Full source hashes are still archived for experiment provenance.

1. Working state: fresh robot observations, executed skills and failed attempts.
2. Object beliefs: model-authored ID/state/pose/confidence with observation ID,
   evidence kind and environment contract. Poses become unknown after motion;
   scene reset expires them. Model confidence is not sensor verification.
3. Experiences: goal, arguments, typed outcome, failure reason and source.
   Outcomes are skill success, execution failure, cancellation, rejected call,
   or unverified. A model claim and policy exit never become scored success.
   Retrieval filters role, environment/policy contract and evidence privileges
   **before** candidate limits. Success/failure banks each retain up to two
   relevant examples within the five-experience budget; spare slots are filled
   by relevance and recency. `recall_memory` accepts a skill filter. Recipes
   remain limited to three. This is bounded lexical retrieval, not an embedding
   or vision retrieval system. Stale notes cannot evict preferences.
4. Recipes: a completed pick → handover → place chain is stored only after the
   native evaluator confirms the complete goal. Recipes contain symbolic skill
   calls and preconditions, not old joint trajectories or object coordinates.
   They remain oracle-validated simulation examples and require re-grounding.

Oracle histories/recipes are excluded by default. `recall_memory` can use a query;
requesting oracle histories explicitly taints the current epoch. Preferences are
user-managed; edits update the original entry atomically without creating a
duplicate or changing another role's notes. Memory changes wait for the current
request or robot operation to end. Scene notes, hypotheses, sensor execution and oracle results remain
distinct. Current VLA requests contain images, state and task, not these notes;
changing this memory system does not train policy weights.

Retrieval returns an explicit evidence policy: an empty filtered result does not
prove pre/post evidence was never stored. A privileged TAMP audit can request
`include_oracle=true`; it must not rerun motion just to replace omitted records.

`list_skills` and memory include recorded capability profiles, separated by
skill/backend and compatible policy configuration. They show counts, failure
codes and an observed skill-success fraction only when scored receipts exist.
Cancellations, rejected calls and unknown policy exits are not in that fraction.
This is not a benchmark, a prediction for a new state or support for novel objects.
Public memory panels label oracle sources; model retrieval excludes them until
explicit privileged recall. The panel can filter outcomes and inspect the
before/after camera evidence without expanding the main page.

Clear removes working copies and chat reference history as well as durable
memory. Deleting an individual experience prunes derived recipe validation
references while preserving other roles, observations and raw experiment files.
Deletion waits for the current request/operation to end to prevent in-flight
feedback from restoring cleared memory.

Each completed skill stores its pre/post robot observation IDs and state,
immutable execution result, archive reference and head/left/right camera
keyframes. Overview debug views do not become model-camera evidence. Failed
TAMP preflight attempts also retain a run, state and camera evidence, explicitly
reporting zero actuator transitions when planning never began. Completion
receipts are frozen by the physics owner before resets or other operations;
an old completion is archived without resuming a new scene or publishing stale
object beliefs. These simulated cameras still require physical calibration.

## Feedback and verification

`.venv/bin/python -B tools/check_agent.py` (run from project root) uses a local
deterministic API provider and **real MuJoCo** to verify three independent skill
chains and both relative/absolute TCP motion. Results, skill execution IDs,
physics recordings and an HTML report are saved in one `runs/*-harness-check/`
folder. It also checks an out-of-workspace goal, recall of its failure, a revised
relative-motion plan, successful arrival and recall of camera-backed success.
It tests the contract, not an external LLM's reasoning quality.

Unit checks: `.venv/bin/python -B -m unittest harness.memory_checks harness.skill_checks
harness.checks harness.workspace_checks harness.model_checks sim.tamp_checks
sim.policy_checks sim.view_checks -q` from the project root (one command).

## Design references

- [Pigey](https://github.com/lianegalanti/Pigey): one-skill-per-decision loop,
  scene working state and failure feedback. No claim of benchmark reproduction.
- [Harness VLA / RPent](https://github.com/RLinf/RPent): memory-guided retryable
  frozen skills; task/global experience and frozen evaluation memory. No import
  of its task-specific benchmark memories into OpenArm.
- [RoboHarness dual memory](https://github.com/LZY-1021/RoboHarness/blob/main/src/soma_memory.py):
  separate success/failure banks. Our retrieval uses typed outcomes and local
  lexical ranking, rather than claiming its multimodal vector-memory implementation.
- [MoveIt Task Constructor](https://moveit.github.io/moveit_task_constructor/):
  compositional stages, state interfaces and separate geometric feasibility.
- [DynaMem](https://github.com/hello-robot/stretch_ai/blob/main/docs/dynamem.md):
  dynamic memory updates; we use a lightweight object-belief layer rather than
  claiming a reconstructed RGB-D map.
- [Voyager](https://github.com/MineDojo/Voyager): skill retrieval and reuse;
  robot recipes here are typed and validated, never arbitrary generated code.
- [TiPToP limitations](https://tiptop-robot.readthedocs.io/en/latest/limitations/):
  object geometry, table assumptions and container cavities need explicit
  treatment; generic grasp/stacking capability cannot be inferred from a name.

Remote company files remain read-only. This adapter controls simulation only;
simulator success does not prove physical transfer or replace calibration.

## Object targets and scene facts

`set_plan.placement_goal` anchors an absolute table/basket point or a table offset
from the bottle at the user-goal start. Recovery can change the arm or path, never
this target. `place` materializes that point; native support, release, footprint
and dwell independently evaluate the goal. The legacy basket benchmark is separate.
Scene memory uses the verified receipt's actual `held_by` and placement surface;
it never assumes right-pick/left-receive/basket. Motion invalidates old scene facts.
`set_gripper` independently opens, partly opens or closes either simulated jaw.
Encoder settling is not grasp evidence. Opening a held object may drop it; use
`place` for verified support. Parameterized skills do not terminate merely because
the old basket benchmark passes; invalid physics and the task-time budget still stop.

The source-MIT arithmetic candidate is offline-only and explicitly configured.
Matching exported software is not proof of capture-time firmware or physical
calibration. The default platform controller remains unchanged.

Placement withdrawal checks that even a closed jaw clears the released bottle
in scratch geometry, then rechecks the actual reached pose. All executed
withdrawal paths still use normal open-jaw native collision checks. A jaw command
reports actual post-command hold/touching state, so an accidental new contact is
visible to the model. This is a simulation geometry check, not calibrated jaw
opening or real force validation.

Finishing an oracle primitive plan requires all explicit skill receipts, or a
recorded failure explaining early termination. Pending unattempted skills cannot
be skipped with an uncertain verdict. This gate does not read hidden VLA
benchmark events; post-execution camera uncertainty remains allowed.

The optional `poses` adapter in the8767 object workbench generates radial
primitive grasp candidates for known non-upright native objects. A locked table
goal provides scratch-only endpoint/lowering lookahead at the original precise
IK tolerance. A confirmed pick receipt can provide an absolute object yaw hint
to `resolve_placement_transit`; that tool still uses the fresh measured grasp
transform and starts no motion. The model explicitly executes its cuRobo request
and subsequently uses native `place` verification. Only fixed block/bottle
development cases have completed this full API route; cylinder cases remain
failed. No learned GraspGenX model is installed or inferred here.

`pick.hold_seconds` optionally requests2–10seconds of native validation with
unchanged actuator commands and the existing2s/.001m/.02rad relative-pose audit.
The skill cannot return success before that requested validation completes.
Initial geometric `set_plan` performs the same fresh native inspection gate
internally; it remains an explicit simulator-oracle operation. A trusted
unheld-object overlap at the original.5mm planning margin is saved with the
original task. Further motion in that task rejects, while fresh-observation
`finish failure/uncertain` is permitted without inventing a run or receipt.
The corresponding original manipulation goal remains failed.

Native non-upright pick methods have explicit start-orientation conditions in
both the bounded default index and on-demand detail. Model hypotheses cannot
be promoted, and forgotten IDs remain forgotten. Raw receipts remain archived
for diagnosis rather than entering either default model memory or user cards.
See `runs/20261009-poses/index.html` for all failed versions, the original API
collector field error and its separate evidence-only correction.
