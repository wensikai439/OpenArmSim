"""Forward/left/back/right targets in one continuous native scene, no teleports."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sim.environment import ROOT,OpenArmEnv
from sim.relocation import RelocationPlanner
from sim.recording import Recorder,dump


def main():
    out=ROOT/'runs/20261008-relocate/native-chain';out.mkdir(exist_ok=False)
    cases=[('forward',[.08,0]),('left',[0,.05]),('back',[-.06,0]),('right',[0,-.05])]
    dump(out/'protocol.json',{'cases':cases,'arm':'right','scene_reset':'once at initial seed 0/episode 0 only',
        'later_goals':'preserve actual robot/bottle positions and velocities; fresh bounded task clock',
        'budget_s_per_goal':60,'physics':'unchanged','source':'native oracle, no model or hardware',
        'scope':'four declared sequential development targets, not independent random episodes'})
    env=OpenArmEnv(images=True);env.reset(seed=0);results=[]
    try:
        for index,(name,offset) in enumerate(cases):
            if index:
                q,v=env.data.qpos.copy(),env.data.qvel.copy()
                env.prepare_task()
                assert np.array_equal(q,env.data.qpos) and np.array_equal(v,env.data.qvel)
            env.set_task_budget(60);initial=env.data.xpos[env.bottle_id].copy();start=env.tick
            rec=Recorder(env,'tamp',root=out,record_images=False,seed=0,
                         details={'case':name,'offset_xy_m':offset,'state_preserved':bool(index),'actual_API':False})
            planner=RelocationPlanner(env,'relocate_object',parameters={'object_id':'bottle','arm':'right',
                'receptacle_id':'table','offset_xy_m':offset})
            status='error';error=None;before=rec.observation(env)
            try:
                iterator=planner.actions()
                while True:
                    q=env.data.qpos.copy()
                    try:action,diag=next(iterator)
                    except StopIteration:status='primitive_completed';break
                    assert np.array_equal(q,env.data.qpos)
                    _,_,term,trunc,info=env.step_tamp(action);after=rec.observation(env)
                    rec.append(before,action,after,info,terminated=term,truncated=trunc,diagnostics=diag);before=after
                    if term or trunc:status='invalid' if info['evaluation']['invalid'] else 'truncated';break
            except Exception as exc:error=str(exc)
            rec.finish(status,env.info(),error=error,metrics={'tamp':planner.result})
            frame=name+'.jpg';Image.fromarray(env.render('overview')).save(out/frame,quality=92)
            row={'case':name,'status':status,'error':error,'initial_bottle_position_m':initial.tolist(),
                'offset_xy_m':offset,'start_tick':start,'end_tick':env.tick,'run':rec.path.name,'frame':frame,
                'feedback':planner.result};results.append(row)
            print(json.dumps({k:row[k] for k in ('case','status','error','start_tick','end_tick')},ensure_ascii=False),flush=True)
            if status!='primitive_completed':break  # do not reset to hide failure
        dump(out/'summary.json',{'results':results,'remaining_cases':len(cases)-len(results),
            'model':env.manifest(),'hardware_access':False,'transfer_ready':False})
    finally:env.close()


if __name__=='__main__':main()
