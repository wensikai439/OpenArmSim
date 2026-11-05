"""Targeted offline command-boundary checks; no robot, camera, model or Forge calls."""
import asyncio,math
from pathlib import Path
from types import SimpleNamespace
import httpx,pytest
from starlette.applications import Starlette
from starlette.routing import Route
from console.real_commands import validate_step,archive_index,video,RealCommands,RelativeTool

def test_numeric_relative_arrays():
    assert validate_step(right_delta=[3,0,0,0,0,0])['right_delta'][0]==3
    for value in ('3,0,0,0,0,0',[0]*5,[0,0,0,0,0,float('nan')]):
        with pytest.raises(ValueError):validate_step(right_delta=value)

def test_gripper_and_ik_exclusive():
    with pytest.raises(ValueError):validate_step(left_gripper=.2)
    with pytest.raises(ValueError):validate_step(right_delta=[0]*6,translation_m=[0,0,.01])
    with pytest.raises(ValueError):validate_step(translation_m=[0,0,.07])
    assert validate_step(left_gripper=.094)['left_gripper']==.094

def test_no_motion_without_frontend_scope():
    c=object.__new__(RealCommands)
    c.current=Path('/test/current');c.owner=SimpleNamespace(job={'real_command_run':'/test/current','allow_motion':False})
    c.cancel_requested=False;c.halted=False
    with pytest.raises(ValueError):c.require_job(motion=True)
    c.owner.job['allow_motion']=True;c.require_job(motion=True)
    c.cancel_requested=True
    with pytest.raises(ValueError):c.require_job(motion=True)

def test_stop_or_failed_stage_cannot_continue():
    c=object.__new__(RealCommands)
    c.current=Path('/test/current');c.owner=SimpleNamespace(job={'real_command_run':'/test/current','allow_motion':True})
    c.cancel_requested=False;c.halted=True
    with pytest.raises(ValueError):c.require_job(motion=True)

@pytest.mark.skipif([len(g["files"]) for g in archive_index()] != [10,8,5], reason="Requires external historical 23-recording archive; not shipped")
def test_archive_exact_selected_groups():
    groups=archive_index()
    assert [len(g['files']) for g in groups]==[10,8,5]
    assert sum(len(g['files']) for g in groups)==23

@pytest.mark.skipif(not any(g["files"] for g in archive_index()), reason="Requires external recording for HTTP Range integration check")
def test_http_video_range_and_traversal():
    async def run():
        app=Starlette(routes=[Route('/video',video)])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            first=archive_index()[0]['files'][0]
            url=first['url'].replace('/api/real-command/video','/video')
            res=await client.get(url,headers={'Range':'bytes=0-31'})
            assert res.status_code==206 and len(res.content)==32
            res=await client.get('/video',params={'group':'console-demo-20261024','path':'../../config/real_runtime.json'})
            assert res.status_code==404
            res=await client.get('/video',params={'group':'old-unselected-run','path':'anything.mp4'})
            assert res.status_code==404
    asyncio.run(run())


def test_failed_action_locks_following_actions():
    async def failed(**kwargs):raise ValueError('invalid stage')
    c=SimpleNamespace(halted=False,action=failed)
    result=asyncio.run(RelativeTool(c).execute())
    assert c.halted and 'invalid stage' in result


def test_unknown_submission_marker_survives_process_cleanup(tmp_path,monkeypatch):
    import console.real_commands as module
    import json
    monkeypatch.setattr(module,'RUNS',tmp_path)
    run=tmp_path/'command-uncertain';run.mkdir()
    (run/'command.json').write_text('{}')
    marker={'run':run.name,'phase':'unknown','invocation_id':None,'do_not_resubmit':True}
    (tmp_path/'active.json').write_text(json.dumps(marker))
    async def process(*args,**kwargs):return 'transport failed'
    c=object.__new__(RealCommands)
    c.current=run;c.invocation=None;c.cancel_requested=False;c.halted=True
    c.owner=SimpleNamespace(process=process)
    asyncio.run(c.process('text','session','request'))
    assert json.loads((tmp_path/'active.json').read_text())==marker


def test_pending_camera_unit_is_reused():
    c=object.__new__(RealCommands)
    c.owner=SimpleNamespace(job=None)
    c.camera_health={'ActiveState':'activating'}
    c.cameras=lambda:{name:{'fresh':False} for name in ('cam_left','cam_right','cam_middle')}
    assert asyncio.run(c._camera('start'))=={'reused':True,'pending':True}
