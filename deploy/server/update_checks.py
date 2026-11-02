"""Deployment fault injection and admission checks; no GPU, API or SSH calls."""
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import Mock

from deploy.server.gateway import Gateway
from deploy.server.update import Updater, atomic_json, exclusive
from sim.curobo_server import planner_config


class UpdateChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.updater=Updater(self.root)
        self.updater.home.mkdir(parents=True)
        self.old=self.updater.releases/('a'*40);self.new=self.updater.releases/('b'*40)
        for path in (self.old,self.new):
            path.mkdir(parents=True);atomic_json(path/'DEPLOYMENT.json',{'source_commit':path.name})
        self.updater.point(self.old)
        self.health={'state':'ready','source_current':True,'policy_ready':True,'curobo_ready':True,
                     'task_active':False,'agent_busy':False,'learning_active':False}
        for name,value in [('fetch',self.new.name),('health',self.health),('stage',self.new),
                           ('preflight',None),('resources',None),('manage',None),('wait_ready',self.health),('git',b'')]:
            setattr(self.updater,name,Mock(return_value=value))

    def test_running_experiment_defers_without_staging_or_stopping(self):
        for field in ('task_active','agent_busy','learning_active'):
            self.updater.health.return_value={**self.health,field:True}
            self.assertEqual(self.updater.once()['phase'],'waiting_for_idle')
        self.updater.manage.assert_not_called();self.updater.stage.assert_not_called()

    def test_unknown_task_state_is_not_treated_as_idle(self):
        self.updater.health.return_value={'state':'ready'}
        self.assertEqual(self.updater.once()['phase'],'waiting_for_idle')
        self.updater.manage.assert_not_called()

    def test_inflight_gateway_request_defers_activation(self):
        with exclusive(self.updater.home/'admission.lock'):
            self.assertEqual(self.updater.once()['phase'],'waiting_for_requests')
        self.assertEqual(self.updater.active(),self.old);self.updater.manage.assert_not_called()

    def test_preflight_failure_quarantines_revision_without_restart(self):
        self.updater.preflight.side_effect=RuntimeError('dependency change')
        result=self.updater.once();self.assertEqual(result['blocked_commit'],self.new.name)
        self.assertEqual(self.updater.once()['phase'],'blocked')
        self.assertEqual(self.updater.preflight.call_count,1);self.updater.manage.assert_not_called()

    def test_validated_release_is_activated_and_previous_is_retained(self):
        result=self.updater.once()
        self.assertEqual(result['active_commit'],self.new.name);self.assertEqual(result['previous_commit'],self.old.name)
        self.assertEqual(self.updater.active(),self.new);self.assertTrue(self.old.is_dir())
        self.assertEqual([tuple(call.args) for call in self.updater.manage.call_args_list],[(self.old,'stop'),(self.new,'start')])

    def test_startup_failure_rolls_back_and_blocks_the_bad_commit(self):
        self.updater.wait_ready.side_effect=[RuntimeError('new release not ready'),self.health]
        result=self.updater.once()
        self.assertEqual(result['phase'],'rolled_back');self.assertEqual(result['blocked_commit'],self.new.name)
        self.assertEqual(self.updater.active(),self.old)
        self.assertEqual([tuple(call.args) for call in self.updater.manage.call_args_list],
                         [(self.old,'stop'),(self.new,'start'),(self.new,'stop'),(self.old,'start')])

    def test_interrupted_activation_recovers_before_fetching(self):
        self.updater.point(self.new)
        atomic_json(self.updater.status_file,{'phase':'error','previous_root':str(self.old),'candidate':str(self.new)})
        self.assertEqual(self.updater.once()['phase'],'rolled_back')
        self.updater.fetch.assert_not_called();self.assertEqual(self.updater.active(),self.old)

    def test_unchanged_main_does_not_restart_and_unhealthy_is_not_success(self):
        self.updater.fetch.return_value=self.old.name
        self.assertEqual(self.updater.once()['phase'],'up_to_date');self.updater.manage.assert_not_called()
        self.updater.health.return_value={**self.health,'source_current':False}
        with self.assertRaisesRegex(RuntimeError,'健康检查'):self.updater.once()

    def test_rewritten_history_never_stops_the_current_release(self):
        self.updater.git.side_effect=RuntimeError('not an ancestor')
        with self.assertRaisesRegex(RuntimeError,'ancestor'):self.updater.once()
        self.updater.manage.assert_not_called();self.assertEqual(self.updater.active(),self.old)

    def test_dependency_change_requires_explicit_environment_work(self):
        for path,value in ((self.old,b'old'),(self.new,b'new')):(path/'requirements.txt').write_bytes(value)
        command=self.updater.command=Mock()
        with self.assertRaisesRegex(RuntimeError,'依赖清单'):Updater.preflight(self.updater,self.new,self.old)
        command.assert_not_called()

    def test_release_paths_cannot_escape_the_owned_folder(self):
        for value in (self.root.parent/'other',self.updater.releases/'../other',self.updater.home/'other'):
            with self.assertRaises(RuntimeError):self.updater.validate_release(value)

    def test_staging_preserves_host_runtime_and_shared_records_and_detects_tampering(self):
        (self.root/'sim').mkdir();(self.root/'sim/learning.json').write_text('{"python":"host-python"}')
        data=io.BytesIO()
        with tarfile.open(fileobj=data,mode='w') as archive:
            body=b'{"python":"developer-python"}';item=tarfile.TarInfo('sim/learning.json');item.size=len(body)
            archive.addfile(item,io.BytesIO(body))
        self.updater.git.side_effect=lambda *args: b'' if args[0]=='ls-tree' else data.getvalue()
        release=Updater.stage(self.updater,'c'*40)
        self.assertEqual((release/'sim/learning.json').read_bytes(),(self.root/'sim/learning.json').read_bytes())
        (release/'runs/record.json').write_text('preserved')
        self.updater.prune({self.old,self.new})
        self.assertEqual((self.root/'runs/record.json').read_text(),'preserved')
        release=Updater.stage(self.updater,'c'*40)
        (release/'sim/learning.json').write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'已被修改'):Updater.stage(self.updater,'c'*40)

    def test_manager_controls_the_explicit_release_without_current_redirect(self):
        self.updater.command=Mock(return_value=b'{}')
        Updater.manage(self.updater,self.old,'start')
        call=self.updater.command.call_args
        self.assertEqual(call.kwargs['env']['OPENARM_MANAGER_DIRECT'],'1')
        self.assertIn(self.old/'deploy/server/manage.py',call.args[0])

    def test_supervisor_uses_validated_active_updater_with_shared_state_root(self):
        path=self.old/'deploy/server/update.py';path.parent.mkdir(parents=True)
        path.write_text('class Updater:\n    def __init__(self, root): self.root=root\n')
        atomic_json(self.old/'DEPLOYMENT.json',{'files':{'deploy/server/update.py':hashlib.sha256(path.read_bytes()).hexdigest()}})
        self.assertEqual(self.updater.worker().root,self.root)
        path.write_text('raise RuntimeError("must not execute modified code")')
        with self.assertRaisesRegex(RuntimeError,'校验记录不符'):self.updater.worker()

    def test_archive_rejects_traversal_and_symlinks_and_cleans_its_stage(self):
        for kind in ('traversal','symlink'):
            data=io.BytesIO()
            with tarfile.open(fileobj=data,mode='w') as archive:
                item=tarfile.TarInfo('../escape' if kind=='traversal' else 'link')
                if kind=='symlink':item.type=tarfile.SYMTYPE;item.linkname='/etc'
                archive.addfile(item)
            self.updater.git.side_effect=lambda *args: b'' if args[0]=='ls-tree' else data.getvalue()
            with self.assertRaises(RuntimeError):Updater.stage(self.updater,'c'*40)
            self.assertFalse((self.root/'escape').exists())
            self.assertFalse(list(self.updater.releases.glob('stage-*')))

    def test_submodule_pin_and_missing_assets_block_before_staging(self):
        self.updater.git.return_value=b'160000 commit '+b'd'*40+b'\topenarm_mujoco\n'
        with self.assertRaisesRegex(RuntimeError,'尚未核验'):Updater.stage(self.updater,'c'*40)
        self.updater.config['submodules']={'openarm_mujoco':{'revision':'d'*40,'files':{'v1/model.xml':'x'}}}
        with self.assertRaisesRegex(RuntimeError,'资源缺失'):Updater.stage(self.updater,'c'*40)
        self.assertFalse((self.updater.releases/('c'*40)).exists())

    def test_verified_submodule_assets_are_copied_inside_the_release(self):
        source=self.root/'openarm_mujoco/v1/model.xml';source.parent.mkdir(parents=True);source.write_text('<mujoco/>')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        self.updater.config['submodules']={'openarm_mujoco':{'revision':'d'*40,'files':{'v1/model.xml':digest}}}
        data=io.BytesIO()
        with tarfile.open(fileobj=data,mode='w') as archive:
            item=tarfile.TarInfo('openarm_mujoco');item.type=tarfile.DIRTYPE;archive.addfile(item)
        self.updater.git.side_effect=lambda *args:b'160000 commit '+b'd'*40+b'\topenarm_mujoco\n' if args[0]=='ls-tree' else data.getvalue()
        release=Updater.stage(self.updater,'c'*40)
        target=release/'openarm_mujoco/v1/model.xml'
        self.assertTrue(target.resolve().is_relative_to(release));self.assertEqual(target.read_bytes(),source.read_bytes())
        source.write_text('mutated source');self.assertEqual(target.read_text(),'<mujoco/>')

    def test_planner_paths_follow_the_release_and_reject_outside_files(self):
        root=self.root/'portable-release';folder=root/'task/planners';folder.mkdir(parents=True)
        (folder/'openarm.urdf').write_text('<robot/>')
        path=folder/'openarm.json'
        atomic_json(path,{'robot_cfg':{'kinematics':{'urdf_path':'task/planners/openarm.urdf'}}})
        self.assertEqual(planner_config(root)['robot_cfg']['kinematics']['urdf_path'],str(folder/'openarm.urdf'))
        atomic_json(path,{'robot_cfg':{'kinematics':{'urdf_path':'../outside.urdf'}}})
        with self.assertRaisesRegex(ValueError,'inside the current project'):planner_config(root)


class AdmissionChecks(unittest.TestCase):
    def test_update_lock_rejects_new_post_and_version_never_exposes_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic_json(root/'DEPLOYMENT.json',{'source_commit':'a'*40,'secret':'not-public'})
            server=ThreadingHTTPServer(('127.0.0.1',0),Gateway);server.root=root
            server.auth=Mock();server.auth.profile.return_value=None
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
            def request(method,path):
                connection=HTTPConnection('127.0.0.1',server.server_port,timeout=3)
                connection.request(method,path,body=b'{}' if method=='POST' else None)
                response=connection.getresponse();status=response.status;body=response.read();connection.close()
                return status,body
            with exclusive(root/'.runtime/update/admission.lock'):
                self.assertEqual(request('POST','/api/chat')[0],503)
                server.auth.profile.assert_not_called()
                status,body=request('GET','/version')
                self.assertEqual(status,200);self.assertNotIn(b'not-public',body)
                self.assertEqual(json.loads(body)['source_commit'],'a'*40)
            self.assertEqual(request('POST','/api/chat')[0],401)


if __name__=='__main__':unittest.main()
