"""Independent8005world-cache candidate; production8004is unchanged."""
import hashlib
import logging
from http.server import HTTPServer
from pathlib import Path

from tools.scene_gpu import Planner as OriginalPlanner,Handler as OriginalHandler
from sim.scene_cache import SceneCacheMixin
from sim.environment import ROOT


class Planner(SceneCacheMixin,OriginalPlanner):
    def __init__(self):
        super().__init__()
        self.metadata.update(world_cache_candidate=True,
            scene_cache_worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            exact_world_cache_source_sha256=hashlib.sha256((ROOT/'sim/scene_cache.py').read_bytes()).hexdigest(),
            world_cache_contract='same topology; exact official cuboid updates and independent GPU storage/pointer checks',
            worker_port=8005,gpu_context_healthy=True)

    def discard(self):
        if self.cached:self.cached[1].destroy()
        self.cached=None;self.cached_locks=None;self.refresh_fk=None;self.warmed_generation=0

    def plan(self,data):
        if not self.metadata['gpu_context_healthy']:raise RuntimeError('candidate GPU context failed; owner restart required')
        try:
            result=super().plan(data)
            # Failed-plan diagnostics can change captured solver buffers.
            # Never carry that state into a later normal planning query.
            if result.get('success') is not True:
                self.discard();result['cache']['invalidated_after_failed_plan']=True
            return result
        except Exception as exc:
            text=str(exc).lower()
            if 'cuda' in text or 'illegal memory access' in text or 'device-side assert' in text:
                self.metadata['gpu_context_healthy']=False
                self.metadata['gpu_failure']=str(exc)
            else:self.discard()
            raise


class Handler(OriginalHandler):
    def do_GET(self):
        if not self.server.planner.metadata['gpu_context_healthy']:
            return self.reply(503,{'error':'candidate GPU context failed; owner restart required'})
        return super().do_GET()
    def do_POST(self):
        if not self.server.planner.metadata['gpu_context_healthy']:
            return self.reply(503,{'error':'candidate GPU context failed; no planning performed'})
        return super().do_POST()


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    server=HTTPServer(('127.0.0.1',8005),Handler)
    try:
        server.planner=Planner()
        print('World cache candidate · 127.0.0.1:8005 · no hardware',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if getattr(server,'planner',None) and server.planner.cached:server.planner.cached[1].destroy()
        server.server_close()


if __name__=='__main__':main()
