"""Explicit scene selection + normal official console reset; no motion or automatic replay."""
import os
from pathlib import Path
import argparse,json,hashlib,urllib.request,urllib.error
ROOT=Path(str(Path(__file__).resolve().parents[1]))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
def request(path,body=None):
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 req=urllib.request.Request("http://127.0.0.1:19081"+path,data=json.dumps(body).encode() if body else None,
  headers={"Content-Type":"application/json"} if body else {})
 with op.open(req,timeout=120 if body else 15) as f:return json.load(f)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--scene-config",type=Path,required=True)
 ap.add_argument("--audit-dir",type=Path,required=True);args=ap.parse_args()
 cfg=args.scene_config.resolve();cfg.relative_to(ROOT/"runs")
 manifest=json.loads((cfg.parent.parent/"scenario-manifest.json").read_text())
 row=next(x for x in manifest["scenarios"] if Path(x["path"]).resolve()==cfg)
 digest=hashlib.sha256(cfg.read_bytes()).hexdigest()
 if row["sha256"]!=digest:raise RuntimeError("Immutable scenario manifest SHA mismatch; no reset")
 audit=args.audit_dir.resolve();audit.mkdir(parents=True,exist_ok=False)
 before=request("/api/state");save(audit/"before.json",before)
 # /api/state is the two-second console cache; ownership must use the current official state store.
 from PhyAgentOS.skill_runtime.state import RuntimeStateStore
 if before.get("skill") not in {"piper-dual-sim","rpent-piper-sim"}:raise RuntimeError("PiPER simulation Skill required; no reset")
 live=RuntimeStateStore().load(before["skill"])
 if live is None:raise RuntimeError("Official runtime state absent; no selection or reset")
 rs=live.to_dict();save(audit/"runtime-state-before.json",rs)
 cached=before.get("runtime",{}).get("state",{}) or {}
 if cached.get("runtime_instance_id")!=rs["runtime_instance_id"]:
  raise RuntimeError("Runtime instance changed since console snapshot; no selection or reset")
 if any(before.get(k) for k in ("job","active_task","control_busy","background_busy","reflection_busy")) or any(rs.get(k) for k in ("active_invocations","active_sessions","active_task_bindings")):
  raise RuntimeError("Runtime has active ownership; no selection or reset")
 if rs.get("profile")!="joint":raise RuntimeError("Joint profile required; no reset")
 selection=ROOT/"config/sim_scene_selection.json";selection.parent.mkdir(exist_ok=True)
 if selection.exists():save(audit/"previous-selection.json",json.loads(selection.read_text()))
 value={"config_path":str(cfg),"sha256":digest,"scenario_id":row["scenario_id"],"split":row["split"]}
 temp=selection.with_suffix(".tmp");save(temp,value);temp.replace(selection);save(audit/"selection.json",value)
 try:
  result=request("/api/runtime",{"action":"reset"});save(audit/"reset-response.json",result)
 except Exception as e:
  save(audit/"reset-error.json",{"type":type(e).__name__,"message":str(e),"effects":"Do not replay; reconcile runtime state first."})
  raise
 if result.get("completed") is not True:raise RuntimeError("Reset not confirmed; no motion")
 after=request("/api/state");save(audit/"after.json",after)
 runtime=after.get("runtime",{})
 if not runtime.get("gateway_ready") or not runtime.get("flow_running"):raise RuntimeError("Reset returned but runtime is not ready; no motion")
 print(json.dumps({"normal_reset_completed":True,**value,"runtime_instance_id":runtime.get("state",{}).get("runtime_instance_id")}),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(main())
