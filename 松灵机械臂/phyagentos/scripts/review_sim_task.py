"""One DS Flash read-only semantic review of a saved real Forge Task; never applies a verdict."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, time, urllib.request
from pathlib import Path
from PhyAgentOS.config.loader import load_config
from PhyAgentOS.cli.commands import _make_provider
from PhyAgentOS.forge.task import AgentTaskRecord
from PhyAgentOS.agent.session_verifier import ForgeTaskVerifier
from PhyAgentOS.verification.request_builder import VerificationRequestBuilder
from PhyAgentOS.verification.engine import VerificationEngine
from PhyAgentOS.verification.service import FORGE_TASK_PROMPT
from PhyAgentOS.verification.contracts import VerificationVerdict

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def get(path):
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open("http://127.0.0.1:19081"+path, timeout=20) as response: return json.load(response)

async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    def save(name,obj): (args.out/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n")
    cfg=load_config()
    model=cfg.agents.verification.model or cfg.agents.defaults.model
    if cfg.agents.defaults.model not in {"deepseek-flash","custom/deepseek-flash"} or model not in {"deepseek-flash","custom/deepseek-flash"}:
        raise RuntimeError("Only explicitly selected DS Flash may review")
    provider=_make_provider(cfg,model)
    if (provider.api_base or "").rstrip("/") not in {"https://api.deepseek.com","https://api.deepseek.com/v1"} or provider._client.max_retries!=0:
        raise RuntimeError("Expected DS endpoint and zero SDK retries")
    wrapper=json.loads(args.task.read_text())
    task=AgentTaskRecord.model_validate(wrapper.get("task",wrapper))
    before=get("/api/tasks/"+task.task_id)["task"]
    if before!=task.model_dump(mode="json"): raise RuntimeError("Saved task differs from current official Task")
    builder=VerificationRequestBuilder(cfg.workspace_path)
    request=builder.build_agent_task(task,events=wrapper.get("events",[]),lessons="")
    tracked={str(args.task):sha(args.task), **{str(path):sha(path) for path in request.artifact_paths}}
    save("before-task.json",before)
    save("request.json",{"system_prompt":FORGE_TASK_PROMPT,"content":request.content})
    save("source-provenance.json",{"task_id":task.task_id,"task_source":str(args.task),"source_sha256":tracked,
        "valid_evidence_refs":sorted(request.valid_evidence_refs),"review_script_sha256":sha(Path(__file__)),
        "model":model,"provider":"custom","endpoint":provider.api_base,"retry_count":0,
        "path":"Official VerificationRequestBuilder + prompt + VerificationEngine + strict verdict/evidence validation; standalone read-only composition, no child HTTP service"})
    calls=0; response=None; status="failed"; error=None; verdict=None; schema_valid=False
    async def chat_once(**kwargs):
        nonlocal calls,response
        if calls: raise RuntimeError("One review request only; do not replay")
        calls+=1
        kwargs.setdefault("max_tokens",provider.generation.max_tokens)
        kwargs.setdefault("reasoning_effort",provider.generation.reasoning_effort)
        save("request-state.json",{"status":"submitted_once","submitted_at":time.time(),"request_count":calls,"automatic_retries":0})
        response=await provider.chat(**kwargs)
        save("response.json",{"content":response.content,"usage":response.usage,"finish_reason":response.finish_reason})
        save("request-state.json",{"status":"response_received_once","request_count":calls,"automatic_retries":0,"received_at":time.time()})
        return response
    provider.chat_with_retry=chat_once
    try:
        engine=VerificationEngine(provider=provider,model=model,timeout_s=180)
        raw=await engine.complete(system_prompt=FORGE_TASK_PROMPT,content=request.content)
        verdict=VerificationVerdict.model_validate(raw)
        schema_valid=True
        save("parsed-verdict.json",{"accepted":False,"parsed":verdict.model_dump(mode="json")})
        ForgeTaskVerifier._validate_generic_verdict(expected_criteria=task.verification.success_criteria,
            valid_evidence_refs=set(request.valid_evidence_refs),verdict=verdict)
        save("review-verdict.json",verdict.model_dump(mode="json"));status="validated_read_only_review"
    except Exception as exc:
        error={"type":type(exc).__name__,"message":str(exc)}
        save("error.json",error)
    finally:
        await provider._client.close()
        after=get("/api/tasks/"+task.task_id)["task"];save("after-task.json",after)
        unchanged=before==after and all(Path(path).is_file() and sha(Path(path))==value for path,value in tracked.items())
        summary={"status":status,"task_id":task.task_id,"model":model,"model_requests":calls,
            "usage":response.usage if response else None,"review_verdict":verdict.verdict if verdict else None,
            "strict_schema_passed":schema_valid,"evidence_reference_validation_passed":status=="validated_read_only_review",
            "schema_and_evidence_validation_passed":status=="validated_read_only_review","original_task_and_artifacts_unchanged":unchanged,
            "original_official_status":after["status"],"original_official_verdict":after.get("verdict"),
            "verdict_applied":False,"actions_dispatched":0,"runtime_restarted":False,
            "new_expert_rollouts":0,"child_HTTP_path_tested":False,"error":error}
        save("summary.json",summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in {"original_official_verdict","error"}},ensure_ascii=False),flush=True)
        if not unchanged: raise RuntimeError("Source changed during review; investigate before further work")
    return 0 if status=="validated_read_only_review" else 1

if __name__=="__main__": raise SystemExit(asyncio.run(main()))
