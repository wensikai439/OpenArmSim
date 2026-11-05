"""Forge endpoint for an already-owned Runtime; never starts hardware services."""
from __future__ import annotations
import asyncio,base64,dataclasses,json,os,queue,sys,threading,time,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from dora import Node
from forge_msgs import CompressedImage
from forge_tool import (TOOL_ENDPOINT_PROTOCOL,ToolEndpointDescriptor,ToolOperationDescriptor,
    ToolEndpointHandler,ToolAccepted,ToolResult,ToolError,ToolEndpointError,
    ToolExecutionStatus,ToolResultResponse,ToolControlResponse,ToolEvent)
from forge_tool._tool_message import ToolMessage
from forge_tool.dora import DoraToolEndpointBinding,tool_message_to_envelope,tool_envelope_to_message
from endpoint_lease import EndpointLease
from runtime_bridge import Bridge,BridgeError,digest

class Endpoint:
    def __init__(self,node=None,bridge=None):
        self.node,self.bridge=node,bridge or Bridge();self.active=None;self.records={};self.done=queue.Queue()
    def images(self,result):
        if self.node is None:return
        for frame in result.get('frames',[]):
            raw=Path(frame['path']).read_bytes()
            if digest(raw)!=frame['sha256']:raise BridgeError('Recorded native image SHA changed')
            self.node.send_output('image/'+frame['camera'],CompressedImage(format='png',data=raw).to_arrow())
    async def query(self,request,context):
        try:
            if context.operation=='capabilities': result=self.bridge.capabilities()
            elif context.operation=='state': result=await asyncio.to_thread(self.bridge.state)
            elif context.operation=='prepare_task': result=await asyncio.to_thread(self.bridge.prepare_task,**request.arguments)
            elif context.operation=='observe':
                result=await asyncio.to_thread(self.bridge.observe);self.images(result)
            elif context.operation=='recover':
                result=await asyncio.to_thread(self.bridge.recover,request.arguments['receipt_reference'])
                if result.get('phase') in {'completed','failed','cancelled'} and result.get('native',{}).get('execution_rights')=='available' and self.active is not None and self.records[self.active]['reference']==request.arguments['receipt_reference']: self.active=None
            else:raise BridgeError('Unsupported query')
        except Exception as exc:raise ToolEndpointError(ToolError('REAL_RUNTIME_UNAVAILABLE',str(exc))) from exc
        return ToolResult(status='succeeded',outputs=result)
    async def start(self,request,context,events):
        operation=getattr(context,'operation','run_task')
        is_cancel=operation=='cancel_task'
        if self.active is not None and not is_cancel:raise ToolEndpointError(ToolError('BUSY','Owned execution is running or unresolved'))
        key=context.execution_key;ref=key.invocation_id+':'+key.attempt_id
        if not is_cancel:
            try:self.bridge.preflight(request.arguments['prepared_id'],ref)
            except Exception as exc:raise ToolEndpointError(ToolError('FIELD_RELEASE_REQUIRED',str(exc))) from exc
        record={'phase':'running','result':None,'cancel':threading.Event(),'events':events,
                'operation':operation,'reference':request.arguments['receipt_reference'] if is_cancel else digest(ref.encode())}
        self.records[key]=record
        if not is_cancel:self.active=key
        def work():
            try:
                output=self.bridge.cancel_owned(record['reference']) if is_cancel else self.bridge.execute(request.arguments['prepared_id'],ref,record['cancel'])
                phase=output['phase'];status={'completed':'succeeded','failed':'failed','cancelled':'cancelled','unknown':'unknown'}[phase]
                error=ToolError('NATIVE_OUTCOME_UNKNOWN' if status=='unknown' else 'NATIVE_TASK_FAILED',str(output.get('error') or output.get('native',{}).get('error') or phase)) if status in {'failed','unknown'} else None
                result=ToolResult(status=status,outputs=output,error=error)
            except Exception as exc:
                path=self.bridge.run/'attempts'/(record['reference']+'.json')
                output=json.loads(path.read_text()) if path.exists() else {}
                unresolved=output.get('phase') in {'dispatch_pending','running','unknown'}
                if unresolved:
                    output.update(phase='unknown',error=str(exc))
                    from runtime_bridge import save
                    save(path,output)
                result=ToolResult(status='unknown' if unresolved else 'failed',outputs={**output,'error':str(exc),'business_success':None},error=ToolError('NATIVE_OUTCOME_UNKNOWN' if unresolved else 'PRE_DISPATCH_FAILED',str(exc)))
            self.done.put((key,result))
        threading.Thread(target=work,name='paos-real-task',daemon=False).start()
        return ToolAccepted(details={'receipt_reference':record['reference'],'native_acceptance':'pending','business_success':None})
    async def drain(self):
        while not self.done.empty():
            key,result=self.done.get_nowait();r=self.records[key];r.update(result=result,phase={'succeeded':'completed','failed':'failed','cancelled':'cancelled','unknown':'unknown'}[result.status])
            for name in ['before','after']:self.images(result.outputs.get(name,{}))
            if result.status!='unknown' and self.active==key:self.active=None
            if result.status in {'succeeded','failed','cancelled'}:
                await r['events'].emit(ToolEvent(type={'succeeded':'executor_completed','failed':'executor_failed','cancelled':'cancelled'}[result.status],data={'receipt_reference':r['reference'],'business_success':None}))
    async def status(self,key):
        r=self.records.get(key)
        if r is None:raise ToolEndpointError(ToolError('NOT_FOUND','Unknown execution'))
        details={'receipt_reference':r['reference'],'business_success':None}
        path=self.bridge.run/'attempts'/(r['reference']+'.json')
        if path.exists():
            native=json.loads(path.read_text()).get('native',{})
            details.update({k:native.get(k) for k in ['task_id','task_state','execution_rights','pending_confirmation','release_evidence']})
        return ToolExecutionStatus(phase=r['phase'],error=r['result'].error if r['result'] else None,details=details)
    async def result(self,key):
        r=self.records.get(key)
        return ToolResultResponse(status='not_found' if r is None else 'available' if r['result'] else 'pending',result=r['result'] if r else None)
    async def cancel(self,key,reason=None):
        r=self.records.get(key)
        if r is None:return ToolControlResponse(command='cancel',status='rejected',error=ToolError('NOT_FOUND','No owned task'))
        if r.get('operation')=='cancel_task':return ToolControlResponse(command='cancel',status='unsupported')
        if r['result'] is not None:return ToolControlResponse(command='cancel',status='terminal')
        r['cancel'].set();r['phase']='stopping'
        return ToolControlResponse(command='cancel',status='accepted',details={'native_cancel':'pending','physical_stop':'unknown'})

def main():
    node=Node();endpoint=Endpoint(node)
    descriptor=ToolEndpointDescriptor(protocol_version=TOOL_ENDPOINT_PROTOCOL,endpoint_id='piper.real_runtime',operations=(
        *(ToolOperationDescriptor(name=op,semantics='query') for op in ['capabilities','state','observe','prepare_task','recover']),
        ToolOperationDescriptor(name='run_task',semantics='action',status_supported=True,cancellable=True),
        ToolOperationDescriptor(name='cancel_task',semantics='action',status_supported=True,cancellable=False)))
    instance='paos-real-adapter-'+uuid.uuid4().hex[:12]
    handler=ToolEndpointHandler(descriptor,endpoint_instance_id=instance,operations={op.name:endpoint for op in descriptor.operations})
    async def sink(batch):node.send_output('tool_out',batch)
    binding=DoraToolEndpointBinding(handler,event_sink=sink);lease=EndpointLease(descriptor,instance);runner=asyncio.Runner()
    def send(envelope):node.send_output('tool_out',tool_envelope_to_message(envelope).to_arrow())
    try:
        for event in node:
            if event['type'] in {'STOP','ERROR'}:break
            registration=lease.poll(time.monotonic())
            if registration:send(registration)
            runner.run(endpoint.drain())
            if event['type']=='INPUT' and event['id']=='tool_in':
                envelope=tool_message_to_envelope(ToolMessage.from_arrow(event['value']))
                if envelope.message_type=='endpoint.registry.response':lease.acknowledge(envelope,time.monotonic())
                else:
                    for batch in runner.run(binding.dispatch_input(event['value'])):runner.run(sink(batch))
    finally:
        # Shutdown does not terminate/unlink any external Runtime or camera process.
        for r in endpoint.records.values():
            if r['result'] is None:r['cancel'].set()
        unregister=lease.stop()
        if unregister:send(unregister)
        runner.close()
if __name__=='__main__':main()
