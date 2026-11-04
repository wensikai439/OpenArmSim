"""Fill the bundled official handler's missing Session dispatch only.

Admission, invocation identity, replay protection, terminal consistency, retention
and event ordering remain in the official ToolEndpointHandler implementation.
"""
from forge_tool import ToolEndpointHandler, ToolControlResponse, ToolEndpointError
from forge_tool.wire import (
    ToolProtocolError, control_request_from_payload, make_control_response_envelope,
    validate_message_envelope,
)


class SessionEndpointHandler(ToolEndpointHandler):
    async def dispatch(self, request, *, event_sink=None):
        self.validate_request_route(request)
        operation = request.operation
        if self._operation_descriptors[operation].semantics != "session":
            return await super().dispatch(request, event_sink=event_sink)
        try:
            validate_message_envelope(request)
        except ToolProtocolError as error:
            return (self._protocol_error_response(error, request),)
        if request.message_type == "tool.invoke.request":
            return await self._handle_action_invoke(request, operation, event_sink)
        endpoint = self._implementations[operation]
        key = self._execution_key(request)
        try:
            if request.message_type == "tool.status.request":
                response = await self._handle_status(request, endpoint, key)
            elif request.message_type == "tool.result.request":
                response = await self._handle_result(request, endpoint, key)
            else:
                command, reason = control_request_from_payload(request.payload)
                result = (await endpoint.stop(key, reason) if command == "stop" else
                          ToolControlResponse(command=command, status="unsupported"))
                if not isinstance(result, ToolControlResponse) or result.command != command:
                    raise TypeError("Session stop must return its matching ToolControlResponse")
                response = make_control_response_envelope(result, request)
        except ToolEndpointError as error:
            return (self._endpoint_error_response(error, request),)
        except ToolProtocolError as error:
            return (self._protocol_error_response(error, request),)
        return (response,)
