"""Domain exceptions.

Every exception carries a ``user_message`` that is safe to show in the UI and
an optional ``hint`` describing the remedy.  Raw tracebacks stay in the server
log; the API only ever returns these curated messages.
"""

from __future__ import annotations


class BlueprintError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    code: str = "internal_error"
    user_message: str = "Something went wrong while processing your request."

    def __init__(self, user_message: str | None = None, *, hint: str = "", detail: str = ""):
        self.user_message = user_message or self.__class__.user_message
        self.hint = hint
        self.detail = detail or self.user_message
        super().__init__(self.detail)

    def to_payload(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.user_message}
        if self.hint:
            payload["hint"] = self.hint
        return payload


class NotFoundError(BlueprintError):
    status_code = 404
    code = "not_found"
    user_message = "The requested item could not be found."


class ValidationError(BlueprintError):
    status_code = 422
    code = "validation_error"
    user_message = "The request could not be validated."


class ProviderUnavailableError(BlueprintError):
    """A provider cannot run - missing key, missing package, missing GPU."""

    status_code = 503
    code = "provider_unavailable"
    user_message = "The selected provider is not available."


class GenerationError(BlueprintError):
    status_code = 502
    code = "generation_failed"
    user_message = "Generation failed. See the job log for details."


class MeshError(BlueprintError):
    status_code = 422
    code = "mesh_error"
    user_message = "The mesh could not be processed."


class ExportError(BlueprintError):
    status_code = 422
    code = "export_failed"
    user_message = "The model could not be exported in the requested format."


class CADToolError(BlueprintError):
    status_code = 503
    code = "cad_tool_unavailable"
    user_message = "The requested CAD tool is not available on this machine."


class JobCancelled(BlueprintError):
    status_code = 409
    code = "cancelled"
    user_message = "The job was cancelled."


class OutOfMemoryError(BlueprintError):
    status_code = 507
    code = "out_of_memory"
    user_message = (
        "The device ran out of memory while generating. Try a smaller image size "
        "or a lower voxel resolution in Settings."
    )
