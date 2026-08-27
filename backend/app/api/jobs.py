"""Job status: polling endpoint plus a WebSocket stream."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.jobs import manager
from app.schemas import Job, JobStatus
from app.storage import validate_project_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["jobs"])

#: How often the socket sends a keep-alive when a job is quiet.
HEARTBEAT_SECONDS = 20.0


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    """Poll a job's status. The WebSocket is preferred, this is the fallback."""
    return manager.get(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=Job)
def cancel_job(job_id: str) -> Job:
    """Request cancellation. The worker stops at its next checkpoint."""
    return manager.cancel(job_id)


@router.get("/projects/{project_id}/jobs", response_model=list[Job])
def jobs_for_project(project_id: str) -> list[Job]:
    return manager.for_project(validate_project_id(project_id))


@router.websocket("/ws/jobs/{job_id}")
async def job_stream(websocket: WebSocket, job_id: str) -> None:
    """Stream live job updates until the job reaches a terminal state."""
    await websocket.accept()
    record = manager.get_record(job_id)
    if record is None:
        await websocket.send_json({"error": "unknown job", "job_id": job_id})
        await websocket.close()
        return

    queue = manager.subscribe(job_id)
    try:
        # Send the current state immediately so a late subscriber is not blind.
        await websocket.send_json(record.to_schema().model_dump(mode="json"))
        if record.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return

        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                await websocket.send_json({"heartbeat": True})
                continue

            await websocket.send_json(payload)
            if payload.get("status") in ("completed", "failed", "cancelled"):
                return
    except WebSocketDisconnect:
        logger.debug("Client disconnected from job %s", job_id)
    except RuntimeError:
        # The socket closed mid-send; nothing useful to do.
        logger.debug("Job socket closed for %s", job_id, exc_info=True)
    finally:
        manager.unsubscribe(job_id, queue)
        # The socket may already be closed by the client.
        with contextlib.suppress(RuntimeError):
            await websocket.close()
