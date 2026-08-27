"""Background job manager.

Generation is CPU/GPU-bound synchronous work, so jobs run on a thread pool and
never block the event loop.  Status is real: progress and stage transitions are
published by the worker itself as it reaches them, and streamed to the UI over
a WebSocket (with a polling endpoint as the fallback).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.db import utcnow
from app.errors import BlueprintError, JobCancelled, NotFoundError
from app.schemas import Job, JobStage, JobStatus, StageStatus

logger = logging.getLogger(__name__)

#: The seven pipeline stages shown in the Studio.
PIPELINE_STAGES: list[tuple[str, str]] = [
    ("prompt_analysis", "Prompt Analysis"),
    ("design_spec", "Design Specification"),
    ("images", "Multi-view Images"),
    ("reconstruction", "3D Reconstruction"),
    ("mesh", "Mesh Processing"),
    ("cad", "CAD Conversion"),
    ("export", "Export"),
]


class CancelledFlag:
    """Cooperative cancellation token shared with the worker thread."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    @property
    def is_set(self) -> bool:
        return self._event.is_set()


@dataclass
class JobRecord:
    id: str
    kind: str
    project_id: str | None
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    hint: str = ""
    stages: list[JobStage] = field(default_factory=list)
    result: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    cancel: CancelledFlag = field(default_factory=CancelledFlag)

    def to_schema(self) -> Job:
        return Job(
            id=self.id,
            project_id=self.project_id,
            kind=self.kind,
            status=self.status,
            progress=round(self.progress, 4),
            message=self.message,
            error=self.error,
            hint=self.hint,
            stages=list(self.stages),
            result=self.result,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class JobContext:
    """Handle passed to a worker so it can report what it is doing."""

    def __init__(self, manager: JobManager, record: JobRecord) -> None:
        self._manager = manager
        self._record = record

    @property
    def job_id(self) -> str:
        return self._record.id

    def check_cancelled(self) -> None:
        """Raise :class:`JobCancelled` if the user cancelled this job."""
        if self._record.cancel.is_set:
            raise JobCancelled()

    def progress(self, fraction: float, message: str = "") -> None:
        self.check_cancelled()
        self._record.progress = max(0.0, min(1.0, float(fraction)))
        if message:
            self._record.message = message
        self._manager._touch(self._record)

    def start_stage(self, key: str, detail: str = "") -> None:
        self._set_stage(key, StageStatus.RUNNING, detail, started=True)

    def finish_stage(self, key: str, detail: str = "") -> None:
        self._set_stage(key, StageStatus.COMPLETED, detail, finished=True)

    def skip_stage(self, key: str, detail: str = "") -> None:
        self._set_stage(key, StageStatus.SKIPPED, detail, finished=True)

    def fail_stage(self, key: str, detail: str = "") -> None:
        self._set_stage(key, StageStatus.FAILED, detail, finished=True)

    def stage_progress(self, key: str) -> Callable[[float, str], None]:
        """Adapter that maps a provider's 0..1 progress onto this stage's slice."""
        keys = [stage.key for stage in self._record.stages]
        try:
            index = keys.index(key)
        except ValueError:
            index = 0
        span = 1.0 / max(1, len(keys))
        base = index * span

        def report(fraction: float, message: str = "") -> None:
            self.progress(base + span * max(0.0, min(1.0, fraction)), message)

        return report

    def _set_stage(self, key: str, status: StageStatus, detail: str,
                   *, started: bool = False, finished: bool = False) -> None:
        for stage in self._record.stages:
            if stage.key == key:
                stage.status = status
                if detail:
                    stage.detail = detail
                if started:
                    stage.started_at = utcnow()
                if finished:
                    stage.finished_at = utcnow()
                break
        self._manager._touch(self._record)


class JobManager:
    """Owns every job's lifecycle and the WebSocket subscriber fan-out."""

    def __init__(self, max_workers: int = 2, keep: int = 200) -> None:
        self._max_workers = max_workers
        # Created lazily so the manager survives a shutdown/restart cycle - a
        # ThreadPoolExecutor cannot be reused once shut down, and the manager
        # is a process-wide singleton that outlives any single app lifespan.
        self._executor: ThreadPoolExecutor | None = None
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self._keep = keep
        self._lock = threading.Lock()
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="pipeline"
            )
        return self._executor

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the event loop so worker threads can publish updates."""
        self._loop = loop

    # ------------------------------------------------------------ submission
    def submit(
        self,
        kind: str,
        worker: Callable[[JobContext], dict[str, Any] | None],
        *,
        project_id: str | None = None,
        stages: list[tuple[str, str]] | None = None,
    ) -> Job:
        job_id = str(uuid.uuid4())
        record = JobRecord(
            id=job_id,
            kind=kind,
            project_id=project_id,
            stages=[JobStage(key=key, label=label) for key, label in (stages or [])],
        )
        with self._lock:
            self._jobs[job_id] = record
            self._order.append(job_id)
            self._evict()

        self._ensure_executor().submit(self._run, record, worker)
        logger.info("Queued job %s (%s) for project %s", job_id, kind, project_id)
        return record.to_schema()

    def _evict(self) -> None:
        """Drop the oldest finished jobs so memory stays bounded."""
        while len(self._order) > self._keep:
            oldest = self._order[0]
            record = self._jobs.get(oldest)
            if record and record.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                break
            self._order.pop(0)
            self._jobs.pop(oldest, None)
            self._subscribers.pop(oldest, None)

    def _run(self, record: JobRecord,
             worker: Callable[[JobContext], dict[str, Any] | None]) -> None:
        context = JobContext(self, record)
        record.status = JobStatus.RUNNING
        record.message = "Starting"
        self._touch(record)
        try:
            result = worker(context)
            if record.cancel.is_set:
                raise JobCancelled()
            record.result = result or {}
            record.status = JobStatus.COMPLETED
            record.progress = 1.0
            record.message = "Completed"
        except JobCancelled:
            record.status = JobStatus.CANCELLED
            record.message = "Cancelled"
            for stage in record.stages:
                if stage.status is StageStatus.RUNNING:
                    stage.status = StageStatus.SKIPPED
            logger.info("Job %s cancelled", record.id)
        except BlueprintError as exc:
            record.status = JobStatus.FAILED
            record.error = exc.user_message
            record.hint = exc.hint
            record.message = exc.user_message
            for stage in record.stages:
                if stage.status is StageStatus.RUNNING:
                    stage.status = StageStatus.FAILED
                    stage.detail = exc.user_message
            logger.error("Job %s failed: %s", record.id, exc.detail, exc_info=True)
        except Exception as exc:
            record.status = JobStatus.FAILED
            record.error = "An unexpected error interrupted this job."
            record.message = record.error
            for stage in record.stages:
                if stage.status is StageStatus.RUNNING:
                    stage.status = StageStatus.FAILED
            logger.exception("Job %s crashed: %s", record.id, exc)
        finally:
            self._touch(record, final=True)

    # -------------------------------------------------------------- querying
    def get(self, job_id: str) -> Job:
        record = self._jobs.get(job_id)
        if record is None:
            raise NotFoundError("That job is no longer available.")
        return record.to_schema()

    def get_record(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job:
        record = self._jobs.get(job_id)
        if record is None:
            raise NotFoundError("That job is no longer available.")
        if record.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            record.cancel.set()
            record.message = "Cancelling"
            self._touch(record)
        return record.to_schema()

    def active_count(self) -> int:
        return sum(
            1 for record in self._jobs.values()
            if record.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        )

    def for_project(self, project_id: str) -> list[Job]:
        return [
            record.to_schema() for record in self._jobs.values()
            if record.project_id == project_id
        ]

    # ------------------------------------------------------------ publishing
    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.setdefault(job_id, set()).add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(job_id)
        if subscribers:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(job_id, None)

    def _touch(self, record: JobRecord, *, final: bool = False) -> None:
        """Publish the current state to every subscriber of this job."""
        record.updated_at = utcnow()
        subscribers = self._subscribers.get(record.id)
        if not subscribers or self._loop is None:
            return
        payload = record.to_schema().model_dump(mode="json")
        if final:
            payload["final"] = True

        def publish() -> None:
            for queue in list(subscribers):
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    logger.debug("Dropping job update for a slow subscriber")

        try:
            self._loop.call_soon_threadsafe(publish)
        except RuntimeError:
            # The loop is shutting down; polling still reflects the state.
            logger.debug("Event loop unavailable while publishing job update")

    def shutdown(self) -> None:
        """Stop the worker pool. Idempotent; a later submit starts a fresh one."""
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


manager = JobManager()
