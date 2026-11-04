"""Cross-process serialization for one Skill's lifecycle mutations."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from types import TracebackType


class SkillOperationBusyError(RuntimeError):
    """Raised when another process owns a Skill lifecycle operation."""


class SkillOperationLock:
    """Hold a non-blocking, crash-safe file lock for one Skill."""

    def __init__(self, state_root: Path, skill_name: str) -> None:
        if skill_name in {"", ".", ".."} or "/" in skill_name or "\\" in skill_name:
            raise ValueError("invalid Skill name for lifecycle lock")
        self.path = state_root.expanduser() / ".locks" / f"{skill_name}.lock"
        self.skill_name = skill_name
        self._handle = None

    def __enter__(self) -> SkillOperationLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            self._lock(handle)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise SkillOperationBusyError(
                    f"Skill {self.skill_name!r} has another lifecycle operation in progress"
                ) from exc
            raise
        self._handle = handle
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is None:
            return
        try:
            self._unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None

    @staticmethod
    def _lock(handle) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
