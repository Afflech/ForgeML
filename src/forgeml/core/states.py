from __future__ import annotations

from enum import Enum

from forgeml.core.errors import StateError


class RunState(str, Enum):
    CREATED = "CREATED"
    PACKAGING = "PACKAGING"
    DATASET_UPLOADING = "DATASET_UPLOADING"
    DATASET_READY = "DATASET_READY"
    KERNEL_SUBMITTING = "KERNEL_SUBMITTING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COLLECTING = "COLLECTING"
    COMPLETED = "COMPLETED"


class FailureState(str, Enum):
    FAILED_CONFIG = "FAILED_CONFIG"
    FAILED_DEPENDENCY = "FAILED_DEPENDENCY"
    FAILED_EXECUTION = "FAILED_EXECUTION"
    FAILED_ARTIFACT = "FAILED_ARTIFACT"
    BLOCKED_QUOTA = "BLOCKED_QUOTA"
    BLOCKED_AUTH = "BLOCKED_AUTH"
    FAILED_TRANSIENT = "FAILED_TRANSIENT"


# States that may be retried (network / transient only)
RETRYABLE_STATES = {FailureState.FAILED_TRANSIENT}

# Terminal failure states — never retry
TERMINAL_FAILURE_STATES = {
    FailureState.FAILED_CONFIG,
    FailureState.FAILED_DEPENDENCY,
    FailureState.FAILED_EXECUTION,
    FailureState.FAILED_ARTIFACT,
    FailureState.BLOCKED_QUOTA,
    FailureState.BLOCKED_AUTH,
}

# Valid forward transitions
TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.PACKAGING},
    RunState.PACKAGING: {RunState.DATASET_UPLOADING},
    RunState.DATASET_UPLOADING: {RunState.DATASET_READY},
    RunState.DATASET_READY: {RunState.KERNEL_SUBMITTING},
    RunState.KERNEL_SUBMITTING: {RunState.QUEUED},
    RunState.QUEUED: {RunState.RUNNING},
    RunState.RUNNING: {RunState.COLLECTING},
    RunState.COLLECTING: {RunState.COMPLETED},
    RunState.COMPLETED: set(),
}

# States from which a resume can skip ahead — maps the last persisted
# RunState to the stage where execution should restart.
RESUME_TARGETS: dict[RunState, RunState] = {
    RunState.CREATED: RunState.PACKAGING,
    RunState.PACKAGING: RunState.PACKAGING,              # packaging may be incomplete
    RunState.DATASET_UPLOADING: RunState.DATASET_UPLOADING,  # upload may be incomplete
    RunState.DATASET_READY: RunState.KERNEL_SUBMITTING,  # dataset confirmed ready, skip to submit
    RunState.KERNEL_SUBMITTING: RunState.KERNEL_SUBMITTING,  # submit may be incomplete
    RunState.QUEUED: RunState.QUEUED,                    # was queued, re-enter monitoring
    RunState.RUNNING: RunState.RUNNING,                  # was running, re-enter monitoring
    RunState.COLLECTING: RunState.COLLECTING,            # collection may be incomplete
}


class RunStateMachine:
    """Enforces valid state transitions for a workflow run."""

    def __init__(self) -> None:
        self._current: RunState | FailureState = RunState.CREATED

    @property
    def current(self) -> RunState | FailureState:
        return self._current

    def transition(self, target: RunState) -> None:
        """Transition to a new RunState. Raises StateError if invalid."""
        if not self.can_transition(target):
            raise StateError(
                f"Invalid transition: {self._current!r} → {target!r}"
            )
        self._current = target

    def fail(self, target: FailureState) -> None:
        """Move to a failure state. Raises StateError if already COMPLETED."""
        if self._current == RunState.COMPLETED:
            raise StateError("Cannot fail a completed run")
        if isinstance(self._current, FailureState):
            raise StateError(
                f"Already in failure state {self._current!r}, cannot fail to {target!r}"
            )
        self._current = target

    def can_transition(self, target: RunState) -> bool:
        """Check whether transitioning to target is valid from current state."""
        if isinstance(self._current, FailureState):
            return False
        return target in TRANSITIONS.get(self._current, set())

    @property
    def is_terminal(self) -> bool:
        """True if the run has reached a terminal state (success or permanent failure)."""
        return (
            self._current == RunState.COMPLETED
            or self._current in TERMINAL_FAILURE_STATES
        )

    @property
    def is_resumable(self) -> bool:
        """True if the run is in a non-terminal, non-CREATED state (partial progress)."""
        if self.is_terminal:
            return False
        if self._current == RunState.CREATED:
            return False
        # Retryable failure states are also resumable
        if isinstance(self._current, FailureState):
            return self._current in RETRYABLE_STATES
        return True
