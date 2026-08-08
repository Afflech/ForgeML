from enum import Enum


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
