import pytest
from forgeml.core.states import (
    RunState,
    FailureState,
    RunStateMachine,
    TERMINAL_FAILURE_STATES,
    RESUME_TARGETS
)
from forgeml.core.errors import StateError


def test_valid_transitions():
    sm = RunStateMachine()
    assert sm.current == RunState.CREATED
    
    # Walk through the happy path
    sm.transition(RunState.PACKAGING)
    assert sm.current == RunState.PACKAGING
    
    sm.transition(RunState.DATASET_UPLOADING)
    sm.transition(RunState.DATASET_READY)
    sm.transition(RunState.KERNEL_SUBMITTING)
    sm.transition(RunState.QUEUED)
    sm.transition(RunState.RUNNING)
    sm.transition(RunState.COLLECTING)
    sm.transition(RunState.COMPLETED)
    assert sm.current == RunState.COMPLETED


def test_invalid_transitions():
    sm = RunStateMachine()
    
    with pytest.raises(StateError):
        sm.transition(RunState.RUNNING)
        
    sm.transition(RunState.PACKAGING)
    
    with pytest.raises(StateError):
        sm.transition(RunState.COMPLETED)


def test_terminal_states():
    sm = RunStateMachine()
    # Reach COMPLETED
    sm.transition(RunState.PACKAGING)
    sm.transition(RunState.DATASET_UPLOADING)
    sm.transition(RunState.DATASET_READY)
    sm.transition(RunState.KERNEL_SUBMITTING)
    sm.transition(RunState.QUEUED)
    sm.transition(RunState.RUNNING)
    sm.transition(RunState.COLLECTING)
    sm.transition(RunState.COMPLETED)
    
    assert sm.is_terminal is True
    
    with pytest.raises(StateError):
        sm.transition(RunState.COLLECTING)
        
    with pytest.raises(StateError):
        sm.fail(FailureState.FAILED_EXECUTION)


def test_failure_states():
    sm = RunStateMachine()
    # Transition to packaging
    sm.transition(RunState.PACKAGING)
    # Fail
    sm.fail(FailureState.FAILED_DEPENDENCY)
    assert sm.current == FailureState.FAILED_DEPENDENCY
    
    assert sm.is_terminal is True  # Because FAILED_DEPENDENCY is terminal
    
    with pytest.raises(StateError):
        sm.transition(RunState.DATASET_UPLOADING)
        
    # Cannot fail again from a failure state
    with pytest.raises(StateError):
        sm.fail(FailureState.FAILED_CONFIG)


def test_is_resumable():
    sm = RunStateMachine()
    assert sm.is_resumable is False  # CREATED
    
    sm.transition(RunState.PACKAGING)
    assert sm.is_resumable is True
    
    sm.fail(FailureState.FAILED_TRANSIENT)
    assert sm.is_resumable is True  # Transient failures are resumable
    
    sm2 = RunStateMachine()
    sm2.transition(RunState.PACKAGING)
    sm2.fail(FailureState.FAILED_CONFIG)
    assert sm2.is_resumable is False  # Terminal failure is not resumable
    
    sm3 = RunStateMachine()
    sm3.transition(RunState.PACKAGING)
    sm3.transition(RunState.DATASET_UPLOADING)
    sm3.transition(RunState.DATASET_READY)
    sm3.transition(RunState.KERNEL_SUBMITTING)
    sm3.transition(RunState.QUEUED)
    sm3.transition(RunState.RUNNING)
    sm3.transition(RunState.COLLECTING)
    sm3.transition(RunState.COMPLETED)
    assert sm3.is_resumable is False


def test_resume_targets():
    assert RESUME_TARGETS[RunState.CREATED] == RunState.PACKAGING
    assert RESUME_TARGETS[RunState.DATASET_READY] == RunState.KERNEL_SUBMITTING
    assert RESUME_TARGETS[RunState.RUNNING] == RunState.RUNNING
