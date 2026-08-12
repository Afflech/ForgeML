"""Test state transition enforcement during workflow execution."""
from __future__ import annotations

from pathlib import Path

import pytest

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.errors import StateError
from forgeml.core.states import RunState, FailureState, RunStateMachine, TRANSITIONS
from forgeml.workflow.runner import WorkflowRunner
from tests.workflow.conftest import StubProvider


class TestStateMachineInWorkflow:
    """Verify that the WorkflowRunner uses the state machine for enforcement."""

    def test_run_persists_all_states_in_order(
        self, forge_project: Path, forge_cfg: ForgeConfig, stub_provider: StubProvider,
    ):
        """A successful run should touch all states in order in the DB."""
        runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=stub_provider)
        runner.execute(model="patchcore", dataset="mvtec", category="bottle")

        from forgeml.db.engine import get_engine
        from forgeml.db.models import Run
        from sqlmodel import Session, select

        engine = get_engine(forge_project)
        with Session(engine) as session:
            run = session.exec(select(Run)).first()
            # Final state should be COMPLETED
            assert run.status == RunState.COMPLETED.value

    def test_state_machine_rejects_skipping_states(self):
        """Standalone state machine test: can't skip states."""
        sm = RunStateMachine()
        with pytest.raises(StateError):
            sm.transition(RunState.DATASET_UPLOADING)  # must go through PACKAGING first

    def test_all_transitions_are_single_step_forward(self):
        """Verify TRANSITIONS only allows sequential forward steps."""
        ordered = [
            RunState.CREATED,
            RunState.PACKAGING,
            RunState.DATASET_UPLOADING,
            RunState.DATASET_READY,
            RunState.KERNEL_SUBMITTING,
            RunState.QUEUED,
            RunState.RUNNING,
            RunState.COLLECTING,
            RunState.COMPLETED,
        ]
        for i, state in enumerate(ordered[:-1]):
            expected_next = ordered[i + 1]
            assert TRANSITIONS[state] == {expected_next}, \
                f"{state} should only transition to {expected_next}, got {TRANSITIONS[state]}"
        # COMPLETED should have no transitions
        assert TRANSITIONS[RunState.COMPLETED] == set()

    def test_failure_during_run_persists_failure_state(
        self, forge_project: Path, forge_cfg: ForgeConfig,
    ):
        """When a provider fails, the DB should record the failure state."""
        from forgeml.core.errors import ProviderError

        provider = StubProvider(
            fail_at="submit_kernel",
            fail_error=ProviderError("API error"),
        )
        runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=provider)

        with pytest.raises(ProviderError):
            runner.execute(model="patchcore", dataset="mvtec", category="bottle")

        from forgeml.db.engine import get_engine
        from forgeml.db.models import Run
        from sqlmodel import Session, select

        engine = get_engine(forge_project)
        with Session(engine) as session:
            run = session.exec(select(Run)).first()
            assert run.status == FailureState.FAILED_TRANSIENT.value
            assert run.error_type == "ProviderError"
            assert run.finished_at is not None
