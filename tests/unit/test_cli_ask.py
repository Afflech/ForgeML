from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from forgeml.cli.main import app
from forgeml.llm.planner import PlannedRun

runner = CliRunner()

def test_ask_command_abort():
    # Test that the ask command successfully generates a plan and handles user abort
    with patch("forgeml.llm.planner.LLMPlanner") as MockPlanner:
        mock_instance = MockPlanner.return_value
        mock_instance.plan.return_value = PlannedRun(
            model="efficientad",
            category="cable",
            seed=42,
            reasoning="mock reasoning"
        )
        
        # Pass 'N' to the confirm prompt "Execute this run?"
        result = runner.invoke(app, ["ask", "run efficientad on cable"], input="N\n")
        
        assert result.exit_code == 0
        assert "Analyzing request:" in result.stdout
        assert "mock reasoning" in result.stdout
        assert "Aborted by user." in result.stdout

def test_ask_command_execute():
    # Test that confirming execution calls WorkflowRunner.execute
    with patch("forgeml.llm.planner.LLMPlanner") as MockPlanner, \
         patch("forgeml.workflow.runner.WorkflowRunner") as MockRunner, \
         patch("forgeml.cli.main.Path.exists", return_value=True):
        
        mock_planner = MockPlanner.return_value
        mock_planner.plan.return_value = PlannedRun(
            model="fastflow",
            category="pill",
            seed=123,
            reasoning="another mock"
        )
        
        mock_runner = MockRunner.return_value
        
        result = runner.invoke(app, ["ask", "fastflow pill 123"], input="Y\n")
        
        assert result.exit_code == 0
        mock_runner.execute.assert_called_once_with(
            model="fastflow",
            dataset="mvtec",
            category="pill",
            seed=123,
            dry_run=False
        )
