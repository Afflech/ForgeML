import pytest
from unittest.mock import MagicMock, patch
from forgeml.llm.planner import LLMPlanner, PlannedRun, PlannerError

def test_planner_fallback():
    # Ensure no API key is set
    with patch("os.getenv", return_value=None):
        planner = LLMPlanner()
        assert planner.client is None

        plan = planner.plan("run padim on screw dataset with seed 99")
        assert plan.model == "padim"
        assert plan.category == "screw"
        assert plan.seed == 99
        assert plan.reasoning is not None

def test_planner_openai_success():
    with patch("os.getenv", return_value="fake-key"):
        planner = LLMPlanner()
        assert planner.client is not None
        
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.parsed = PlannedRun(
            model="efficientad",
            category="cable",
            seed=123,
            reasoning="AI reasoned this."
        )
        mock_response.choices = [MagicMock(message=mock_message)]
        
        # Patch the parse method
        with patch.object(planner.client.beta.chat.completions, "parse", return_value=mock_response):
            plan = planner.plan("i want efficientad on cables 123")
            assert plan.model == "efficientad"
            assert plan.category == "cable"
            assert plan.seed == 123
            assert plan.reasoning == "AI reasoned this."

def test_planner_openai_failure():
    with patch("os.getenv", return_value="fake-key"):
        planner = LLMPlanner()
        
        with patch.object(planner.client.beta.chat.completions, "parse", side_effect=Exception("API Down")):
            with pytest.raises(PlannerError, match="Failed to generate plan from LLM: API Down"):
                planner.plan("test")
