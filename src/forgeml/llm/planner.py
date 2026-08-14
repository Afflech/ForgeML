import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# We will define the Pydantic schema for the LLM to output
class PlannedRun(BaseModel):
    model: Optional[str] = Field(default=None, description="The ML model to run.")
    category: Optional[str] = Field(default=None, description="The object category for the dataset.")
    dataset: Optional[str] = Field(default=None, description="The dataset name.")
    seed: int = Field(default=42, description="Random seed for the run.")
    reasoning: Optional[str] = Field(None, description="Brief explanation of why these parameters were chosen.")

class PlannerError(Exception):
    pass

class LLMPlanner:
    def __init__(self, catalog: Optional[dict] = None):
        self.catalog = catalog
        try:
            from dotenv import load_dotenv
            # Try to load .env from the current working directory
            load_dotenv(Path.cwd() / ".env")
        except ImportError:
            pass

        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise PlannerError("The 'openai' package is required when OPENAI_API_KEY is set.")
        else:
            self.client = None

    def plan(self, user_prompt: str) -> PlannedRun:
        """Use OpenAI structured outputs to parse the user prompt. Falls back to basic regex if no API key."""
        if not self.client:
            # Mock / Fallback logic if no API key is provided
            from rich.console import Console
            Console().print("[dim]Using local fallback parser (no OPENAI_API_KEY found)...[/dim]")
            return self._mock_plan(user_prompt)

        models_list = ", ".join(self.catalog.get("models", [])) if self.catalog else "unknown (make your best guess)"
        categories_list = ", ".join(self.catalog.get("categories", [])) if self.catalog else "unknown (make your best guess)"
        
        system_prompt = (
            "You are ForgeML Planner, an assistant that configures ML training runs. "
            "Extract the model name, dataset category, and any requested seed from the user's request. "
            f"Supported models: {models_list}. "
            f"Supported categories: {categories_list}. "
            "If the user uses Vietnamese (e.g. 'chai lọ', 'ốc vít'), translate it to the appropriate English category. "
            "If the user is vague, make a reasonable guess based on their keywords."
        )

        try:
            # Using GPT-4o-mini as it's fast, cheap, and supports structured outputs
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=PlannedRun,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            raise PlannerError(f"Failed to generate plan from LLM: {e}")

    def _mock_plan(self, prompt: str) -> PlannedRun:
        import re
        prompt = prompt.lower()

        models = self.catalog.get("models", []) if self.catalog else []
        model = models[0] if models else None
        for m in models:
            if m in prompt:
                model = m
                break

        categories = self.catalog.get("categories", []) if self.catalog else []
        category = categories[0] if categories else None
        for c in categories:
            if c in prompt:
                category = c
                break

        # Map vietnamese
        viet_map = {"ốc vít": "screw", "viên thuốc": "pill", "dây cáp": "cable", "chai lọ": "bottle"}
        for v_key, v_val in viet_map.items():
            if v_key in prompt and v_val in categories:
                category = v_val

        # Detect seed
        seed = 42
        seed_match = re.search(r"seed\s*(\d+)", prompt)
        if seed_match:
            seed = int(seed_match.group(1))

        return PlannedRun(
            model=model,
            category=category,
            seed=seed,
            reasoning="Using local regex parser fallback (No OPENAI API key found)"
        )
