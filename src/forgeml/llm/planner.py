import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# We will define the Pydantic schema for the LLM to output
class PlannedRun(BaseModel):
    model: str = Field(description="The ML model to run (e.g., patchcore, padim, fastflow, efficientad).")
    category: str = Field(description="The object category for the dataset (e.g., bottle, cable, hazelnut, screw).")
    dataset: str = Field(default="mvtec", description="The dataset name. Usually 'mvtec'.")
    seed: int = Field(default=42, description="Random seed for the run.")
    reasoning: Optional[str] = Field(None, description="Brief explanation of why these parameters were chosen.")

class PlannerError(Exception):
    pass

class LLMPlanner:
    def __init__(self):
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

        system_prompt = (
            "You are ForgeML Planner, an assistant that configures ML training runs. "
            "Extract the model name, dataset category, and any requested seed from the user's request. "
            "Supported models: patchcore, padim, fastflow, efficientad. "
            "Supported categories (MVTec): bottle, cable, capsule, carpet, grid, hazelnut, leather, "
            "metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper. "
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

        # Detect model
        model = "patchcore"
        if "padim" in prompt: model = "padim"
        elif "fastflow" in prompt: model = "fastflow"
        elif "efficientad" in prompt: model = "efficientad"

        # Detect category (basic heuristic)
        category = "bottle"
        categories = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                      "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
                      "transistor", "wood", "zipper", "ốc vít", "viên thuốc", "dây cáp"]
        for c in categories:
            if c in prompt:
                category = c
                break

        # Map vietnamese
        viet_map = {"ốc vít": "screw", "viên thuốc": "pill", "dây cáp": "cable"}
        category = viet_map.get(category, category)

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
