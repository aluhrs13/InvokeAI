# Step-by-step migrating GroundingDINO from https://github.com/IDEA-Research/GroundingDINO/blob/main/demo/inference_on_a_image.py
# Step 1 - Basic Files.
from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, InvocationContext
from .prompt import PromptOutput


class GroundingDinoInvocation(BaseInvocation):
    """GroundingDINO - https://github.com/IDEA-Research/GroundingDINO"""
    #fmt: off
    type: Literal["grounding_dino"] = "grounding_dino"
    prompt: str = Field(default=None, description="The input prompt")
    #fmt: on

    def invoke(self, context: InvocationContext) -> PromptOutput:
        return PromptOutput(prompt=self.prompt)
