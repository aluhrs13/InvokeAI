from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, InvocationContext
from .prompt import PromptOutput


class PromptPassthroughInvocation(BaseInvocation):
    """Just passthrough the prompt"""
    #fmt: off
    type: Literal["prompt_passthrough"] = "prompt_passthrough"
    prompt: str = Field(default=None, description="The input prompt")
    prompt2: str = Field(default=None, description="The input prompt2")

    #fmt: on

    def invoke(self, context: InvocationContext) -> PromptOutput:
        return PromptOutput(prompt=self.prompt +self.prompt2)
