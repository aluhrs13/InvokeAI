from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards import WildcardManager
from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, InvocationContext
from pathlib import Path
from .prompt import PromptOutput


class DynamicPromptInvocation(BaseInvocation):
    """Uses DynamicPrompts to generate a prompt with wildcards"""
    #fmt: off
    type: Literal["dynamic_prompt"] = "dynamic_prompt"
    prompt: str = Field(default=None, description="The input prompt")
    #fmt: on

    def invoke(self, context: InvocationContext) -> PromptOutput:
        wm = WildcardManager(
            Path("D:\\StableDiffusion\\InvokeAI\\invokeai\\app\\invocations\\collections"))
        generator = RandomPromptGenerator(wildcard_manager=wm)
        thing = generator.generate(self.prompt, num_images=4)
        return PromptOutput(prompt=thing[0])
