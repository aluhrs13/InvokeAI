from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards import WildcardManager
from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, InvocationContext
from pathlib import Path
from .prompt import PromptOutput

# TODO: Magic Prompts
    # Magic Prompts Length
    # Magic Prompts Creativity
# TODO: I'm Feeling Lucky
# TODO: Attention Grabber
# TODO: Jinja Templates

# TODO: Multiple invocations
    # Combinatorial Mode
    # Fixed Seed

# TODO: Config
    # Wilcards Directory

class DynamicPromptInvocation(BaseInvocation):
    """Uses DynamicPrompts to generate a prompt with wildcards"""
    #fmt: off
    type: Literal["dynamic_prompt"] = "dynamic_prompt"
    prompt: str = Field(default=None, description="The input prompt")
    feeling_lucky: bool = Field(default=False, description="I'm feeling lucky")
    attention_grabber: bool = Field(default=False, description="Attention Grabber")
    magic_prompts: bool = Field(default=False, description="Magic Prompts")
    magic_prompts_length: int = Field(default=100, description="Magic Prompts Length")
    magic_prompts_creativity: float = Field(default=0.5,gt=0.0,lt=1.0, description="Magic Prompts Creativity")
    #fmt: on

    def invoke(self, context: InvocationContext) -> PromptOutput:
        wm = WildcardManager(
            Path("D:\\StableDiffusion\\InvokeAI\\invokeai\\app\\invocations\\collections"))
        generator = RandomPromptGenerator(wildcard_manager=wm)
        thing = generator.generate(self.prompt, num_images=4)
        return PromptOutput(prompt=thing[0])
