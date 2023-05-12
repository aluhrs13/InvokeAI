# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext
from .math import IntOutput
from .prompt import PromptOutput
from .boolean import BooleanOutput

# Pass-through parameter nodes - used by subgraphs

class ParamIntInvocation(BaseInvocation):
    """An integer parameter"""
    #fmt: off
    type: Literal["param_int"] = "param_int"
    a: int = Field(default=0, description="The integer value")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a)

class ParamPromptInvocation(BaseInvocation):
    """A prompt parameter"""
    #fmt: off
    type: Literal["param_prompt"] = "param_prompt"
    prompt: str = Field(default="", description="The prompt value")
    #fmt: on

    def invoke(self, context: InvocationContext) -> PromptOutput:
        return PromptOutput(prompt=self.prompt)

class ParamBoolInvocation(BaseInvocation):
    """A boolean parameter"""
    #fmt: off
    type: Literal["param_bool"] = "param_bool"
    value: bool = Field(default=True, description="The boolean value")
    #fmt: on

    def invoke(self, context: InvocationContext) -> BooleanOutput:
        return BooleanOutput(value=self.value)
