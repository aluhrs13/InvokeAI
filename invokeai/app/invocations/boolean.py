from typing import Literal

from pydantic.fields import Field

from .baseinvocation import BaseInvocationOutput


class BooleanOutput(BaseInvocationOutput):
    """Base class for invocations that output a boolean"""
    #fmt: off
    type: Literal["boolean"] = "boolean"

    value: bool = Field(default=None, description="The output boolean")
    #fmt: on

    class Config:
        schema_extra = {
            'required': [
                'type',
                'value',
            ]
        }
