from pydantic import BaseModel, Field
from typing import Dict


class TestPrompt(BaseModel):
    """Represents a single natural language prompt test case."""
    prompt: str = Field(
        ...,
        description=("The original natural-language" " request from the user.")
        )


class ParameterProperty(BaseModel):
    """Represents the property metadata of a single function parameter."""
    type: str = Field(
        ...,
        description="The primitive data type "
        "(e.g., 'number', 'string', 'boolean')."
        )


class ReturnProperty(BaseModel):
    """Represents the return type metadata of a function."""
    type: str = Field(
        ...,
        description="The primitive return data type of the function."
        )


class FunctionDefinition(BaseModel):
    """Represents the complete schema definition of an available function."""
    name: str = Field(
        ...,
        description="The precise name of the function"
        " (e.g., 'fn_add_numbers')."
        )
    description: str = Field(
        ...,
        description="A short description explaining what the function does."
        )
    parameters: Dict[str, ParameterProperty] = Field(
        ...,
        description="A dictionary mapping parameter "
        "names to their respective type schemas."
    )
    returns: ReturnProperty = Field(
        ...,
        description="The return type definition of this function.")
