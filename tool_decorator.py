from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="This is the first number for multiplication")
    b: int = Field(required=True, description="This is the second number for multiplication")

def multiply(a: int, b: int) -> int:
    """Multiply two integers"""
    return a*b;

tool = StructuredTool.from_function(
    func=multiply,
    name="Multipy",
    description="Multiply two integers",
    args_schema=MultiplyInput
)

print(tool.invoke({'a': 2, 'b': 5}))