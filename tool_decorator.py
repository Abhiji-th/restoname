from langchain_community.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers"""
    return a*b;

result = multiply.invoke({'a': 2, 'b': 5})

print(result)