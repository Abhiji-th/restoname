from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

@tool
def multiply(a: int, b: int) -> int:
    """Given two integers, return the product as an integer"""
    return a*b

llm_with_tools = llm.bind_tools([multiply])

query = "Multiply 2 with 5"

messages = [HumanMessage(query)]

response = llm_with_tools.invoke(query)

messages.append(response)

tool_call = response.tool_calls[0]

tool_response = multiply.invoke(tool_call)

messages.append(tool_response)

result = llm_with_tools.invoke(messages)

print(result.content[0]['text'])

