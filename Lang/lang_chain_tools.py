from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def multiply(a: int, b: int) -> int:
    """This tool multiplies two numbers."""
    return a * b


llm = ChatOpenAI(model="gpt-4o-mini")
llm_tools = llm.bind_tools([multiply])  # exposing the tools to the model
resp = llm_tools.invoke("What is 1*2? If needed, use a tool.")
print(resp.tool_calls or resp.content)  # model may emit a tool call with arguments
