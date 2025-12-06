# The state
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import END, START, StateGraph, add_messages


class State(TypedDict):
    messages: Annotated[List, add_messages]


llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100, "top_k": 50, "temperature": 0.1},
)


# node
def model_node(state: State):
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}


# graph
graph = StateGraph(State)
graph.add_node("model", model_node)
graph.add_edge(START, "model")
graph.add_edge("model", END)

# compile
app = graph.compile()


result = app.invoke({"messages": [HumanMessage(content="Give me 3 focus tips")]})

print(result["messages"][-1].content)
