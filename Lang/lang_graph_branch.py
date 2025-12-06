from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# The state
class State(TypedDict):
    messages: Annotated[List, add_messages]


# The model
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100, "top_k": 50, "temperature": 0.1},
)


# The branches
def answer_node(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def clarify_node(state: State):
    return {"messages": [AIMessage(content="Could you share a bit more so I can be precise?")]}


# The route function
def route(state: State):
    last_human = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
    if not last_human:
        return "clarify"
    words = len(str(last_human.content).split())
    return "answer" if words >= 3 else "clarify"


graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.add_node("clarify", clarify_node)

# add a conditional flow
graph.add_conditional_edges(
    START,
    route,
    {
        "answer": "answer",
        "clarify": "clarify",
    },
)
graph.add_edge("answer", END)
graph.add_edge("clarify", END)

# compile
app = graph.compile()
