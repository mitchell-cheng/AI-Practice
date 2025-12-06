from lanchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [("system", "Be brief"), MessagesPlaceholder(variable_name="history"), ("human", "{input}")]
)
chain = prompt | ChatOpenAI(model="gpt-4o")

store = {}  # session_id -> history


def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_history = RunnableWithMessageHistory(
    chain,
    lambda cfg: get_history(cfg["configurable"]["session_id"]),
    input_messages_key="input",
    history_messages_key="history",
)

cfg = {"configurable": {"session_id": "u1"}}
print(with_history.invoke({"input": "My name is Vaibhav."}, config=cfg))
print(with_history.invoke({"input": "What is my name?"}, config=cfg))
