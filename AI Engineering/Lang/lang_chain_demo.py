from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100, "top_k": 50, "temperature": 0.1},
)
prompt = ChatPromptTemplate.from_messages([("system", "You are concise"), ("human", "{question}")])

# The chain
chain = prompt | llm | StrOutputParser()

# A single call
"""
print(chain.invoke({"question": "Give me 3 focus tips"}))
"""

# Run in batch
"""
qs = [{"question": q} for q in ["One tactic for RAG?", "Explain LCEL in 1 line."]]

print(chain.batch(qs))  # many calls at once (batch)
"""

# Run in streaming
for chunk in chain.stream({"question": "Give me 3 focus tips"}):
    print(chunk, end="", flush=True)
