from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class TaskPlan(BaseModel):
    title: str
    steps: List[str] = Field(..., min_items=3, description="actionable steps")


structured = ChatOpenAI(model="gpt-4o-mini").with_structured_output(TaskPlan)
plan = structured.invoke("Plan a 20-minute deep-work session for AI Agent notes.")

print(plan.model_dump())
