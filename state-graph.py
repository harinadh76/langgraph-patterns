"""
LESSON 2: Understanding State in Depth
======================================
Learn how state flows and accumulates through the graph
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

# ============================================
# CONCEPT: State Reducers
# ============================================
# By default, new values REPLACE old values
# With Annotated + operator.add, values ACCUMULATE

class CounterState(TypedDict):
    # This will be REPLACED each time
    current_step: str
    
    # This will ACCUMULATE (add to list)
    history: Annotated[list[str], operator.add]
    
    # This will ACCUMULATE (add numbers)
    total: Annotated[int, operator.add]


def step_one(state: CounterState) -> dict:
    print("📍 Step One")
    return {
        "current_step": "one",
        "history": ["Completed step 1"],  # Adds to list
        "total": 10                         # Adds 10
    }


def step_two(state: CounterState) -> dict:
    print("📍 Step Two")
    return {
        "current_step": "two",
        "history": ["Completed step 2"],  # Adds to list
        "total": 20                         # Adds 20
    }


def step_three(state: CounterState) -> dict:
    print("📍 Step Three")
    return {
        "current_step": "three",
        "history": ["Completed step 3"],  # Adds to list
        "total": 30                         # Adds 30
    }


# Build the graph
graph_builder = StateGraph(CounterState)

graph_builder.add_node("step_one", step_one)
graph_builder.add_node("step_two", step_two)
graph_builder.add_node("step_three", step_three)

graph_builder.add_edge(START, "step_one")
graph_builder.add_edge("step_one", "step_two")
graph_builder.add_edge("step_two", "step_three")
graph_builder.add_edge("step_three", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    initial_state = {
        "current_step": "start",
        "history": [],
        "total": 0
    }
    
    result = graph.invoke(initial_state)
    
    print("\n✅ Final Result:")
    print(f"   Current Step: {result['current_step']}")  # Only "three"
    print(f"   History: {result['history']}")  # All three items
    print(f"   Total: {result['total']}")  # 10 + 20 + 30 = 60