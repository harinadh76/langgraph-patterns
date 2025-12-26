"""
LESSON 6: Supervisor-Worker Pattern
===================================
The Supervisor decides which Worker to use and when to finish.
This is the most powerful multi-agent pattern!
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

load_dotenv()

# ============================================
# State Definition
# ============================================

class SupervisorState(TypedDict):
    # The original task/query
    task: str
    
    # Messages history
    messages: Annotated[list, operator.add]
    
    # Which worker should work next (or FINISH)
    next_worker: str
    
    # Results from each worker
    worker_results: Annotated[dict, lambda x, y: {**x, **y}]
    
    # Final answer
    final_answer: str
    
    # Iteration counter (to prevent infinite loops)
    iteration: int


# ============================================
# Initialize LLM
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================
# Worker Definitions
# ============================================

WORKERS = ["researcher", "coder", "writer"]


def researcher_worker(state: SupervisorState) -> dict:
    """Research Worker: Finds information and facts"""
    print("  🔍 [Researcher Worker] Working...")
    
    messages = [
        SystemMessage(content="""You are a Research Worker.
        Your job is to find relevant information, facts, and data.
        Be thorough and cite your reasoning.
        Keep your response focused and under 200 words."""),
        HumanMessage(content=f"""
        Task: {state['task']}
        
        Previous work done: {json.dumps(state.get('worker_results', {}), indent=2)}
        
        Provide your research findings:
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "worker_results": {"researcher": response.content},
        "messages": [AIMessage(content=f"[Researcher]: {response.content}")]
    }


def coder_worker(state: SupervisorState) -> dict:
    """Coder Worker: Writes code and technical solutions"""
    print("  💻 [Coder Worker] Working...")
    
    messages = [
        SystemMessage(content="""You are a Coding Worker.
        Your job is to write code, technical solutions, or pseudocode.
        Explain your code clearly.
        Keep your response focused."""),
        HumanMessage(content=f"""
        Task: {state['task']}
        
        Previous work done: {json.dumps(state.get('worker_results', {}), indent=2)}
        
        Provide your technical solution:
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "worker_results": {"coder": response.content},
        "messages": [AIMessage(content=f"[Coder]: {response.content}")]
    }


def writer_worker(state: SupervisorState) -> dict:
    """Writer Worker: Creates polished content"""
    print("  ✍️ [Writer Worker] Working...")
    
    messages = [
        SystemMessage(content="""You are a Writing Worker.
        Your job is to create clear, well-structured content.
        Synthesize information into readable format.
        Keep your response focused."""),
        HumanMessage(content=f"""
        Task: {state['task']}
        
        Previous work done: {json.dumps(state.get('worker_results', {}), indent=2)}
        
        Create your written content:
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "worker_results": {"writer": response.content},
        "messages": [AIMessage(content=f"[Writer]: {response.content}")]
    }


# ============================================
# Supervisor Node
# ============================================

def supervisor_node(state: SupervisorState) -> dict:
    """
    Supervisor: Decides which worker to call next or finish
    """
    print("\n👔 [Supervisor] Evaluating...")
    
    # Check iteration limit
    current_iteration = state.get("iteration", 0)
    if current_iteration >= 5:
        print("  ⚠️ Maximum iterations reached. Finishing.")
        return {
            "next_worker": "FINISH",
            "iteration": current_iteration + 1
        }
    
    # Build the supervisor prompt
    supervisor_prompt = f"""You are a Supervisor managing a team of workers.

Available workers:
- researcher: Finds information, facts, and data
- coder: Writes code and technical solutions
- writer: Creates polished, well-written content

Current Task: {state['task']}

Work completed so far:
{json.dumps(state.get('worker_results', {}), indent=2)}

Based on the task and work done:
1. If the task is COMPLETE, respond with: FINISH
2. If more work is needed, respond with the name of the next worker to use

IMPORTANT: Respond with ONLY one word - either a worker name (researcher, coder, writer) or FINISH.
"""
    
    messages = [
        SystemMessage(content="You are a task supervisor. Respond with only one word."),
        HumanMessage(content=supervisor_prompt)
    ]
    
    response = llm.invoke(messages)
    next_action = response.content.strip().lower()
    
    # Validate response
    if next_action not in WORKERS and next_action != "finish":
        # Default to FINISH if response is unclear
        next_action = "finish"
    
    print(f"  📋 Decision: {next_action.upper()}")
    
    return {
        "next_worker": next_action.lower(),
        "iteration": current_iteration + 1,
        "messages": [AIMessage(content=f"[Supervisor]: Next action -> {next_action}")]
    }


# ============================================
# Routing Function
# ============================================

def route_supervisor(state: SupervisorState) -> Literal["researcher", "coder", "writer", "finish"]:
    """Route based on supervisor's decision"""
    next_worker = state.get("next_worker", "").lower()
    
    if next_worker == "researcher":
        return "researcher"
    elif next_worker == "coder":
        return "coder"
    elif next_worker == "writer":
        return "writer"
    else:
        return "finish"


# ============================================
# Final Answer Node
# ============================================

def compile_final_answer(state: SupervisorState) -> dict:
    """Compile all worker results into final answer"""
    print("\n📝 [Finalizer] Compiling final answer...")
    
    messages = [
        SystemMessage(content="""Compile the work from all workers into a final, 
        cohesive response. Organize it clearly."""),
        HumanMessage(content=f"""
        Original Task: {state['task']}
        
        Worker Outputs:
        {json.dumps(state.get('worker_results', {}), indent=2)}
        
        Create the final comprehensive answer:
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {"final_answer": response.content}


# ============================================
# Build the Supervisor-Worker Graph
# ============================================

print("""
┌─────────────────────────────────────────────────────────────────┐
│                 SUPERVISOR-WORKER PATTERN                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      ┌────────────────┐                         │
│                      │   SUPERVISOR   │◀─────────────┐          │
│                      │     👔         │              │          │
│                      └───────┬────────┘              │          │
│                              │                       │          │
│              ┌───────────────┼───────────────┐       │          │
│              │               │               │       │          │
│              ▼               ▼               ▼       │          │
│       ┌──────────┐    ┌──────────┐    ┌──────────┐  │          │
│       │RESEARCHER│    │  CODER   │    │  WRITER  │──┘          │
│       │    🔍    │    │    💻    │    │    ✍️    │             │
│       └──────────┘    └──────────┘    └──────────┘             │
│                                                                  │
│                              │                                   │
│                              ▼ (When FINISH)                     │
│                      ┌────────────────┐                         │
│                      │   FINALIZER    │                         │
│                      │      📋        │                         │
│                      └────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
""")

# Create the graph
graph_builder = StateGraph(SupervisorState)

# Add all nodes
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("researcher", researcher_worker)
graph_builder.add_node("coder", coder_worker)
graph_builder.add_node("writer", writer_worker)
graph_builder.add_node("finish", compile_final_answer)

# Start with supervisor
graph_builder.add_edge(START, "supervisor")

# Supervisor routes to workers or finish
graph_builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "coder": "coder",
        "writer": "writer",
        "finish": "finish"
    }
)

# All workers go back to supervisor
graph_builder.add_edge("researcher", "supervisor")
graph_builder.add_edge("coder", "supervisor")
graph_builder.add_edge("writer", "supervisor")

# Finish goes to END
graph_builder.add_edge("finish", END)

# Compile
graph = graph_builder.compile()


# ============================================
# Run the Supervisor-Worker System
# ============================================

if __name__ == "__main__":
    # Test tasks
    tasks = [
        "Create a Python function that calculates compound interest and explain how it works",
        # "Research the benefits of meditation and write a short blog post about it",
    ]
    
    for task in tasks:
        print("\n" + "="*70)
        print(f"📋 TASK: {task}")
        print("="*70)
        
        initial_state: SupervisorState = {
            "task": task,
            "messages": [],
            "next_worker": "",
            "worker_results": {},
            "final_answer": "",
            "iteration": 0
        }
        
        result = graph.invoke(initial_state)
        
        print("\n" + "="*70)
        print("✅ FINAL ANSWER:")
        print("="*70)
        print(result["final_answer"])
        
        print("\n📊 Workers Used:")
        for worker, output in result["worker_results"].items():
            print(f"  - {worker}: {len(output)} characters of output")