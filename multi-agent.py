"""
LESSON 5: Multiple Agents Working Together
==========================================
Create specialized agents that collaborate
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


# ============================================
# State Definition
# ============================================

class MultiAgentState(TypedDict):
    user_query: str
    messages: Annotated[list, operator.add]
    research_result: str
    analysis_result: str
    final_response: str
    current_agent: str


# ============================================
# Create Specialized LLMs (Agents)
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ============================================
# Agent Nodes
# ============================================

def researcher_agent(state: MultiAgentState) -> dict:
    """
    Researcher Agent: Gathers information about the topic
    """
    print("🔍 Researcher Agent working...")
    
    messages = [
        SystemMessage(content="""You are a Research Agent. 
        Your job is to provide factual information about topics.
        Be thorough but concise. Focus on key facts."""),
        HumanMessage(content=f"Research this topic: {state['user_query']}")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "research_result": response.content,
        "current_agent": "researcher",
        "messages": [AIMessage(content=f"[Researcher]: {response.content}")]
    }


def analyst_agent(state: MultiAgentState) -> dict:
    """
    Analyst Agent: Analyzes the research and provides insights
    """
    print("📊 Analyst Agent working...")
    
    messages = [
        SystemMessage(content="""You are an Analysis Agent.
        Your job is to analyze information and provide insights.
        Look for patterns, implications, and key takeaways."""),
        HumanMessage(content=f"""
        Original Query: {state['user_query']}
        
        Research Findings:
        {state['research_result']}
        
        Please analyze this information and provide key insights.
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "analysis_result": response.content,
        "current_agent": "analyst",
        "messages": [AIMessage(content=f"[Analyst]: {response.content}")]
    }


def writer_agent(state: MultiAgentState) -> dict:
    """
    Writer Agent: Creates the final response
    """
    print("✍️ Writer Agent working...")
    
    messages = [
        SystemMessage(content="""You are a Writing Agent.
        Your job is to create clear, engaging responses.
        Synthesize information into a well-structured answer."""),
        HumanMessage(content=f"""
        Original Query: {state['user_query']}
        
        Research:
        {state['research_result']}
        
        Analysis:
        {state['analysis_result']}
        
        Create a comprehensive, well-written response.
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "final_response": response.content,
        "current_agent": "writer",
        "messages": [AIMessage(content=f"[Writer]: {response.content}")]
    }


# ============================================
# Build the Multi-Agent Graph
# ============================================

graph_builder = StateGraph(MultiAgentState)

# Add agent nodes
graph_builder.add_node("researcher", researcher_agent)
graph_builder.add_node("analyst", analyst_agent)
graph_builder.add_node("writer", writer_agent)

# Define the flow: Researcher → Analyst → Writer
graph_builder.add_edge(START, "researcher")
graph_builder.add_edge("researcher", "analyst")
graph_builder.add_edge("analyst", "writer")
graph_builder.add_edge("writer", END)

# Compile
graph = graph_builder.compile()


# ============================================
# Visualize the Multi-Agent Flow
# ============================================

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐              │
│   │ 🔍        │    │ 📊        │    │ ✍️         │              │
│   │ RESEARCHER│───▶│ ANALYST   │───▶│ WRITER    │              │
│   │           │    │           │    │           │              │
│   │ Gathers   │    │ Analyzes  │    │ Creates   │              │
│   │ facts     │    │ insights  │    │ response  │              │
│   └───────────┘    └───────────┘    └───────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
""")


# ============================================
# Run the Multi-Agent System
# ============================================

if __name__ == "__main__":
    query = "What are the benefits and challenges of remote work?"
    
    print(f"\n📝 Query: {query}\n")
    print("="*60)
    
    initial_state = {
        "user_query": query,
        "messages": [],
        "research_result": "",
        "analysis_result": "",
        "final_response": "",
        "current_agent": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n" + "="*60)
    print("📋 FINAL RESPONSE:")
    print("="*60)
    print(result["final_response"])