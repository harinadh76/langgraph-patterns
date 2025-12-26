"""
ADVANCED PROJECT: AI Customer Support System
=============================================
A complete supervisor-worker system with:
- Intent detection
- Multiple specialized workers
- Escalation handling
- Response quality check
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from datetime import datetime

load_dotenv()


# ============================================
# State Definition
# ============================================

class SupportState(TypedDict):
    # Customer info
    customer_message: str
    customer_id: str
    
    # Processing info
    intent: str
    priority: str
    messages: Annotated[list, operator.add]
    
    # Worker results
    worker_outputs: Annotated[dict, lambda x, y: {**x, **y}]
    
    # Supervisor control
    next_action: str
    iteration: int
    
    # Final outputs
    response: str
    needs_human: bool
    ticket_created: bool


# ============================================
# LLM Setup
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# ============================================
# Intent Classifier Node
# ============================================

def classify_intent(state: SupportState) -> dict:
    """Classify the customer's intent"""
    print("🏷️ [Intent Classifier] Analyzing message...")
    
    prompt = f"""Classify this customer message into ONE category:

Customer Message: "{state['customer_message']}"

Categories:
- billing: Payment issues, invoices, refunds, subscription
- technical: Product bugs, errors, how-to questions
- account: Login issues, password reset, profile updates
- feedback: Complaints, suggestions, compliments
- sales: Pricing questions, upgrades, new purchases

Also determine priority:
- high: Urgent, customer frustrated, service down
- medium: Standard issues
- low: General inquiries, feedback

Respond in JSON format:
{{"intent": "category", "priority": "level"}}
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        result = json.loads(response.content)
        intent = result.get("intent", "general")
        priority = result.get("priority", "medium")
    except:
        intent = "general"
        priority = "medium"
    
    print(f"   Intent: {intent}, Priority: {priority}")
    
    return {
        "intent": intent,
        "priority": priority,
        "messages": [AIMessage(content=f"Classified as {intent} ({priority} priority)")]
    }


# ============================================
# Specialized Workers
# ============================================

def billing_worker(state: SupportState) -> dict:
    """Handle billing-related queries"""
    print("💳 [Billing Worker] Processing...")
    
    prompt = f"""You are a Billing Support Specialist.
    
Customer Issue: {state['customer_message']}

Provide helpful information about:
- How to view/download invoices
- Refund policies (30-day money back guarantee)
- Payment method updates
- Subscription management

Be empathetic and solution-oriented. Keep response under 150 words.
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "worker_outputs": {"billing": response.content},
        "messages": [AIMessage(content=f"[Billing]: {response.content}")]
    }


def technical_worker(state: SupportState) -> dict:
    """Handle technical support queries"""
    print("🔧 [Technical Worker] Processing...")
    
    prompt = f"""You are a Technical Support Specialist.
    
Customer Issue: {state['customer_message']}

Provide:
1. Possible causes of the issue
2. Step-by-step troubleshooting steps
3. When to escalate to engineering

Be clear and technical but accessible. Keep response under 150 words.
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "worker_outputs": {"technical": response.content},
        "messages": [AIMessage(content=f"[Technical]: {response.content}")]
    }


def account_worker(state: SupportState) -> dict:
    """Handle account-related queries"""
    print("👤 [Account Worker] Processing...")
    
    prompt = f"""You are an Account Support Specialist.
    
Customer Issue: {state['customer_message']}

Provide help with:
- Password reset procedures
- Account security
- Profile updates
- Account recovery

Be security-conscious but helpful. Keep response under 150 words.
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "worker_outputs": {"account": response.content},
        "messages": [AIMessage(content=f"[Account]: {response.content}")]
    }


def general_worker(state: SupportState) -> dict:
    """Handle general queries"""
    print("💬 [General Worker] Processing...")
    
    prompt = f"""You are a Customer Support Representative.
    
Customer Message: {state['customer_message']}

Provide a helpful, friendly response. If you need more information,
politely ask for it. Keep response under 150 words.
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "worker_outputs": {"general": response.content},
        "messages": [AIMessage(content=f"[General]: {response.content}")]
    }


# ============================================
# Supervisor Node
# ============================================

def supervisor(state: SupportState) -> dict:
    """Supervisor decides next action"""
    print("\n👔 [Supervisor] Making decision...")
    
    iteration = state.get("iteration", 0)
    
    # First iteration: route based on intent
    if iteration == 0:
        intent = state.get("intent", "general")
        
        if intent == "billing":
            next_action = "billing_worker"
        elif intent == "technical":
            next_action = "technical_worker"
        elif intent == "account":
            next_action = "account_worker"
        else:
            next_action = "general_worker"
        
        print(f"   → Routing to: {next_action}")
        
        return {
            "next_action": next_action,
            "iteration": iteration + 1
        }
    
    # Second iteration: check if we need quality review
    elif iteration == 1:
        # High priority gets quality check
        if state.get("priority") == "high":
            print("   → High priority: Adding quality check")
            return {
                "next_action": "quality_check",
                "iteration": iteration + 1
            }
        else:
            print("   → Proceeding to finalize")
            return {
                "next_action": "finalize",
                "iteration": iteration + 1
            }
    
    # Otherwise: finalize
    else:
        return {
            "next_action": "finalize",
            "iteration": iteration + 1
        }


def route_supervisor(state: SupportState) -> str:
    """Route based on supervisor decision"""
    return state.get("next_action", "finalize")


# ============================================
# Quality Check Node
# ============================================

def quality_check(state: SupportState) -> dict:
    """Check response quality for high-priority issues"""
    print("✅ [Quality Check] Reviewing response...")
    
    worker_outputs = state.get("worker_outputs", {})
    
    prompt = f"""Review this customer support response for quality:

Customer Message: {state['customer_message']}

Draft Response: {json.dumps(worker_outputs)}

Check for:
1. Is it empathetic and professional?
2. Does it address the customer's concern?
3. Are next steps clear?
4. Should this be escalated to a human?

Respond with JSON:
{{"quality": "good/needs_improvement", "needs_human": true/false, "improved_response": "..."}}
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        result = json.loads(response.content)
        needs_human = result.get("needs_human", False)
        improved = result.get("improved_response", "")
        
        if improved:
            return {
                "worker_outputs": {"quality_improved": improved},
                "needs_human": needs_human,
                "messages": [AIMessage(content="Quality check completed")]
            }
    except:
        pass
    
    return {
        "needs_human": False,
        "messages": [AIMessage(content="Quality check completed")]
    }


# ============================================
# Finalize Response
# ============================================

def finalize_response(state: SupportState) -> dict:
    """Create the final customer response"""
    print("📝 [Finalizer] Creating final response...")
    
    worker_outputs = state.get("worker_outputs", {})
    
    # Use quality-improved version if available
    if "quality_improved" in worker_outputs:
        content = worker_outputs["quality_improved"]
    else:
        # Combine all worker outputs
        content = "\n\n".join(worker_outputs.values())
    
    # Create ticket for high priority
    ticket_created = state.get("priority") == "high"
    
    prompt = f"""Format this as a polished customer support email response:

Content: {content}

Include:
- Friendly greeting
- Clear solution/next steps
- Professional sign-off

Keep the same information but make it flow well.
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "response": response.content,
        "ticket_created": ticket_created
    }


# ============================================
# Build the Graph
# ============================================

print("""
┌─────────────────────────────────────────────────────────────────┐
│              CUSTOMER SUPPORT SYSTEM FLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐     ┌────────────────┐                       │
│   │    START     │────▶│ INTENT CLASSIFY│                       │
│   └──────────────┘     └───────┬────────┘                       │
│                                │                                 │
│                        ┌───────▼────────┐                       │
│                        │   SUPERVISOR   │◀──────────┐           │
│                        └───────┬────────┘           │           │
│                                │                    │           │
│          ┌─────────────────────┼─────────────────┐  │           │
│          ▼           ▼         ▼         ▼       │  │           │
│     ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │  │           │
│     │BILLING │ │TECHNIC │ │ACCOUNT │ │GENERAL │──┘  │           │
│     │   💳   │ │   🔧   │ │   👤   │ │   💬   │     │           │
│     └────────┘ └────────┘ └────────┘ └────────┘     │           │
│                                                      │           │
│                        ┌───────────────┐            │           │
│                        │ QUALITY CHECK │────────────┘           │
│                        │      ✅       │ (high priority)        │
│                        └───────┬───────┘                        │
│                                │                                 │
│                        ┌───────▼────────┐                       │
│                        │   FINALIZE     │                       │
│                        │      📝        │                       │
│                        └───────┬────────┘                       │
│                                │                                 │
│                        ┌───────▼────────┐                       │
│                        │      END       │                       │
│                        └────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
""")

# Build graph
graph_builder = StateGraph(SupportState)

# Add nodes
graph_builder.add_node("classify", classify_intent)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("billing_worker", billing_worker)
graph_builder.add_node("technical_worker", technical_worker)
graph_builder.add_node("account_worker", account_worker)
graph_builder.add_node("general_worker", general_worker)
graph_builder.add_node("quality_check", quality_check)
graph_builder.add_node("finalize", finalize_response)

# Add edges
graph_builder.add_edge(START, "classify")
graph_builder.add_edge("classify", "supervisor")

# Supervisor conditional routing
graph_builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "billing_worker": "billing_worker",
        "technical_worker": "technical_worker",
        "account_worker": "account_worker",
        "general_worker": "general_worker",
        "quality_check": "quality_check",
        "finalize": "finalize"
    }
)

# Workers go back to supervisor
graph_builder.add_edge("billing_worker", "supervisor")
graph_builder.add_edge("technical_worker", "supervisor")
graph_builder.add_edge("account_worker", "supervisor")
graph_builder.add_edge("general_worker", "supervisor")
graph_builder.add_edge("quality_check", "supervisor")

# Finalize goes to END
graph_builder.add_edge("finalize", END)

# Compile
support_graph = graph_builder.compile()


# ============================================
# Test the System
# ============================================

if __name__ == "__main__":
    test_messages = [
        {
            "message": "I was charged twice for my subscription last month and I'm really frustrated!",
            "customer_id": "CUST-001"
        },
        {
            "message": "How do I reset my password? I forgot it.",
            "customer_id": "CUST-002"
        },
        {
            "message": "The app keeps crashing when I try to upload files. This is urgent - I have a deadline!",
            "customer_id": "CUST-003"
        }
    ]
    
    for test in test_messages:
        print("\n" + "="*70)
        print(f"📧 Customer Message: {test['message']}")
        print(f"   Customer ID: {test['customer_id']}")
        print("="*70)
        
        initial_state: SupportState = {
            "customer_message": test["message"],
            "customer_id": test["customer_id"],
            "intent": "",
            "priority": "",
            "messages": [],
            "worker_outputs": {},
            "next_action": "",
            "iteration": 0,
            "response": "",
            "needs_human": False,
            "ticket_created": False
        }
        
        result = support_graph.invoke(initial_state)
        
        print("\n" + "-"*70)
        print("📤 RESPONSE TO CUSTOMER:")
        print("-"*70)
        print(result["response"])
        print("-"*70)
        print(f"📊 Ticket Created: {result['ticket_created']}")
        print(f"🚨 Needs Human: {result['needs_human']}")
        print(f"🏷️ Intent: {result['intent']} | Priority: {result['priority']}")