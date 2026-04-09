# tools.py

from rag_system import RAGSystem
from datetime import datetime

def rag_tool(query, rag: RAGSystem):
    return rag.ask(query)

def calculator_tool(expression):
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

def time_tool(_):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [
    {
        "name": "rag",
        "description": "Use for paper/document questions",
        "func": rag_tool
    },
    {
        "name": "calculator",
        "description": "Use for math calculations",
        "func": calculator_tool
    },
    {
        "name": "time",
        "description": "Get current time",
        "func": time_tool
    }
]