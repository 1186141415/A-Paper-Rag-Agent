# tools.py
from http.client import responses

from rag_system import RAGSystem
from datetime import datetime
from llm_utils import client


def rag_tool(query, rag: RAGSystem):
    return rag.ask(query)


def calculator_tool(expression):
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"


def time_tool(_):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def llm_tool(query):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content


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
    },
    {
        "name": "llm",
        "description": "Use for general questions",
        "func": llm_tool
    }
]
