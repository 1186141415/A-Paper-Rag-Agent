# agent.py

import json
from llm_utils import client


def choose_tool(query, tools):
    tool_desc = "\n".join([
        f"{t['name']}: {t['description']}" for t in tools
    ])

    prompt = f"""
                You are an AI agent.
                
                Available tools:
                {tool_desc}
                
                User question:
                {query}
                
                Return JSON:
                {{"tool": "...", "input": "..."}}
              """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except:
        return {"tool": "llm", "input": query}


def execute_tool(decision, tools, rag=None):
    tool_name = decision["tool"]
    tool_input = decision["input"]

    for t in tools:
        if t["name"] == tool_name:
            if tool_name == "rag":
                result = t["func"](tool_input, rag)
            else:
                result = t["func"](tool_input)

            return {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": result
            }

    return {
        "tool_name": "none",
        "tool_input": tool_input,
        "tool_output": "No valid tool found."
    }


def generate_final_answer(query, tool_result):
    prompt = f"""
                You are an AI assistant
                
                The user asked:
                {query}
                
                A tool was used:
                Tool name: {tool_result['tool_name']}
                Tool input: {tool_result['tool_input']}
                Tool output: {tool_result['tool_output']}
              
                Now provide a final helpful answer to the user.
              """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def run_agent(query, tools, rag=None):
    print("=== Agent Start ===")
    print("User query:", query)

    decision = choose_tool(query, tools)
    print("Decision:", decision)

    tool_result = execute_tool(decision, tools, rag)
    print("Tool result:", tool_result)

    final_answer = generate_final_answer(query, tool_result)
    print("=== Agent End ===")

    return final_answer
