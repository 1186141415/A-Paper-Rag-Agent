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

    return json.loads(response.choices[0].message.content)


def run_agent(query, tools, rag=None):
    decision = choose_tool(query, tools)

    tool_name = decision["tool"]
    tool_input = decision["input"]

    #print("input:")
    #print(tool_input)

    for t in tools:
        if t["name"] == tool_name:
            # 特殊处理 rag_tool（需要rag实例）
            if tool_name == "rag":
                return t["func"](tool_input, rag)
            else:
                return t["func"](tool_input)

    return "No valid tool found"