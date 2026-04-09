import numpy as np
import time
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

client2 = OpenAI(
    api_key=os.getenv("SHUBIAOBIAO_API_KEY"),
    base_url="https://api.shubiaobiao.com/v1"
)


def get_embedding(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client2.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding, dtype="float32")

        except Exception as e:
            print(f"Embedding失败，第{attempt + 1}次重试...")
            time.sleep(2)

    print("Embedding最终失败，返回零向量")
    return np.zeros(1536, dtype="float32")  # embedding维度



def decide_tool(query):
    prompt = f"""  
    You are an AI assistant.

    Decide whether the following question needs document retrieval.

    Question:
    {query}

    Answer ONLY:
    - "RAG" if it needs document-based answer
    - "LLM" if it can be answered directly   
              """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()