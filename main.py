from openai import OpenAI

client = OpenAI(
    api_key="sk-7ceb2d65728a4e278332a8cf940f7003",
    base_url="https://api.deepseek.com"
)

context = "Transformer 是一种用于处理序列数据的模型结构，广泛应用于自然语言处理任务。"

question = "Transformer是干嘛的？"

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个擅长解释论文的助手"},
        {"role": "user", "content": context + "\n\n问题：" + question}
    ]
)

print(response.choices[0].message.content)