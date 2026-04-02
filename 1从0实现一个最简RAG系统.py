from openai import OpenAI #语言模型还是选择Openai  这里其实也可以用一些国产模型或者其他模型
                          #但是有意思的是现在基本上所有模型都能兼容openai，
                          #什么意思，意思是如果你用了openai的基座，现在市面上所有的模型
                          #你都能直接拿来用，只需要改一下api_key和base_url就行

client = OpenAI(
    api_key="",
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