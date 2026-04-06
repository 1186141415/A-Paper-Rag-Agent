import os
import faiss
import numpy as np
from openai import OpenAI
from pyexpat.errors import messages
from pypdf import PdfReader
import re
import time

client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)

client2 = OpenAI(
    api_key="",
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


def split_text(text, chunk_size=200, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                documents.append({
                    "text": text,
                    "source": filename
                })

    return documents


def process_documents(documents):
    all_chunks = []

    for doc in documents:
        chunks = split_text(doc["text"], chunk_size=200, overlap=50)

        for c in chunks:
            all_chunks.append({
                "text": c,
                "source": doc["source"]
            })

    return all_chunks


def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # 多行空白换成一行
    text = re.sub(r'\s+', ' ', text)  # 将所有连续空白字符（空格、制表符、换行等）替换成单个空格，实现“规范化空白”。
    return text


def load_pdfs(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            text = clean_text(text)

            documents.append({
                "text": text,
                "source": filename
            })

    return documents


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


class RAGSystem:
    def __init__(self, chunks, top_k=20, rerank_k=10):
        self.chunks = chunks
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.index = None
        self.embeddings = None

        self.chat_history = []  # 新增记忆

    # 把rag变成一个工具
    def rag_tool(self, query):
        return self.ask(query)

    def build_index(self):
        texts = [c["text"] for c in self.chunks]

        if self.embeddings is None:  # 加缓存避免重复计算消耗API
            embeddings = [get_embedding(t) for t in texts]
            self.embeddings = np.vstack(embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query, k=5):
        query_vec = get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, k)
        return [self.chunks[i] for i in indices[0]]

    def rerank(self, query, chunks):
        prompt = f"""
                You are a ranking assistant.

                Query:
                {query}

                Rank the following passages from most relevant to least relevant.

                Passages:
                """

        for i, c in enumerate(chunks):
            prompt += f"\n[{i}] {c}\n"

        prompt += "\nReturn ONLY the indices in sorted order, like [2,0,1]."

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )

        import ast
        try:
            return ast.literal_eval(response.choices[0].message.content)
        except:
            return list(range(len(chunks)))

    # 限制历史长度，否则会爆token
    def trim_history(self, max_turn=3):
        if len(self.chat_history) > max_turn * 2:
            self.chat_history = self.chat_history[-max_turn * 2:]

    def ask(self, question):
        retrieved = self.retrieve(question, k=self.top_k)

        # 可以加 rerank（你已经有了）
        # context = "\n".join(retrieved)

        # rerank（用text）
        texts = [c["text"] for c in retrieved]
        sorted_indices = self.rerank(question, texts)
        best_chunks = [retrieved[i] for i in sorted_indices[:self.rerank_k]]

        # 拼context（加来源！）
        context = ""
        for c in best_chunks:
            context += f"[Source: {c['source']}]\n{c['text']}\n\n"

        # 构造messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer based on context and coversation history."
            }
        ]

        # 加历史对话
        messages.extend(self.chat_history)

        # 当前问题
        messages.append({
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}"
        })

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )

        answer = response.choices[0].message.content

        self.chat_history.append({
            "role": "user",
            "content": question,
        })

        self.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        self.trim_history()

        return answer

    def ask_with_agent(self, question):
        decision = decide_tool(question)
        print("🧠 Decision:", decision)

        if "RAG" in decision:
            print("📚 Using RAG...")
            return self.ask(question)

        else:
            print("💬 Using LLM...")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ]
            )

            return response.choices[0].message.content


if __name__ == '__main__':
    docs = load_pdfs("data")  # 你的文件夹
    chunks = process_documents(docs)

    rag = RAGSystem(chunks)
    rag.build_index()

    answer = rag.ask("What are the differences between paper1 and paper2?")
    print(answer)
