import faiss
import numpy as np
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)

client2 = OpenAI(
    api_key="",
    base_url="https://api.shubiaobiao.com/v1"
)

def get_embedding(text):
    response = client2.embeddings.create(
        model="text-embedding-3-small",  # 改这里
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

def split_text(text, chunk_size=200, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


class RAGSystem:
    def __init__(self, chunks, top_k=5, rerank_k=3):
        self.chunks = chunks
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.index = None
        self.embeddings = None

    def build_index(self):
        if self.embeddings is None:   #加缓存避免重复计算消耗API
            embeddings = [get_embedding(c) for c in self.chunks]
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

    def ask(self, question):
        retrieved = self.retrieve(question, k=self.top_k)

        # 可以加 rerank（你已经有了）
        #context = "\n".join(retrieved)
        sorted_indices = self.rerank(question, retrieved)
        best_chunks = [retrieved[i] for i in sorted_indices[:self.rerank_k]]

        context = "\n".join(best_chunks)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Answer based only on context."},
                {"role": "user", "content": context + "\n\n" + question}
            ]
        )

        return response.choices[0].message.content



if __name__ == '__main__':
    context = "Voice over IP (VoIP) steganography based on low-bit-rate speech codecs has attracted increasing attention due to its high imperceptibility and large embedding capacity, particularly in the fixed codebook (FCB) parameter domain. However, effective steganalysis remains challenging under extreme embedding conditions. At low embedding rates, steganographic artifacts are weak and sparsely distributed, making them difficult to distinguish from natural speech variations. In contrast, at high embedding rates, the recompression-based calibration process may introduce structural distortions that interfere with reliable feature extraction. To address these challenges, this paper proposes a calibration-aware cross-view steganalysis netword for VoIP steganalysis (CACVAN). An embedding-rate-aware data augmentation (ERADA) strategy is first introduced to construct cross-intensity training samples, which improves the robustness of the model under embedding-rate mismatch scenarios. Furthermore, a cross-view interaction backbone (CVIB) is designed to jointly analyze the original speech stream and its recompressed counterpart, enabling the network to capture subtle inconsistencies introduced by steganographic embedding while suppressing content-related variations. A hybrid attention refinement neck (HARN) is then employed to enhance discriminative feature responses and stabilize the modeling of sparse steganographic artifacts. Extensive experiments on public VoIP steganalysis datasets demonstrate that the proposed method consistently outperforms existing state-of-the-art approaches under various embedding rates and speech durations, especially in challenging scenarios involving low embedding rates and short speech segments. Moreover, the proposed framework achieves high computational efficiency and satisfies the real-time requirements of streaming VoIP steganalysis, indicating its practical applicability."
    chunks = split_text(context, chunk_size=200)

    rag = RAGSystem(chunks)
    rag.build_index()

    answer = rag.ask("What is the core contribution?")
    print(answer)