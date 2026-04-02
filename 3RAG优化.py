from openai import OpenAI
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)



#context = "伴随着互联网的足迹，全球已迈入高度信息化的新时代，信息安全问题日益凸显，其已成为全球各国高度关注的焦点[1,2]。在 2016 年 12 月 27 日，国家互联网信息办公室发布《国家网络空间安全战略》[3]，标志着信息安全已成为我国国家战略。在信息安全领域中，信息隐藏与加密技术是两大主要研究内容，加密技术主要保护安全通信的内容，而信息隐藏则旨在保护安全通信的事实。现如今高度发达的各类分析技术对非受控环境下的安全通信提出了更高的要求，不仅要求内容保密，还希望过程能够隐蔽。因此，信息隐藏技术是加密技术的必要补充。其中，隐写（Steganography）是一种能将秘密信息难以感知地隐藏至可公开载体的信息隐藏技术[4]，它在保护保密通信或者存储内容的同时，也保护了隐藏这一行为事实，拥有较高的应用价值，由此成为了研究人员重点关注的方向之一。隐写嵌入信息的关键在于利用人类感知系统的不敏感性和载体数据的冗余性，能够完美契合两项特性且在互联网通讯中广泛存在的各类多媒体载体，如视频[5]、音频[6]、图像[7]和文本[8]等，均为隐写技术所青睐的隐藏载体。然而，各类层出不穷的隐写技术给信息保护带来便利的同时，也成为易被不法分子（如基地组织、ISIS 组织）恶意利用的工具[9]。这些未经授权主体可以通过在消息发送端添加隐写模块来发送大量的非法信息，进而轻易地逃避有关监管机构的审查。因此，研究相应的隐写检测技术对于保障国家的网络空间安全具有十分重要的意义与价值，正日益引起国内外学术界及相关部门的高度重视，如近期欧洲各国发起的联合项目 UNCOVER 其单项资助金额就高达 5000 万人民币[10]。在各种多媒体隐藏载体中，随着移动通讯和各类即时通讯软件的普及，网络语音（Voice over Internet Protocol，VoIP）已成为最受关注的研究载体之一[11]。相比于其它多媒体载体，VoIP 拥有三个重要的优势[12]：首先，VoIP 的连续动态传输为大容量的信息隐藏提供了有利条件。其次，VoIP 的多维载体空间（可用于隐藏的多个相关网络协议、语音编码器及语音编码参数）能为实现安全隐写提供多种可能性。第三，VoIP 通话的即时性给隐写检测带来了极大的挑战，其某种程度在客观上增强了隐蔽通信的安全性。以上这些独特的优势催生出大量基于VoIP的隐写的相关研究，大致可分为基于协议的隐写和基于语音流的隐写[13]。前者主要利用 VoIP 涉及到的相关协议（如 IP[14]，UDP[15]，RTP/RTCP[16]）的冗余来隐藏信息，这一类方法的原理与传统协议隐写的原理类似[17]；而后者则能够巧妙地将秘密信息嵌入到语音流中。相比之下，后者在技术多样性和隐写性能两方面上都更胜一筹，因此基于语音流的隐写及隐写分析已成为该领域的研究主流，也是本文主要研究的内容。典型的基于语音流的隐写及隐写检测场景如图 1.1所示。目前，VoIP 语音中的主流语音编码器，即混合编码器（如 G.729，G.723.1，AMR），通常拥有三个语音编码参数域用于信息嵌入：线性预测系数（LinearPredictive Coefficients，LPC）参数域，自适应码本（Adaptive Codebook，ACB）参数域和固定码本（Fixed Codebook，FCB）参数域。利用这些参数在动态编码搜索过程中的冗余性，隐写者可以在保持良好语音质量的同时嵌入大量的隐秘信息，整个过程高度隐蔽且仅携有微小的失真，给隐写的检测工作带来了极大的挑战性。在现实中，为了实现对语音流中潜在隐写的实时检测，通常采用滑动窗口的方式进行操作，即动态检测每个窗口中的语音样本。这种检测场景对隐写检测方法带来了更高的挑战性，包括：(1) 低嵌入强度检测的有效性：为了降低被检测出的概率，隐写者倾向于降低单位长度样本中的嵌入强度（即以低嵌入率执行隐写操作），因此隐写检测方法必须能够适应这种变化，即在低嵌入率下依然能够提供有效的检测结果；(2) 短窗口样本检测的有效性：为了能够快速发现异常，我们通常需要在尽可能短的样本条件下给出有效的检测结果，窗口尺寸越小，越有利于定位信息嵌入的位置，这同样也是未来实现嵌入信息提取的前提条件；(3) 快速检测：显而易见的是，越早给出检测结果则越有利于减轻未授权信息泄露的不利影响，即隐写检测方法应该能够以较为合理的时间开销给出检测结果；(4) 通用检测：在实际检测场景中往往需要面对的是多种参数域上的隐写方法，因此隐写检测方法必须能够同时检测多个参数域上的隐写方法。面对上述挑战，国内外众多研究者相继展开研究。近年来，深度学习凭借其强大的深度表征能力，在计算机视觉、自然语言处理和语音等领域大放异彩[18]，同样也给基于网络语音流的隐写检测带来新一波发展高峰，催生出许多先进的隐写检测方法，然而其检测准确率以及通用性距离实际应用需求还有一定的差距。鉴于此，进一步研究符合实际应用需求的高准确率、高通用性隐写检测方法，对于相关领域研究以及维护国家网络空间安全都有着极其重大的意义。"

context = "Voice over IP (VoIP) steganography based on low-bit-rate speech codecs has attracted increasing attention due to its high imperceptibility and large embedding capacity, particularly in the fixed codebook (FCB) parameter domain. However, effective steganalysis remains challenging under extreme embedding conditions. At low embedding rates, steganographic artifacts are weak and sparsely distributed, making them difficult to distinguish from natural speech variations. In contrast, at high embedding rates, the recompression-based calibration process may introduce structural distortions that interfere with reliable feature extraction. To address these challenges, this paper proposes a calibration-aware cross-view steganalysis netword for VoIP steganalysis (CACVAN). An embedding-rate-aware data augmentation (ERADA) strategy is first introduced to construct cross-intensity training samples, which improves the robustness of the model under embedding-rate mismatch scenarios. Furthermore, a cross-view interaction backbone (CVIB) is designed to jointly analyze the original speech stream and its recompressed counterpart, enabling the network to capture subtle inconsistencies introduced by steganographic embedding while suppressing content-related variations. A hybrid attention refinement neck (HARN) is then employed to enhance discriminative feature responses and stabilize the modeling of sparse steganographic artifacts. Extensive experiments on public VoIP steganalysis datasets demonstrate that the proposed method consistently outperforms existing state-of-the-art approaches under various embedding rates and speech durations, especially in challenging scenarios involving low embedding rates and short speech segments. Moreover, the proposed framework achieves high computational efficiency and satisfies the real-time requirements of streaming VoIP steganalysis, indicating its practical applicability."
# 假设你已经有chunks


def split_text(text, chunk_size=100):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = split_text(context, chunk_size=200)

#Step 2：获取 embedding（用 DeepSeek）
client2 = OpenAI(
   api_key="",
   base_url="https://api.shubiaobiao.com/v1"
#

) #由于openai是付费的，所以咱们不用，用其他的本地嵌入
def get_embedding(text):
   response = client2.embeddings.create(
       model="text-embedding-3-small",  # 改这里
       input=text
   )
   return np.array(response.data[0].embedding, dtype="float32")

#model = SentenceTransformer("all-MiniLM-L6-v2")
#def get_embedding(text):
#   return np.array(model.encode(text), dtype="float32")

embeddings = get_embedding(chunks[0])

#Step 3：构建向量库（FAISS）
# 计算每个chunk的向量
embeddings = [get_embedding(c) for c in chunks]

# 转成矩阵
embedding_matrix = np.vstack(embeddings)

# 建立索引
dim = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dim)

# 加入向量
index.add(embedding_matrix)

#Step 4：查询最相关chunk（核心！）
def retrieve(query, k=1):
    query_vec = get_embedding(query).reshape(1, -1)

    distances, indices = index.search(query_vec, k)

    return [chunks[i] for i in indices[0]]

question = "What is the core contribution of this paper?"

relevant_chunks = retrieve(question, k=5)

context_for_llm = "\n".join(relevant_chunks)

print(relevant_chunks)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a professional research paper analysis assistant. Answer based only on the provided context."},
        {"role": "user", "content": context_for_llm + "\n\nquestion：" + question}
    ]
)

print(response.choices[0].message.content)
