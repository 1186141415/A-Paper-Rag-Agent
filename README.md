# Paper RAG Agent System

一个基于 **FastAPI + FAISS + LLM** 实现的论文分析与问答系统。  
项目支持从本地 PDF 论文中抽取文本、切分文档、构建向量索引，并通过 **RAG + Tool Calling + Session Memory** 的方式完成论文问答。

这个项目的目标不是只做一个“能跑的 Demo”，而是逐步搭建一个更接近 AI 工程岗位要求的、可扩展的 **RAG + Agent 系统原型**。

---

## 1. Project Overview

在论文阅读与分析场景中，直接向大模型提问往往会遇到这些问题：

- 模型回答脱离论文原文
- 无法稳定引用具体文档内容
- 长文档理解效果不稳定
- 一轮对话之后缺乏上下文记忆
- 不同任务类型缺少明确的工具分发机制

为了解决这些问题，本项目实现了一套基础的论文问答系统：

- 使用 **PDF Loader** 读取本地论文
- 对文本进行清洗与切分
- 基于 **Embedding + FAISS** 构建向量检索
- 通过 **RAG** 为大模型补充上下文
- 使用 **Agent** 对问题进行工具选择
- 使用 **Session Manager** 管理多会话历史

---

## 2. Main Features

当前项目已经具备以下能力：

### 2.1 Document Processing
- 加载 `data/` 目录下的 PDF 文件
- 对 PDF 提取出的文本进行清洗
- 将文档切分为带 overlap 的文本块

### 2.2 Retrieval-Augmented Generation (RAG)
- 使用 embedding 模型对文本块进行向量化
- 使用 **FAISS** 建立向量索引
- 根据用户问题进行相似度检索
- 使用 LLM 对召回结果进行 rerank
- 拼接高相关上下文后生成最终回答

### 2.3 Agent Tool Routing
系统支持基础工具路由，当前包括：

- `rag`：处理论文 / 文档相关问题
- `calculator`：处理简单计算
- `time`：获取当前时间
- `llm`：处理通用问答

### 2.4 Session-Based Memory
- 基于 `session_id` 维护独立对话历史
- 不同会话之间互不干扰
- 支持历史轮数裁剪，避免上下文无限增长

### 2.5 API Service
- 基于 **FastAPI** 提供问答接口
- 支持服务启动时自动加载文档并建立索引
- 提供基础异常处理与日志输出

---

## 3. Tech Stack

本项目当前主要使用以下技术：

- **Python**
- **FastAPI**
- **FAISS**
- **OpenAI-compatible API**
- **DeepSeek Chat Model**
- **Embedding API**
- **PyPDF**
- **NumPy**
- **dotenv**

---

## 4. Project Structure

```text
paper-rag-agent/
├── app.py                 # FastAPI 入口
├── agent.py               # Agent 主流程：工具选择、执行、最终回答生成
├── rag_system.py          # RAG 核心逻辑：索引构建、检索、rerank、问答
├── tools.py               # 工具定义与工具列表
├── session_manager.py     # 多会话历史管理
├── data_loader.py         # PDF 加载、清洗、切分
├── llm_utils.py           # LLM / Embedding 客户端与工具判断
├── config.py              # 配置项与环境变量读取
├── logger_config.py       # 日志配置
├── requirements.txt       # 项目依赖
├── README.md              # 项目说明文档
└── data/                  # 本地 PDF 数据目录
```

---

## 5. Workflow

系统的整体流程如下：

```text
Service Startup
    ↓
Load PDFs from data/
    ↓
Clean & Split Documents
    ↓
Build Embeddings
    ↓
Build FAISS Index
    ↓
Receive User Question
    ↓
Agent Chooses Tool
    ↓
Execute Tool (RAG / LLM / Calculator / Time)
    ↓
Generate Final Answer
    ↓
Store Session History
```

如果问题被路由到 `rag` 工具，则内部流程为：

```text
User Question
    ↓
Vector Retrieval
    ↓
Top-K Chunks
    ↓
LLM Rerank
    ↓
Build Context with Sources
    ↓
LLM Answer Generation
```

---

## 6. Core Modules

### 6.1 `app.py`
负责：

- 启动 FastAPI 服务
- 在启动阶段加载 PDF 数据并初始化 RAG 系统
- 提供 `/ask` 接口
- 管理 session 历史并返回最终回答

### 6.2 `rag_system.py`
这是项目的核心模块，主要实现：

- 文本向量化
- FAISS 索引构建
- 相似度检索
- 基于 LLM 的 rerank
- 带上下文与对话历史的问答生成

### 6.3 `agent.py`
实现一个简单的 Agent 主流程：

1. 根据用户问题选择工具
2. 执行对应工具
3. 根据工具结果生成最终回答

### 6.4 `tools.py`
将系统能力封装为工具，方便 Agent 调用。

### 6.5 `session_manager.py`
用于管理多用户 / 多 session 的会话历史，避免不同会话共享状态。

### 6.6 `data_loader.py`
负责：

- 加载 PDF
- 提取文本
- 文本清洗
- 文档切分

---

## 7. Configuration

项目通过 `.env` 文件配置模型与接口地址。

示例：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://api.shubiaobiao.com/v1

CHAT_MODEL=deepseek-chat
EMBEDDING_MODEL=text-embedding-3-small
DATA_DIR=data
```

---

## 8. Quick Start

### 8.1 Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 8.2 Create Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 8.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 8.4 Prepare Data

将需要分析的 PDF 论文放到 `data/` 目录下，例如：

```text
data/
├── paper1.pdf
└── paper2.pdf
```

### 8.5 Run the Service

```bash
uvicorn app:app --reload
```

启动后可以访问：

```text
http://127.0.0.1:8000/docs
```

---

## 9. API Example

### 9.1 Ask Question

**POST** `/ask`

请求体：

```json
{
  "session_id": "user_001",
  "question": "What are the differences between paper1 and paper2?"
}
```

返回示例：

```json
{
  "session_id": "user_001",
  "question": "What are the differences between paper1 and paper2?",
  "answer": "..."
}
```

### 9.2 Clear Session

**POST** `/clear/{session_id}`

用于清空某个会话的历史记录。

---

## 10. Design Highlights

这个项目当前体现的重点包括：

### 10.1 从“单轮 RAG”走向“会话型 RAG”
不仅支持基于文档的单轮问答，也支持按 `session_id` 管理会话历史。

### 10.2 从“直接问模型”走向“工具驱动问答”
通过 Agent 先判断问题适合使用哪种工具，再执行相应流程。

### 10.3 从“粗召回”走向“检索 + rerank”
除了 FAISS 相似度检索，还额外加入 LLM rerank，提高上下文质量。

### 10.4 从“学习型脚本”走向“工程化原型”
项目中已经加入：

- 配置管理
- 日志系统
- 异常处理
- API 化接口
- 会话边界管理

---

## 11. Current Limitations

当前版本仍然是一个持续迭代中的工程原型，还存在一些可继续改进的地方：

- 文档切分策略较基础
- rerank 依赖 LLM 调用，成本较高
- 工具选择逻辑仍然较简单
- 暂未引入更完整的工作流编排
- Web 展示层尚未完善
- 缺少更系统的评测与测试

---

## 12. Future Work

后续计划推进的方向包括：

- [ ] 引入 LangChain 重构部分 RAG 链路
- [ ] 引入 LangGraph 实现多步骤工作流
- [ ] 增加更多工具与 Router 策略
- [ ] 支持多论文对比分析
- [ ] 接入 Django 或前端页面作为展示层
- [ ] 增加测试、评测与更稳定的工程结构
- [ ] 优化检索策略与上下文构造方式

---

## 13. Why This Project Matters

这个项目希望体现的不是“调用一个大模型接口”，而是：

- 如何从文档中构建可检索知识
- 如何把 RAG 做成一个可以服务化的系统
- 如何把问答系统从单轮调用推进到 Agent + Session 的结构
- 如何把学习项目逐步打磨成更接近实际岗位要求的 AI 工程原型

---

## 14. License

This project is for learning, experimentation, and AI engineering practice.