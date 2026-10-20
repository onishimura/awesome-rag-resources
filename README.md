# Awesome RAG Resources

A curated list of awesome Retrieval Augmented Generation (RAG) resources, tools, and frameworks.

## Contents

- [Frameworks & Platforms](#frameworks--platforms)
- [Agent Frameworks](#agent-frameworks)
- [Vector Stores](#vector-stores)
- [Embedding & Retrieval](#embedding--retrieval)
- [Document Processing](#document-processing)
- [Security and Privacy](#security-and-privacy)
- [Graph RAG](#graph-rag)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Guides & Research](#guides--research)

## Frameworks & Platforms

Production-ready frameworks and platforms for building RAG applications:

- [LangChain](https://www.langchain.com/) - Popular framework for building LLM applications with RAG capabilities
- [LlamaIndex](https://www.llamaindex.ai/) - Data framework for LLM applications with focus on RAG
- [Haystack](https://haystack.deepset.ai/) - Open-source framework for building search and RAG applications
- [Ragie](https://www.ragie.ai/) - Managed RAG platform for ingestion, indexing and retrieval
- [Kotaemon](https://github.com/Cinnamon/kotaemon) - Open-source RAG-based tool for chatting with your documents
- [R2R](https://github.com/SciPhi-AI/R2R) - Research to Reality: Framework for production RAG applications
- [Ragflow](https://github.com/infiniflow/ragflow) - Efficient RAG workflow engine
- [Letta](https://www.letta.com/) - Platform for building stateful AI agents with memory capabilities
- [DSPy](https://github.com/stanfordnlp/dspy) - Framework for programming (not prompting) LM pipelines, with optimizers that tune RAG prompts and weights
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything) - All-in-one multimodal RAG over text, images, tables and equations, built on LightRAG
- [PageIndex](https://github.com/VectifyAI/PageIndex) - Vectorless, reasoning-based RAG that navigates a document tree index instead of chunk similarity

## Agent Frameworks

Frameworks for building AI agents with RAG capabilities:

- [LangGraph](https://github.com/langchain-ai/langgraph) - Framework for building stateful agent workflows
- [PydanticAI](https://pydantic.dev/docs/ai/overview/) - Type-safe AI development with Pydantic
- [LlamaIndex Workflows](https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/) - Multi-agent workflows in LlamaIndex
- [Smolagents](https://github.com/huggingface/smolagents) - Lightweight agents framework by Hugging Face
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) - Multi-agent framework by Microsoft, successor to AutoGen
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) - RL framework for training LLMs to interleave reasoning with search engine calls

## Vector Stores

Vector databases and stores for RAG applications:

- [Qdrant](https://qdrant.tech/) - Vector database with extended filtering and performance focus
- [PostgreSQL (pgvector)](https://github.com/pgvector/pgvector) - Vector similarity search for PostgreSQL
- [Weaviate](https://weaviate.io/) - Open-source vector database
- [Chroma](https://www.trychroma.com/) - Embedded vector store for RAG applications
- [Faiss](https://github.com/facebookresearch/faiss) - Efficient similarity search by Meta
- [Milvus](https://milvus.io/) - Open-source vector database for AI applications
- [Pinecone](https://www.pinecone.io/) - Managed vector database service
- [LanceDB](https://github.com/lancedb/lancedb) - Embedded, serverless vector database on the Lance columnar format with hybrid and multimodal search
- [Vespa](https://github.com/vespa-engine/vespa) - Search engine and vector database with hybrid search and multi-phase ranking at scale
- [sqlite-vec](https://github.com/asg017/sqlite-vec) - Vector search SQLite extension that runs anywhere SQLite does

## Embedding & Retrieval

Tools and resources for embedding generation and retrieval:

- [ColPali (Vespa)](https://vespa-engine.github.io/pyvespa/examples/simplified-retrieval-with-colpali-vlm_Vespa-cloud.html) - Simplified retrieval with ColPali and Vespa
- [ColPali (Qdrant)](https://danielvanstrien.xyz/posts/post-with-code/colpali-qdrant/2024-10-02_using_colpali_with_qdrant.html) - Using ColPali with Qdrant
- [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) - Anthropic's approach to contextual retrieval
- [Contextual Retrieval (Instructor)](https://python.useinstructor.com/blog/2024/09/26/implementing-anthropics-contextual-retrieval-with-async-processing/) - Implementing contextual retrieval with Instructor
- [Hugging Face Embeddings](https://huggingface.co/models?other=embeddings) - Collection of embedding models
- [Fine-tuning Embedding Models](https://www.youtube.com/watch?v=v28Pu7hsJ0s) - Guide to fine-tuning embedding models
- [Qwen3 Embedding & Reranker](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) - Apache-2.0 multilingual embedding and reranking models from 0.6B to 8B parameters
- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) - 300M-parameter on-device embedding model from Google
- [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) - Multimodal embedding and reranking models for text, image and visual-document retrieval
- [ColPali](https://github.com/illuin-tech/colpali) - Reference implementation and training code for ColPali/ColQwen visual document retrievers
- [PyLate](https://github.com/lightonai/pylate) - Library for training and serving late-interaction (ColBERT) models on Sentence Transformers
- [Sentence Transformers](https://github.com/huggingface/sentence-transformers) - Library for training and using dense, sparse and reranker embedding models
- [rerankers](https://github.com/AnswerDotAI/rerankers) - Unified API over cross-encoder, ColBERT, LLM and API rerankers
- [bm25s](https://github.com/xhluca/bm25s) - Fast pure-Python BM25 for the lexical half of hybrid search
- [Chonkie](https://github.com/feyninc/chonkie) - Lightweight chunking library with token, semantic, late and code chunkers
- [Evaluating Chunking Strategies](https://www.trychroma.com/research/evaluating-chunking) - Chroma's research comparing chunking strategies on retrieval quality

## Document Processing

Tools for processing and parsing documents:

- [LlamaParse](https://www.llamaindex.ai/llamaparse) - Document parsing tool by LlamaIndex
- [LlamaParse Documentation](https://developers.llamaindex.ai/llamaparse/parse/getting_started/) - Official documentation for LlamaParse document parsing
- [Docling](https://github.com/docling-project/docling) - Document understanding toolkit
- [Markitdown](https://github.com/microsoft/markitdown) - Document processing framework by Microsoft
- [MinerU](https://github.com/opendatalab/MinerU) - PDF and Office-to-Markdown/JSON parser with layout, table, formula and OCR support
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR and document parsing toolkit including the compact PaddleOCR-VL model
- [Marker](https://github.com/datalab-to/marker) - Fast PDF/EPUB/Office to Markdown and JSON conversion with optional LLM assist (GPL code, restricted model weights)
- [olmOCR](https://github.com/allenai/olmocr) - Ai2's VLM-based toolkit for linearizing PDFs into clean text at scale
- [LiteParse](https://github.com/run-llama/liteparse) - Fast local PDF parser with bounding boxes, no LLM or cloud required

## Security and Privacy

Resources for securing RAG applications:

- [OpenFGA with LangChain](https://auth0.com/blog/building-a-secure-rag-with-python-langchain-and-openfga/#Proposed-Solution--Secure-Retrieval-with-LangChain--OpenAI--and-OpenFGA) - Build a secure RAG agent using LangChain and Auth0 FGA on Node.js
- [Secure RAG with LlamaIndex](https://developer.auth0.com/resources/labs/authorization/building-rag-apps-with-llamaindex-and-fga#overview) - Authorization in RAG applications
- [Qdrant Security](https://qdrant.tech/documentation/security/) - API keys and JWT-based role-based access control in Qdrant
- [LlamaIndex and LLMGuard](https://www.llamaindex.ai/blog/secure-rag-with-llamaindex-and-llm-guard-by-protect-ai) - Secure RAG implementation with LLMGuard
- [LangSmith User Management](https://docs.langchain.com/langsmith/user-management) - User management and access control in LangSmith
- [OWASP LLM08: Vector and Embedding Weaknesses](https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/) - OWASP Top 10 for LLMs entry on RAG-specific risks and mitigations

## Graph RAG

Knowledge-graph and structure-aware retrieval approaches:

- [GraphRAG](https://github.com/microsoft/graphrag) - Microsoft's modular graph-based RAG pipeline with community summaries
- [LightRAG](https://github.com/HKUDS/LightRAG) - Simple, fast graph-based RAG with dual-level retrieval and incremental updates
- [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) - Memory-inspired graph RAG for multi-hop retrieval with cheap indexing
- [Neo4j GraphRAG for Python](https://github.com/neo4j/neo4j-graphrag-python) - Official Neo4j package for building knowledge graphs and graph retrievers
- [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) - Benchmark and analysis of when graph structure actually helps RAG (ICLR'26)

## Evaluation & Benchmarks

Tools and benchmarks for measuring retrieval and RAG quality:

- [Ragas](https://github.com/vibrantlabsai/ragas) - Metrics and synthetic test-set generation for evaluating RAG pipelines
- [DeepEval](https://github.com/confident-ai/deepeval) - Pytest-style LLM and RAG evaluation for CI regression gates
- [Phoenix](https://github.com/Arize-ai/phoenix) - OpenTelemetry-based tracing and evaluation for LLM and retrieval pipelines (ELv2)
- [TruLens](https://github.com/truera/trulens) - Feedback-function evaluation and tracing for RAG apps (RAG triad)
- [MTEB](https://github.com/embeddings-benchmark/mteb) - Massive Text Embedding Benchmark and leaderboard for choosing embedding models
- [ViDoRe V3](https://arxiv.org/abs/2601.08620) - Benchmark for visual document retrieval and RAG in enterprise scenarios

## Guides & Research

Essays, research, and reference architectures on building RAG systems:

- [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - Anthropic's guide to curating what goes into an agent's context window
- [Context Rot](https://www.trychroma.com/research/context-rot) - Chroma's research on how LLM performance degrades as input length grows
- [How We Built Our Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system) - Anthropic's engineering write-up on agentic search and research
