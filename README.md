# RAG Chatbot for NVIDIA CUDA Documentation Site

#### Info:
* The data for the RAG Chatbot is sourced from https://docs.nvidia.com/cuda/, scraped up to 5 levels of sub-link depth using Scrapy.
* The scraped data was chunked using semantic chunking with HuggingFace's sentence-transformers/all-mpnet-base-v2 embedding model using Langchain's Semantic Chunker.
* The data is converted to HF embedding vectors and stored alongside the metadata in MILVUS indexed by HNSW.
* BGE-M3 colbert+sparse+dense re-ranking pipeline is being used to re-rank MILVUS vector search results.
* Mixtral-8x7B-Instruct-v0.1 LLM via the HF inference API is being used. Ideally, a larger and a locally hosted model should yield best performance.
* Gradio is being used as the UI interface.
