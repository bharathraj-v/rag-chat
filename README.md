# RAG Chatbot for NVIDIA CUDA Documentation Site

#### Working screenshot:

<p align="center">
<img src="working_screenshot.png" width="800">
</p>


#### Info:
* The data is sourced from https://docs.nvidia.com/cuda/, scraped up to 5 levels of sub-link depth using Scrapy.
* The scraped data was chunked using semantic chunking with HuggingFace's sentence-transformers/all-mpnet-base-v2 embedding model using Langchain's Semantic Chunker.
* The data is converted to HF embedding vectors and stored alongside the metadata in MILVUS indexed by HNSW.
* BGE-M3 colbert+sparse+dense re-ranking pipeline is being used to re-rank MILVUS vector search results. Local hybrid BGE-M3 re-ranking is better but was not chosen due to compute limitations.
* Mixtral-8x7B-Instruct-v0.1 LLM via the HF inference API is being used. Ideally, a larger and locally hosted model should yield the best performance, but compute limitatons.
* Gradio is being used for the user interface.

#### Instructions to run the demo:
1. `pip install -r requirements.txt`
2. Set up your MILVUS account and modify the Milvus client code according to your set-up - (different for docker, zilliz, etc.)
3. Set up HUGGINGFACE_TOKEN env variable in .env file
4. Run the gradio_app.py via `gradio_app.py`
