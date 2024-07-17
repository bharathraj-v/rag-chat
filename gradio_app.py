
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import MilvusClient, DataType
import os
import requests
from huggingface_hub import InferenceClient
import json
import gradio as gr

hf_embeddings = HuggingFaceEmbeddings()


CLUSTER_ENDPOINT = "https://in03-6f08bdb85fec6ea.api.gcp-us-west1.zillizcloud.com"
TOKEN = os.getenv('ZILLIZ_TOKEN')

client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN 
)

API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-m3"
headers = {"Authorization": "Bearer "+str(os.getenv('HUGGINGFACE_TOKEN'))}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": {
	"source_sentence": ["placeholder_to_load_the_model"],
	"sentences":["placeholder_to_load_the_model", "placeholder_to_load_the_model"],
},
})


llm_client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    timeout=120,
    token=str(os.getenv('HUGGINGFACE_TOKEN'))
)


def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]


call_llm(llm_client, "This is a test context")

def chatbot_chat(question, history):
    embedded_query = hf_embeddings.embed_query(question)

    client.load_collection(
    collection_name="cuda_data",
    )

    res = client.search(
        collection_name="cuda_data", 
        data=[embedded_query],
        limit=10, 
        search_params={"metric_type": "COSINE", "params": {}}
    )
    ids = [i["id"] for i in res[0]]
    res = client.get(
        collection_name="cuda_data",
        ids=ids
    )

    res_chunks = [[i["url"], i["chunk"]] for i in res]

    output = query({
        "inputs": {
        "source_sentence": question,
        "sentences":list([i[1] for i in res_chunks])
    },
    })

    context = "\nSource: ".join(res_chunks[output.index(max(output))])
    
    prompt = """
Your task is to answer a question given a context.
Question asked: """+ question+"""
Context provided to you from NVIDIA Documentation site: """+ context[:1500]+"""
You always answer in one or two paragraphs.
You always provide the URL source for context right after the answer.
Answer:
"""
    res = call_llm(llm_client,prompt)
    return res.replace(prompt, "")


demo = gr.ChatInterface(fn=chatbot_chat, title="NVIDIA CUDA Documentation Chatbot")
demo.launch()
