from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

model = None

def load():
    global model
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')  

def embed_query(text:str):
    instruction = "为这个句子生成表示以用于检索相关文章："   
    q_embeddings = model.encode([instruction+text], normalize_embeddings=True)    
    return q_embeddings[0]

def embed_documents(texts:List[str]):
    p_embeddings = model.encode(texts, normalize_embeddings=True)
    return p_embeddings
