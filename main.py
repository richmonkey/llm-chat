import faiss   
from knowledge_file import KnowledgeFile
import uuid
import embedding
import numpy as np

if __name__ == "__main__":
    embedding.load()
    f = KnowledgeFile("test.pdf", "")
    docs = f.file2docs()
    txtdocs = f.docs2texts(docs)    
    print("docs:", docs) 
    print("text docs:", txtdocs)
    texts = [doc.page_content for doc in txtdocs]
    metadatas = [doc.metadata for doc in docs]
    embeddings = embedding.embed_documents(texts=texts)
    print("doc embedings:", embeddings)
    print("metadatas:", metadatas)
    ids = [str(uuid.uuid1()) for _ in range(len(texts))]
    q_embedding = embedding.embed_query("hello")
    print("query embeding:", q_embedding)
    dimension = len(q_embedding)
    flat_index = faiss.IndexFlatIP(dimension)    
    assert flat_index.is_trained

    flat_index.add(embeddings)
    query = q_embedding[np.newaxis, :]
    D, I = flat_index.search(query, 4)
    print("D:", D)
    print("I:", I)
    #TODO rerank