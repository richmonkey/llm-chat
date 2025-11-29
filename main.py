from knowledge_file import KnowledgeFile
import embedding
import rerank
import numpy as np
from docx.document import Document
import sys
# faiss必须在sentence_transformers之后导入， 否则程序会崩溃
import faiss   

class FaissSearch:
    flat_index: faiss.IndexFlatIP
    documents: list[Document]
    def __init__(self):
        q_embedding = embedding.embed_query("hello")
        dimension = len(q_embedding)
        flat_index = faiss.IndexFlatIP(dimension)    
        assert flat_index.is_trained
        self.flat_index = flat_index
        self.documents = []
        

    def add_file(self, filename:str, flat_index: faiss.IndexFlatIP):
        f = KnowledgeFile(filename, "")

        docs:list[Document] = f.file2docs()
        txtdocs: list[Document] = f.docs2texts(docs)    

        texts = [doc.page_content for doc in txtdocs]
        metadatas = [doc.metadata for doc in txtdocs]
        embeddings = embedding.embed_documents(texts=texts)
        print("doc embedings:", embeddings)
        print("metadatas:", metadatas)
        print("texts:", texts)
        self.flat_index.add(embeddings)    
        self.documents.extend(txtdocs)        

    def search(self, qs:str):
        q_embedding = embedding.embed_query(qs)
        print("query embeding:", q_embedding)

        query = q_embedding[np.newaxis, :]
        D, I = self.flat_index.search(query, 4)
        print("D:", D.shape, D)
        print("I:", I.shape, I)        

        # 根据索引获取文档内容
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.documents) and idx >= 0:  # 确保索引有效
                results.append({
                    'rank': i + 1,
                    'document': self.documents[idx],
                    'similarity': distance,
                    'index': idx
                })
        
        return results        

    
if __name__ == "__main__":
    embedding.load()
    rerank.load()

    flat_index = FaissSearch()
    
    flat_index.add_file("test.pdf", flat_index)
    flat_index.add_file("test.html", flat_index)
    flat_index.add_file("test.txt", flat_index)    
    flat_index.add_file("test.docx", flat_index)

    q = "Faiss时踩过的坑"
    res = flat_index.search(q)
    docs = [r["document"] for r in res]
    print("search res:", [(r["similarity"], r["document"].metadata) for r in res])

    results = rerank.rerank(q, docs)
    for doc, score in results:
        print(f"得分: {score:.4f} - {doc.metadata}")    

