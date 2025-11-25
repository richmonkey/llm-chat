from sentence_transformers import CrossEncoder
from langchain.docstore.document import Document
import numpy as np

model = None


def load():
    global model
    model = CrossEncoder(
        'BAAI/bge-reranker-v2-m3',
        max_length=1024,           # 最大序列长度
        device='cpu',            # 使用GPU加速
    )    
    
# 自定义预测参数
def rerank(query:str, documents:list[Document]) -> list[tuple[Document, float]]:
    pairs = [(query, doc.page_content) for doc in documents]

    scores:np.array = model.predict(
        pairs,
        batch_size=32,           # 批处理大小
        show_progress_bar=True,  # 显示进度条
        convert_to_numpy=True,   # 转换为numpy数组
        activation_fn=None      # 不使用激活函数（原始分数）
    )

    results = list(zip(documents, scores))
    return sorted(results, key=lambda x: x[1], reverse=True)
        

def advanced_rerank(query:str, documents:list[Document]):
    pairs = [(query, doc) for doc in documents]

    scores:np.array  = model.predict(
        pairs,
        batch_size=32,           # 批处理大小
        show_progress_bar=True,  # 显示进度条
        convert_to_numpy=True,   # 转换为numpy数组
        activation_fct=None      # 不使用激活函数（原始分数）
    )

    results = list(zip(documents, scores))
    return sorted(results, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":

    load()
    # 使用示例
    query = "深度学习框架"
    documents = ["TensorFlow是Google开发的深度学习框架", 
                "PyTorch由Facebook开发，研究友好",
                "Keras是高级神经网络API"]

    #results = advanced_rerank(query, documents)
    results = rerank(query, [1, 2, 3])    
    for doc, score in results:
        print(f"得分: {score:.4f} - {doc}")