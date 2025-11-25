from sentence_transformers import CrossEncoder

# 带配置的模型加载
model = CrossEncoder(
    'BAAI/bge-reranker-v2-m3',
    max_length=512,           # 最大序列长度
    device='cpu',            # 使用GPU加速
)

# 自定义预测参数
def advanced_rerank(query, documents, model):
    pairs = [(query, doc) for doc in documents]
    
    # 使用详细配置进行预测
    scores = model.predict(
        pairs,
        batch_size=16,           # 批处理大小
        show_progress_bar=True,  # 显示进度条
        convert_to_numpy=True,   # 转换为numpy数组
        activation_fct=None      # 不使用激活函数（原始分数）
    )
    
    return list(zip(documents, scores))

# 使用示例
query = "深度学习框架"
documents = ["TensorFlow是Google开发的深度学习框架", 
            "PyTorch由Facebook开发，研究友好",
            "Keras是高级神经网络API"]

results = advanced_rerank(query, documents, model)
for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"得分: {score:.4f} - {doc}")