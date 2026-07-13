import torch
from langchain_huggingface import HuggingFaceEmbeddings
from retrieve import get_retriever, rephrase_retrieve, get_llm, get_rag_chain

embeddings_model = HuggingFaceEmbeddings(
    model_name="./model/bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)

chat_history = []

llm = get_llm()

# 1. 获取检索器
retriever = get_retriever(k=20, embedding_model=embeddings_model)

while True:
    query = input("请输入问题：")
    message={"query": query, "history": chat_history}

    # 2. 用户问题重述，并根据重述后的问题检索
    retrieve_result = rephrase_retrieve(message, llm, retriever)

    # 3. 获取 rag 链
    rag_chain = get_rag_chain(retrieve_result, llm)

    # 4. 执行 rag 链
    result = rag_chain.invoke(message) # astream
    print(result)
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "ai", "content": result})