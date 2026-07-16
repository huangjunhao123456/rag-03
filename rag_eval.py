import asyncio
import torch
from langchain_huggingface import HuggingFaceEmbeddings
#from retrieve import rephrase_retrieve, get_rag_chain, get_llm, get_retriever
# from retrieve_tuning_before_1 import rephrase_retrieve, get_rag_chain, get_llm, get_retriever
# from retrieve_tuning_before_2 import rephrase_retrieve, get_rag_chain, get_llm, get_retriever
# from retrieve_tuning_after_1 import rephrase_retrieve, get_rag_chain, get_llm, get_retriever
# from retrieve_tuning_after_2 import rephrase_retrieve, get_rag_chain, get_llm, get_retriever
from retrieve_tuning_hy import rephrase_retrieve, get_rag_chain, get_llm, get_retriever, get_bm25_retriever

from datasets import Dataset
from ragas.metrics import ContextRelevance, answer_relevancy, faithfulness, ResponseGroundedness
from ragas import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 存储对话历史
chat_history = []
# 将需要评估的数据存储起来
retrieve_history = []

# 1、初始化Embedding模型
embedding_model = HuggingFaceEmbeddings(
    model_name="./model/bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)

# 2、初始化 LLM
llm = get_llm()

# rerank_tokenizer = AutoTokenizer.from_pretrained("./model/bge-reranker-base")
# rerank_model = AutoModelForSequenceClassification.from_pretrained("./model/bge-reranker-base")
"""
ragas进行评估：
 - 用户问题：query
 - 上下文:retrieve_result
 - 模型回复:answer
"""
async def invoke_rag(query,conversation_id,chat_history):

    answer = ""

    input={"query":query,"history":chat_history}

    # 1、获取检索器
    retriever=get_retriever(k=20,embedding_model=embedding_model)
    bm25_retriever = get_bm25_retriever()
    
    # 2、执行重述、检索
    #retrieve_result= rephrase_retrieve(input,llm,retriever) #普通检索
    # retrieve_result = rephrase_retrieve(input, llm, retriever, 4) #多查询
    # retrieve_result = rephrase_retrieve(input, llm, retriever) # 假设性文档：HyDE
    # retrieve_result = rephrase_retrieve(input, llm, retriever, 4) #多查询+RRF重排
    # retrieve_result = rephrase_retrieve(input, llm, retriever,  rerank_tokenizer, rerank_model) #使用Reranker模型重排序
    
    retrieve_result = rephrase_retrieve(input, llm, retriever, bm25_retriever) #多查询+RRF重排

    # 3、获取RAG链
    rag_chain = get_rag_chain(retrieve_result,llm)
    # 4、异步执行RAG链，流式输出
    async for chunk in rag_chain.astream(input):
        answer += chunk
        yield chunk # 将大模型生成的内容逐块(chunk)地返回给调用者，而不是等待整个回答完成后一次性返回

    # 5、更新对话历史，添加用户查询和AI回答
    chat_history.append(
        {"role": "user", "content": query, "conversation_id": conversation_id}
    )
    chat_history.append(
        {"role": "ai", "content": answer, "conversation_id": conversation_id}
    )

    # 存储数据，供后续进行评估
    retrieve_history.append({
        "query": query,
        "contexts": [
            doc.page_content for doc in retrieve_result
        ],
        "answer": answer
    })

def rag_evaluate(datas):
    """
        使用RAGAS 对RAG进行评估
    """
    # 1.构建评估数据集
    ragas_data = {
        "user_input": [d["query"] for d in datas],  # 用户查询
        "response": [d["answer"] for d in datas],  # AI回答
        "retrieved_contexts": [d["contexts"] for d in datas],  # 检索到的上下文
    }
    dataset = Dataset.from_dict(ragas_data)

    # 2.定义评估指标
    metrics = [
        ContextRelevance(), #上下文的相关性
        answer_relevancy,  # 回复的相关性
        faithfulness,  # 可信度
        ResponseGroundedness() # 响应的真实性
    ]

    # 3.执行评估
    result = evaluate(
        dataset, 
        metrics,
        llm,
        embeddings=embedding_model
    )

    datas.clear()
    return result


if __name__ == '__main__':
    async def main():
        # query_list = ["中国科学院国家天文台2023年部门预算总额是多少", "该预算中，科学技术支出具体是多少？"]
        query_list = ["不动产或者动产被人占有怎么办", "那要是被损毁了呢"]
        # query_list = ["因意外事件下落不明申请宣告死亡，需要满足多长时间？","什么情况下不受该时间限制"]
        # query_list = ["未成年人的父母已经死亡或没有监护能力时，法定监护人按什么顺序担任？","监护人出现哪些情形时，人民法院可以根据申请撤销其监护人资格？"]
        # query_list = ["2023 年国家天文台一般公共预算拨款收入为多少万元？", "占收入总计的比例是多少？？"]
        for query in query_list: 
            print(f"===== 查询: {query} =====")
            async for chunk in invoke_rag(query,1,chat_history):
                print(chunk, end="", flush=True)

        ############################
        print("\n\n RAG 评估结果如下：-------------------------------------")
        eva_res = rag_evaluate(retrieve_history)  

        # 输出评估结果的关键指标
        import pandas as pd
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(
            eva_res.to_pandas()[
                [
                    "nv_context_relevance",
                    "answer_relevancy",
                    "faithfulness",
                    "nv_response_groundedness",
                ]
            ]
        )

    asyncio.run(main())