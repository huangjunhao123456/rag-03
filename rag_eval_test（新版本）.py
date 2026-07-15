import asyncio
import torch
from langchain_huggingface import HuggingFaceEmbeddings
#from retrieve import rephrase_retrieve, get_rag_chain, get_llm, get_retriever
from retrieve_tuning_before_1 import rephrase_retrieve, get_rag_chain, get_llm, get_retriever

from ragas.metrics.collections import ContextRelevance, AnswerRelevancy, Faithfulness, ResponseGroundedness
from ragas.llms.base import llm_factory
from openai import AsyncOpenAI
import pandas as pd
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

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

eval_embeddings = RagasHuggingFaceEmbeddings(
        model="./model/bge-base-zh-v1.5",
        device="cuda" if torch.cuda.is_available() else "cpu",
        normalize_embeddings=True,
)

# 2、初始化 LLM
llm = get_llm()
client = AsyncOpenAI()
eval_llm = llm_factory(client=client, model="qwen-plus")
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
    retriever=get_retriever(k=5,embedding_model=embedding_model)
    # 2、执行重述、检索
    #retrieve_result= rephrase_retrieve(input,llm,retriever)
    retrieve_result = rephrase_retrieve(input, llm, retriever, 3) #多查询

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

async def rag_evaluate(datas):
    """
        使用RAGAS 对RAG进行评估
    """
    

    metrics = {
        "nv_context_relevance": ContextRelevance(llm=eval_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=eval_llm,
            embeddings=eval_embeddings,
        ),
        "faithfulness": Faithfulness(llm=eval_llm),
        "nv_response_groundedness": ResponseGroundedness(llm=eval_llm),
    }
    
    rows = []

    for data in datas:
        results = await asyncio.gather(
            # 只评估：检索上下文是否与问题相关
            metrics["nv_context_relevance"].ascore(
                user_input=data["query"],
                retrieved_contexts=data["contexts"],
            ),

            # 只评估：回答是否回答了问题
            metrics["answer_relevancy"].ascore(
                user_input=data["query"],
                response=data["answer"],
            ),

            # 评估：回答中的事实能否被上下文支撑
            metrics["faithfulness"].ascore(
                user_input=data["query"],
                response=data["answer"],
                retrieved_contexts=data["contexts"],
            ),

            # 评估：回答是否由检索上下文支撑
            metrics["nv_response_groundedness"].ascore(
                response=data["answer"],
                retrieved_contexts=data["contexts"],
            ),
        )

        rows.append({
            "nv_context_relevance": results[0].value,
            "answer_relevancy": results[1].value,
            "faithfulness": results[2].value,
            "nv_response_groundedness": results[3].value,
        })

    datas.clear()
    return pd.DataFrame(rows)


if __name__ == '__main__':
    async def main():
        query_list = ["中国科学院国家天文台2023年部门预算总额是多少", "该预算中，科学技术支出具体是多少？"]
        for query in query_list: 
            print(f"===== 查询: {query} =====")
            async for chunk in invoke_rag(query,1,chat_history):
                print(chunk, end="", flush=True)

        ############################
        print("\n\n RAG 评估结果如下：-------------------------------------")
        eva_res = await rag_evaluate(retrieve_history)  

        # 输出评估结果的关键指标
        import pandas as pd
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(
        eva_res[
            [
                "nv_context_relevance",
                "answer_relevancy",
                "faithfulness",
                "nv_response_groundedness",
            ]
        ]
    )

    asyncio.run(main())