from typing import Dict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def format_history(history,max_epoch=3):
    # 每轮对话有 用户问题 和 助手回复
    if len(history) > 2 * max_epoch:
        history = history[-2 * max_epoch :]
    return "\n".join([f"{i['role']}：{i['content']}" for i in history])

def format_docs(docs: list[Document]) -> str:
    # """拼接多个 Document 的 page_content """
    # return "\n\n".join(doc.page_content for doc in docs)
    """将检索到的 Document 格式化为提示词上下文。"""
    if not docs:
        return "未检索到相关资料。"

    parts = []
    for doc in docs:
        page = doc.metadata.get("page_number", 0)
        source = doc.metadata.get("source", "未知文件")
        parts.append(
            f"[来源：{source}，第 {page} 页]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)
def get_retriever(k=20,embedding_model=None):
    """获取向量数据库的检索器"""
    # 1、初始化 Chroma 客户端
    vectorstore = Chroma(
        persist_directory="vectorstore2",
        embedding_function=embedding_model,
    )

    # 2、创建向量数据库检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 检索方式，similarity 或 mmr
        search_kwargs={"k": k},
    )
    return retriever

def get_llm():
    # 大模型
    load_dotenv()
    llm = init_chat_model(model="qwen-plus",model_provider="openai")
    return llm

def rephrase_retrieve(input:Dict[str,str], llm, retriever):
    """重述用户query，检索向量数据库"""

    # 1、重述query的prompt
    rephrase_prompt = PromptTemplate.from_template(
    """
    根据对话历史简要完善最新的用户消息，使其更加具体。只输出完善后的问题。如果问题不需要完善，请直接输出原始问题。
    
    {history}
    用户：{query}
    """
    )

    # 2、重述链条：根据历史和当前 query 生成更具体问题
    rephrase_chain = (
        {
            "history": lambda x :format_history(x.get("history")),
            "query": lambda x: x.get("query"),
        }
        | rephrase_prompt
        | llm
        | StrOutputParser()
        | (lambda x: print(f"===== 重述后的查询: {x}=====") or x)
    )

    # ---------------------------检索前优化：HyDE假设文档----------------------
    # 3、HyDE提示模板
    hyde_prompt = PromptTemplate.from_template(
        """
        请根据常识和推理，为问题编写一段看起来合理且详细的回答性段落，哪怕你不确定真实答案。
        问题：{query}
        """
    )
    hyde_chain = hyde_prompt | llm | StrOutputParser() # 根据问题生成答案

    # 4、HyDE 链条
    retrieve_chain = (
        rephrase_chain
        | hyde_chain
        | (lambda x: print(f"===== 假设文档：{x}=====") or x)
        | (lambda x: retriever.invoke(x))
    )

    retriever_result = retrieve_chain.invoke({"history": input.get("history"),"query": input.get("query")})
    return retriever_result

def get_rag_chain(retrieve_result,llm):
    """构建RAG链条：使用检索结果、历史记录、用户查询，提交大模型生成回复"""

    # 1、Prompt 模板
    prompt = PromptTemplate(
        input_variables=["context", "history", "query"],
        template="""
    你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
    请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。
    回答使用中文，尽量简洁；如涉及事实，请在对应句子后标注来源和页码，例如“来源：xx.pdf 文档 （第 3 页）”

    背景资料：{context}
    
    历史消息：[{history}]
    
    问题：{query}
    
    回答：""",
    )

    # 2、定义 RAG 链条
    rag_chain = (
        {
            "context": lambda x:format_docs(retrieve_result),
            "history": lambda x: format_history(x.get("history")),
            "query": lambda x: x.get("query"),
        }
        | prompt
        | (lambda x: print(x.text, end="") or x) #打印
        | llm
        | StrOutputParser()  # 输出解析器，将输出解析为字符串
    )

    return rag_chain