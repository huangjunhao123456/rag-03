import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

embeddings_model = HuggingFaceEmbeddings(
    model_name="./model/bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)

vectorstore = Chroma(
    embedding_function=embeddings_model,
    persist_directory="./vectorstore" #
)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
#LCEL方式来构建检索生成过程
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    你是一个专业的中文问答助手，擅长基于提供的资料回答用户问题。
    请仅根据以下背景资料回答问题，如无法找到答案，请直接回答“我不知道”。
    
    背景资料：{context}
    
    问题：{question}

    回答：
    """,
)

def format_docs(docs):
    formatted_docs = "\n\n".join(doc.page_content for doc in docs)
    return formatted_docs

llm = init_chat_model("qwen-plus",model_provider="openai")

# pipeline 数据管道
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  
    | prompt 
    | llm
    | StrOutputParser()
    )