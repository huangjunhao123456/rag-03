
1、RAG系统架构
    核心模块：
        - 索引：加载、分块、嵌入、存储
        - 检索：查询嵌入-》相似度计算-》提取上下文
        - 生成：提示词工程-》LLM调度-》后处理
        - 服务接口：REST API-》流式响应-》监控
    主流技术栈：
        - 嵌入模型：SentenceTransformers(bge)
        - 向量数据库：Chroma(轻量级基于内存)、Milvus、FAISS
        - LLM:在线的开源模型、vLLM
        - 服务框架：FastAPI

2、索引过程
    文本加载：
        基础格式：TextLoader、UnstructuredMarkdownLoader
        pdf处理：PyPDFLoader(支持页级拆分、布局提取)、UnstructuredPDFLoader（支持OCR和结构化提取，需要配置Poppler和Tesseract ORC，策略推荐hi_res以识别表格和布局）
        其他：UnstructuredWordDocumentLoader、WebBaseLoader。。。
    文本清洗
        内容清洗：正则表达式去除换行、空格、异常字符 等等
        元数据清洗：Chroma向量数据库的metadata仅支持str、int、float、bool
    文本分块
        常用方法：RecursiveCharacterTextSpliter(递归按字符切分)
        核心参数：chunk_size、chunk_overlap
        表格处理：model="elements" 当 category=“Table”,应单独保留该元素（优先实现text_as_html保留结构）
    向量嵌入和存储
        使用HuggfaceEmbeddings加载模型，开启normalize_embeddings=True优化余弦相似度计算的
        存入Chroma的Collection,包含：ID、嵌入向量、文档内容、元数据键值对

3、检索过程
    - 相似度检索：基于余弦相似度返回Top-k最相关的块
    - MMR:最大边际相关性检索，在保证相关性的同时，惩罚已选块的相似度，减少结果重复、提供多样性
    - Top-k:20，为了提高模型的回复能力，一般建议设置20左右
4、生成过程
    LCEL使用管道符（|）将组件串联在一起（Retrieve|Prompt|LLM|OutputParser）
    多轮对话
        - 历史记录管理：截取、压缩
        - 查询重述：针对多轮对话中的“指代消解”
    Agent模式->Agentic RAG


5、工程化和部署
    模块化设计：索引、检索、执行链、API封装
    流式响应

RAG常见的面试题
1、pdf中表格处理
    避免暴露切分
    结构化提取
    分类处理
    保留格式：html/minerU（json）

2、分块，chunk_size|chunk_overlap
    chunk_size:300~500
    chunk_overlap:10%~20%

3、相似度检索、MMR检索