import os
import sys
import logging
# 使用新版chroma包
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

# 配置日志 - 同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_vectorstore.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置DASHSCOPE_API_KEY环境变量
os.environ['DASHSCOPE_API_KEY'] = 'sk-7940c68d86644583bc69778b8651d7e2'

# 检查API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
if not api_key:
    logger.error('未找到 DASHSCOPE_API_KEY 环境变量！')
    sys.exit(1)
logger.info('API Key 已加载')

# 初始化embedding模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=api_key
)
logger.info('embedding模型初始化完成')

# 尝试使用内存模式的Chroma向量库
logger.info("正在初始化内存模式向量数据库...")

try:
    # 创建一个简单的文档
    doc = Document(
        page_content="这是一个测试文档，用于验证向量库添加功能。",
        metadata={"source": "test_doc.txt"}
    )
    logger.info(f"创建测试文档: {doc.page_content}")

    # 配置Chroma，禁用匿名遥测并使用内存模式
    from chromadb.config import Settings
    chroma_settings = Settings(
        anonymized_telemetry=False,
        # 内存模式不需要设置persist_directory
    )

    # 初始化Chroma向量库
    vectorstore = Chroma(
        embedding_function=embedding_model,
        client_settings=chroma_settings
    )
    logger.info(f"成功创建内存模式 Chroma 实例")

    # 测试向量库基本功能
    logger.info("测试向量库集合创建...")
    collection = vectorstore._client.create_collection("test_collection")
    logger.info("向量库集合创建成功")

    # 测试集合基本操作 - 使用手动生成的embedding
    logger.info("测试集合添加数据...")
    try:
        # 使用DashScopeEmbeddings生成嵌入向量
        embeddings = embedding_model.embed_documents([doc.page_content])
        logger.info(f"成功生成embedding，长度: {len(embeddings[0])}")
        
        # 手动添加文档和嵌入向量
        logger.info("开始调用collection.add()...")
        collection.add(
            documents=[doc.page_content],
            metadatas=[doc.metadata],
            ids=["test_id_1"],
            embeddings=embeddings
        )
        logger.info("collection.add()调用完成")
        
        # 验证添加结果
        count = collection.count()
        logger.info(f"集合中当前文档数量: {count}")
        if count > 0:
            logger.info("集合添加数据成功")
        else:
            logger.warning("集合添加数据后数量仍为0，可能添加失败")
    except Exception as e:
        logger.error(f"集合添加数据时出错: {str(e)}", exc_info=True)

    # 测试查询功能
    logger.info("测试集合查询功能...")
    results = collection.query(query_texts=["测试"])
    logger.info(f"查询结果: {results}")

    # 尝试使用LangChain的add_documents方法
    logger.info("开始使用add_documents添加文档...")
    result = vectorstore.add_documents([doc])
    logger.info(f"add_documents添加完成，结果: {result}")

    # 验证添加结果
    logger.info("验证添加结果...")
    try:
        collection = vectorstore._client.get_collection("langchain")
        count = collection.count()
        logger.info(f"向量库中共有 {count} 个向量")
    except Exception as e:
        logger.error(f"获取向量库计数失败: {str(e)}")

    logger.info("测试完成！")

except Exception as e:
    logger.error(f"测试过程中出错: {str(e)}", exc_info=True)
    sys.exit(1)