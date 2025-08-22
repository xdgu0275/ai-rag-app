import json
import os
import sys
import logging
import traceback
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # 新版 Chroma 集成
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.docstore.document import Document

# 配置日志 - 同时输出到控制台和文件（UTF-8编码）
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_text.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# ================== 配置 ==================
current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(current_dir, "docs.json")  # 输入文档JSON路径
VECTOR_DB_PATH = os.path.abspath(os.path.join(current_dir, "..", "vectorstore"))  # 向量库存储路径
EMBEDDING_MODEL = "text-embedding-v1"  # DashScope嵌入模型

# 设置API Key（建议通过环境变量加载，此处仅为示例）
os.environ["DASHSCOPE_API_KEY"] = "sk-7940c68d86644583bc69778b8651d7e2"
api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    logger.error("❌ 未找到 DASHSCOPE_API_KEY 环境变量！")
    exit(1)
logger.info("✅ API Key 已加载")


# ================== 主程序 ==================
def main():
    logger.info("🚀 开始文本分块与向量化...")

    # 1. 读取输入文档JSON
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            docs = json.load(f)
        logger.info(f"✅ 加载 {len(docs)} 个文档")
    except Exception as e:
        logger.error(f"❌ 读取 docs.json 失败: {e}")
        traceback.print_exc()
        exit(1)

    # 2. 初始化文本分块器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每块最大字符数
        chunk_overlap=50,  # 块间重叠字符数（保证上下文连贯）
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]  # 中文优先分隔符
    )

    # 3. 文档分块处理
    chunked_documents = []
    total_chunks = 0

    for doc in docs:
        if doc["status"] != "success":
            logger.warning(f"⚠️ 跳过状态异常文档: {doc.get('filename', '未知')}（状态: {doc['status']}）")
            continue

        content = doc["content"]
        filename = doc["filename"]
        doc_type = doc["type"]

        try:
            # 分块处理
            chunks = text_splitter.split_text(content)
            logger.info(f"✅ 文档 {filename} 分块完成，生成 {len(chunks)} 个块")

            # 转换为LangChain Document对象（包含元数据）
            for i, chunk in enumerate(chunks):
                chunked_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk_id": i,
                            "type": doc_type
                        }
                    )
                )
                total_chunks += 1
        except Exception as e:
            logger.error(f"❌ 处理文档 {filename} 时出错: {e}")
            traceback.print_exc()
            continue

    if total_chunks == 0:
        logger.warning("⚠️ 未生成任何知识块，程序退出")
        exit(0)
    logger.info(f"✅ 文本分块完成！共生成 {total_chunks} 个知识块")

    # 4. 初始化嵌入模型（使用DashScope，替代旧的ONNX嵌入）
    logger.info("🧠 正在初始化嵌入模型...")
    try:
        embedding_function = DashScopeEmbeddings(
            model_name=EMBEDDING_MODEL,
            api_key=api_key
        )
        # 测试嵌入模型有效性
        test_embedding = embedding_function.embed_query("测试嵌入")
        logger.info(f"✅ 嵌入模型初始化成功（维度: {len(test_embedding)}）")
    except Exception as e:
        logger.error(f"❌ 初始化嵌入模型失败: {e}")
        traceback.print_exc()
        exit(1)

    # 5. 初始化向量数据库（使用langchain-chroma适配新版）
    logger.info("🧠 正在初始化向量数据库...")
    try:
        # 创建/连接向量库
        db = Chroma(
            collection_name="document_collection",
            embedding_function=embedding_function,
            persist_directory=VECTOR_DB_PATH  # 持久化到本地（避免内存模式数据丢失）
        )
        logger.info(f"✅ 向量数据库初始化成功（存储路径: {VECTOR_DB_PATH}）")
    except Exception as e:
        logger.error(f"❌ 初始化向量数据库失败: {e}")
        traceback.print_exc()
        exit(1)

    # 6. 批量添加知识块到向量库
    try:
        logger.info("📥 开始添加知识块到向量库...")
        start_time = time.time()

        # 批量添加（自动处理嵌入生成）
        db.add_documents(documents=chunked_documents)
        db.persist()  # 持久化数据

        end_time = time.time()
        logger.info(f"✅ 成功添加 {len(chunked_documents)} 个知识块（耗时: {end_time - start_time:.2f}秒）")

        # 验证添加结果
        collection = db.get()
        logger.info(f"✅ 向量库当前总知识块数量: {len(collection['ids'])}")
    except Exception as e:
        logger.error(f"❌ 添加知识块失败: {e}")
        traceback.print_exc()
        exit(1)

    # 7. 测试语义搜索功能
    logger.info("\n==============================")
    logger.info("🔍 测试语义搜索...")
    test_query = "报告写了什么？"

    try:
        results = db.similarity_search(query=test_query, k=2)  # 搜索Top2结果
        if results:
            for i, doc in enumerate(results, 1):
                logger.info(f"\n🎯 匹配结果 {i}:")
                logger.info(f"来源: {doc.metadata['source']} (块ID: {doc.metadata['chunk_id']})")
                logger.info(f"内容: {doc.page_content[:200]}...")
        else:
            logger.info("⚠️ 未找到匹配的结果")
    except Exception as e:
        logger.error(f"❌ 搜索失败: {e}")
        traceback.print_exc()

    logger.info("\n🎉 所有流程执行完成")


if __name__ == "__main__":
    main()