import json
import os
import sys
import logging
import traceback
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # 更新为新的导入
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.docstore.document import Document

# 配置日志 - 同时输出到控制台和文件，使用UTF-8编码
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_text.log')
# 设置文件处理器使用UTF-8编码
file_handler = logging.FileHandler(log_file, encoding='utf-8')
# 设置控制台处理器
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# ================== 配置 ==================
# 获取当前脚本的绝对路径（比如 D:\ai-rag-app\src\split_text.py）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建其他文件的绝对路径
INPUT_JSON = os.path.join(current_dir, "docs.json")
# 向量数据库保存路径
VECTOR_DB_PATH = os.path.abspath(os.path.join(current_dir, "..", "vectorstore")) # ../vectorstore
# DashScope Embedding 模型
EMBEDDING_MODEL = "text-embedding-v1"

# 设置 DASHSCOPE_API_KEY 环境变量
os.environ["DASHSCOPE_API_KEY"] = "sk-7940c68d86644583bc69778b8651d7e2"

# 检查 API Key 是否设置
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    logger.error("未找到 DASHSCOPE_API_KEY 环境变量！")
    exit(1)
logger.info("API Key 已加载")

# ================== 主程序 ==================
def main():
    logger.info("🚀 开始文本分块与向量化...")

    # 1. 读取昨天生成的 docs.json
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            docs = json.load(f)
        logger.info(f"✅ 加载 {len(docs)} 个文档")
    except Exception as e:
        logger.error(f"读取 docs.json 失败: {e}")
        traceback.print_exc()
        exit(1)

    # 2. 创建文本分块器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每块 500 字符
        chunk_overlap=50,  # 块之间重叠 50 字符（防止断句）
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )

    # 3. 提取文本并分块
    chunked_documents = []
    total_chunks = 0

    # 在分块后，创建 Document 对象
    for doc in docs:
        if doc["status"] != "success":
            logger.warning(f"跳过状态为 {doc['status']} 的文档: {doc.get('filename', '未知')}")
            continue

        content = doc["content"]
        filename = doc["filename"]

        try:
            chunks = text_splitter.split_text(content)
            logger.info(f"文档 {filename} 分块完成，生成 {len(chunks)} 个块")

            for i, chunk in enumerate(chunks):
                # 使用 Document 类
                chunked_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk_id": i,
                            "type": doc["type"]
                        }
                    )
                )
                total_chunks += 1
        except Exception as e:
            logger.error(f"处理文档 {filename} 时出错: {e}")
            traceback.print_exc()
            continue

    logger.info(f"✅ 文本分块完成！共生成 {total_chunks} 个知识块")

    # 4. 创建嵌入模型
    logger.info("🧠 正在生成向量 embeddings...（可能需要几秒/每块 ）")
    # 尝试使用ONNX Runtime或备用方案
    embedding_function = None
    try:
        # 尝试导入ONNX Runtime
        import onnxruntime
        logger.info(f"✅ ONNX Runtime已安装，版本: {onnxruntime.__version__}")
        
        # 禁用遥测
        import os
        os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
        logger.info("✅ 已禁用遥测")
        
        # 尝试设置ONNX Runtime执行提供程序以提高兼容性
        try:
            providers = onnxruntime.get_available_providers()
            logger.info(f"✅ ONNX Runtime可用提供程序: {providers}")
            # 使用CPU提供程序作为备选
            if "CPUExecutionProvider" in providers:
                os.environ["ONNX_RUNTIME_EXECUTION_PROVIDER"] = "CPUExecutionProvider"
                logger.info("✅ 已设置ONNX Runtime执行提供程序为CPUExecutionProvider")
        except Exception as ep_e:
            logger.warning(f"设置ONNX Runtime执行提供程序时出错: {ep_e}")
        
        # 使用ONNX Runtime嵌入函数
        try:
            from chromadb.utils import embedding_functions
            # 尝试使用本地模型或禁用远程加载
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("✅ 已启用离线模式")
            embedding_function = embedding_functions.OnnxEmbeddingFunction()
            logger.info("✅ 使用ONNX Runtime嵌入函数")
        except Exception as onnx_e:
            logger.error(f"❌ 初始化ONNX Runtime嵌入函数失败: {str(onnx_e)}")
            logger.info("🔄 尝试使用简单的嵌入函数替代方案")
            # 创建一个简单的嵌入函数包装类，提供name()方法
            class SimpleEmbeddingFunctionWrapper:
                def __init__(self, embedding_function):
                    self.embedding_function = embedding_function
                
                def __call__(self, input):
                    return self.embedding_function(input)
                
                def name(self):
                    return "simple_embedding_function"
            
            # 简单的基于字符长度的嵌入（仅用于演示）
            def simple_embedding_function(texts):
                return [[len(text)] for text in texts]
            
            # 使用包装类包装嵌入函数
            embedding_function = SimpleEmbeddingFunctionWrapper(simple_embedding_function)
            logger.info("✅ 使用简单的嵌入函数替代方案（带name()方法的包装类）")
    except ImportError as e:
        logger.error(f"❌ 导入ONNX Runtime失败: {str(e)}")
        logger.error("尝试安装依赖: pip install onnxruntime")
        raise
    except Exception as e:
        logger.error(f"❌ 初始化ONNX Runtime时发生其他错误: {str(e)}")
        logger.error("尝试使用sentence-transformers作为备选嵌入模型")
        
        # 尝试使用sentence-transformers作为备选
        try:
            from chromadb.utils import embedding_functions
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("✅ 已切换到sentence-transformers嵌入模型")
        except Exception as st_e:
            logger.error(f"❌ 初始化sentence-transformers嵌入模型失败: {str(st_e)}")
            logger.error("尝试安装依赖: pip install sentence-transformers")
            raise

    # 5. 分步生成向量并存入 Chroma
    logger.info("🧠 正在初始化向量数据库...")

    # 修改chromadb初始化部分的代码如下

    # 初始化chromadb客户端
    try:
        import chromadb
        logger.info("正在初始化chromadb客户端...")
        client = chromadb.EphemeralClient()  # 使用内存模式，避免配置问题
        logger.info("✅ chromadb客户端初始化成功（内存模式）")

        # 创建或获取集合
        collection_name = "document_collection"
        logger.info(f"正在创建或获取集合: {collection_name}")
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function  # ✅ 关键：把你的 embedding_function 传进去
        )
        logger.info(f"✅ 成功创建/获取集合: {collection_name}")

    except Exception as e:
        logger.error(f"❌ 初始化chromadb失败: {type(e).__name__}: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        exit(1)

    # 尝试添加文档（简化版）
    try:
        logger.info("准备添加文档到集合...")
        # 只取第一个文档进行测试
        if chunked_documents:
            doc = chunked_documents[0]
            logger.info(f"测试添加第一个知识块: '{doc.page_content[:50]}...'")

            # 准备数据
            doc_id = "test_doc_1"
            documents = [doc.page_content]
            metadatas = [doc.metadata]
            ids = [doc_id]

            logger.info(f"文档数据准备完毕，ID: {doc_id}")
            logger.info(f"文档内容前50字符: {doc.page_content[:50]}...")
            logger.info(f"元数据: {metadatas[0]}")

            # 尝试添加文档
            logger.info(f"开始执行collection.add()，ID: {doc_id}")
            start_add = time.time()
            try:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                end_add = time.time()
                logger.info(f"✅ collection.add()执行成功，耗时: {end_add - start_add:.2f}秒")

                # 验证添加结果
                logger.info("开始验证添加结果...")
                try:
                    count = collection.count()
                    logger.info(f"✅ 集合中的向量数量: {count}")

                    # 尝试检索
                    retrieved = collection.get(ids=[doc_id])
                    if len(retrieved['ids']) > 0:
                        logger.info(f"✅ 成功检索到添加的文档，ID: {retrieved['ids'][0]}")
                        logger.info(f"检索到的文档内容: {retrieved['documents'][0][:50]}...")
                        logger.info(f"检索到的元数据: {retrieved['metadatas'][0]}")
                    else:
                        logger.warning(f"⚠️ 未能检索到添加的文档，ID: {doc_id}")
                except Exception as verify_e:
                    logger.error(f"❌ 验证添加结果时出错: {type(verify_e).__name__}: {str(verify_e)}")
                    logger.error(f"错误详情: {traceback.format_exc()}")
            except Exception as add_e:
                logger.error(f"❌ 执行collection.add()时出错: {type(add_e).__name__}: {str(add_e)}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                # 尝试打印嵌入函数信息，排查问题
                logger.info(f"嵌入函数类型: {type(embedding_function)}")
                logger.info(f"嵌入函数: {embedding_function}")
        else:
            logger.warning("⚠️ 没有找到可添加的文档")
    except Exception as e:
        logger.error(f"❌ 添加文档到集合时发生未预期错误: {type(e).__name__}: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")

    # 文档添加完成
    logger.info("✅ 文档添加流程已完成")

    logger.info("✅ 向量数据库操作完成 (简化测试模式)")

    # 6. 测试搜索 (如果添加成功)
    if 'collection' in locals():
        logger.info("\n==============================")
        logger.info("🔍 测试语义搜索...")

        query = "报告写了什么？"
        try:
            # 使用collection进行搜索
            results = collection.query(
                query_texts=[query],
                n_results=2
            )
            if results and 'documents' in results and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    logger.info(f"\n🎯 匹配 {i + 1}:")
                    logger.info(f"👉 {doc[:200]}...")
            else:
                logger.info("⚠️ 未找到匹配结果")
        except Exception as e:
            logger.error(f"❌ 搜索失败！错误: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
    else:
        logger.info("⚠️ 未创建集合，跳过搜索测试")


if __name__ == "__main__":
    main()
