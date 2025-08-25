# qwen_embedding_rag.py - 调用通义千问 Embedding API + Chroma 存储
import json
import os
import time
import logging
import dotenv
from typing import List, Optional, Dict, Any

# 加载环境变量
dotenv.load_dotenv()

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 向量库
from langchain_chroma import Chroma
from chromadb.config import Settings

# 禁用 Chroma telemetry
logger.info("🔧 已设置禁用 Chroma telemetry")
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 通义千问 SDK
import dashscope
from dashscope import TextEmbedding
import requests  # 新增，用于直接HTTP请求

# ================== 配置 ==================
# 从环境变量获取 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

dashscope.api_key = DASHSCOPE_API_KEY

# 配置参数
# 使用测试成功的模型版本
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")  # 通义千问模型名
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore_qwen")
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# 分块配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))  # 通义千问免费版限流：每分钟 5 次请求
REQUEST_INTERVAL = int(os.getenv("REQUEST_INTERVAL", "12"))  # 请求间隔时间(秒)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # 最大重试次数

current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(current_dir, "docs.json")

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ================== 通义千问 Embedding 类 ==================
class QwenEmbeddings:
    """封装通义千问 Embedding API 调用"""
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.MAX_RETRIES = MAX_RETRIES
        self.REQUEST_INTERVAL = REQUEST_INTERVAL
        self.BATCH_SIZE = BATCH_SIZE

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 DashScope Embedding API，仅使用HTTP调用方式"""
        retry_count = 0
        while retry_count < self.MAX_RETRIES:
            try:
                logger.info(f"📤 发送HTTP请求: {len(texts)} 个文本 (重试: {retry_count}/{self.MAX_RETRIES-1})")
                
                # 根据阿里云官方文档，所有API调用本质都是HTTP请求
                url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {dashscope.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "text-embedding-v4",
                    "input": texts,
                    "dimensions": 1024,
                    "encoding_format": "float"
                }
                response = requests.post(url, json=data, headers=headers)
                logger.info(f"📥 收到 HTTP 响应: 状态码 {response.status_code}")
                if response.status_code == 200:
                        result = response.json()
                        # 检查响应结构
                        if "data" not in result:
                            raise Exception("HTTP API 响应格式不正确，缺少 data 字段")
                        embeddings_data = [item["embedding"] for item in result["data"]]
                        results = embeddings_data
                        logger.info(f"✅ 成功获取 {len(results)} 个嵌入向量")
                        return results
                elif response.status_code in [429, 500, 502, 503, 504]:
                        # 限流或服务器错误，重试
                        raise Exception(f"HTTP API 错误: 状态码 {response.status_code}, 响应: {response.text}")
                else:
                        raise Exception(f"HTTP API 错误: 状态码 {response.status_code}, 响应: {response.text}")
            except Exception as e:
                retry_count += 1
                logger.error(f"❌ Embedding 调用失败: {str(e)}, 重试 {retry_count}/{self.MAX_RETRIES-1}")
                if retry_count < self.MAX_RETRIES:
                    wait_time = self.REQUEST_INTERVAL * (2 ** retry_count)
                    time.sleep(wait_time)
                else:
                    raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入（自动分批）"""
        all_embeddings = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i+self.BATCH_SIZE]
            batch_num = i // self.BATCH_SIZE + 1
            logger.info(f"🌐 调用通义千问 Embedding (批次: {batch_num}/{(len(texts)-1)//self.BATCH_SIZE + 1}, 数量: {len(batch)})")
            batch_embeds = self._call_api(batch)
            all_embeddings.extend(batch_embeds)
            logger.info(f"✅ 批次 {batch_num} 处理完成")
            # 避免限流
            if i + BATCH_SIZE < len(texts):  # 不是最后一批才需要等待
                time.sleep(REQUEST_INTERVAL)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入"""
        return self._call_api([text])[0]


# ================== 主程序 ==================
def main():
    logger.info("🚀 开始使用通义千问 Embedding 进行向量化")

    # 1. 读取 docs.json
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            docs = json.load(f)
        logger.info(f"✅ 加载 {len(docs)} 个文档")
    except FileNotFoundError:
        logger.error("❌ 找不到 ../docs.json")
        return
    except Exception as e:
        logger.error(f"❌ 读取失败: {e}")
        return

    # 2. 分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    all_chunks = []

    for doc in docs:
        if doc.get("status") != "success":
            logger.warning(f"⚠️ 跳过状态非success的文档: {doc.get('filename', '未知文件名')}")
            continue
        content = doc.get("content", "").strip()
        if not content:
            logger.warning(f"⚠️ 跳过空内容文档: {doc.get('filename', '未知文件名')}")
            continue
        if len(content) < 10:
            logger.warning(f"⚠️ 跳过内容过短文档: {doc.get('filename', '未知文件名')} (长度: {len(content)})")
            continue

        chunks = splitter.split_text(content)
        logger.info(f"✂️  {doc.get('filename', '未知文件名')} → 生成 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks):
            clean_chunk = chunk.replace("\r", "").strip()
            if len(clean_chunk) < 5:
                logger.warning(f"⚠️ 跳过过短块 ({len(clean_chunk)} 字符): {clean_chunk[:20]}...")
                continue
            doc_obj = Document(
                page_content=clean_chunk,
                metadata={"source": doc["filename"], "chunk_id": i}
            )
            all_chunks.append(doc_obj)
            logger.info(f"✅ 添加块 {i+1}/{len(chunks)}: {clean_chunk[:30]}...")

    if not all_chunks:
        logger.error("❌ 没有可处理的文本块")
        return
    logger.info(f"✅ 共生成 {len(all_chunks)} 个文本块")

    # 3. 初始化通义千问 Embedding
    try:
        embeddings = QwenEmbeddings()
        # 测试调用
        test_embed = embeddings.embed_query("测试")
        logger.info(f"✅ 通义千问 Embedding 初始化成功，向量维度: {len(test_embed)}")
    except Exception as e:
        logger.error("❌ 初始化失败，请检查 API Key 或网络")
        return

    # 4. 创建 Chroma 向量库
    try:
        logger.info(f"🔄 开始创建向量库，共 {len(all_chunks)} 个文档块")
        # 定义向量库绝对路径
        abs_vector_path = os.path.abspath(VECTOR_DB_PATH)
        logger.info(f"🔍 向量库绝对路径: {abs_vector_path}")

        # 直接使用chromadb客户端创建向量库
        from chromadb import PersistentClient
        client = PersistentClient(path=abs_vector_path)
        logger.info(f"✅ 成功初始化PersistentClient")

        # 删除旧集合（如果存在）
        try:
            client.delete_collection(name="qwen_rag")
            logger.info("🔄 删除旧集合: qwen_rag")
        except Exception as e:
            logger.info(f"ℹ️ 集合不存在或无法删除: {e}")

        # 创建符合Chroma接口的嵌入函数适配器
        class ChromaEmbeddingAdapter:
            def __init__(self, embedding_model):
                self.embedding_model = embedding_model

            def __call__(self, input):
                # Chroma期望input是一个文档列表
                return self.embedding_model.embed_documents(input)

        # 使用适配器包装嵌入模型
        embedding_adapter = ChromaEmbeddingAdapter(embeddings)

        # 创建新集合
        collection = client.create_collection(
            name="qwen_rag",
            embedding_function=embedding_adapter
        )
        logger.info(f"✅ 创建集合: qwen_rag")

        # 手动添加文档
        if all_chunks:
            logger.info(f"🔄 准备添加 {len(all_chunks)} 个文档到向量库...")
            document_ids = [str(i) for i in range(len(all_chunks))]
            documents = [chunk.page_content for chunk in all_chunks]
            metadatas = [chunk.metadata for chunk in all_chunks]

            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=document_ids
            )
            logger.info(f"✅ 成功添加 {len(all_chunks)} 个文档到向量库")
            logger.info(f"📊 向量库统计信息: 文档数量={collection.count()}")

            # 尝试获取文档ID
            results = collection.get(limit=5)
            logger.info(f"✅ 获取到 {len(results['ids'])} 个文档ID: {results['ids'][:5]}")
        else:
            logger.warning("⚠️ 没有文档可添加到向量库")

        # 创建LangChain的Chroma实例（如果需要）
        vectorstore = Chroma(
            client=client,
            collection_name="qwen_rag",
            embedding_function=embeddings
        )
    except Exception as e:
        logger.error(f"❌ 向量库创建失败: {str(e)}")
        return

    # 5. 搜索测试
    try:
        logger.info("🔍 开始搜索测试")
        query = "报告写了什么？"
        logger.info(f"📝 搜索查询: {query}")
        results = vectorstore.similarity_search(query, k=2)
        logger.info(f"✅ 找到 {len(results)} 个相关结果")
        logger.info(f"\n🔍 搜索 '{query}' 的结果：")
        for i, r in enumerate(results):
            logger.info(f"{i+1}. 来源: {r.metadata['source']}")
            logger.info(f"   内容: {r.page_content[:100]}...\n")
        logger.info("✅ 搜索测试完成")
    except Exception as e:
        logger.error(f"❌ 搜索测试失败: {str(e)}", exc_info=True)

    logger.info("🏁 程序执行完成")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {str(e)}", exc_info=True)
    logger.info("🏁 程序执行完成")