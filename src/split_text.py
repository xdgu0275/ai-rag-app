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

# ================== 配置 ==================
# 从环境变量获取 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

dashscope.api_key = DASHSCOPE_API_KEY

# 配置参数
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")  # 通义千问 Embedding 模型名
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

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 DashScope Embedding API，包含重试机制"""
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                logger.info(f"📤 发送请求: {len(texts)} 个文本 (重试: {retry_count}/{MAX_RETRIES-1})")
                response = TextEmbedding.call(
                    model=self.model,
                    input=texts
                )
                logger.info(f"📥 收到响应: 状态码 {response.status_code}")
                if response.status_code == 200:
                    # 检查响应结构
                    if not hasattr(response, 'output') or 'embeddings' not in response.output:
                        raise Exception("API 响应格式不正确，缺少 embeddings 字段")
                    embeddings_data = response.output['embeddings']
                    # 提取每个 embedding 向量
                    results = [item['embedding'] for item in embeddings_data]
                    logger.info(f"✅ 成功获取 {len(results)} 个嵌入向量")
                    return results
                elif response.status_code in [429, 500, 502, 503, 504]:
                    # 限流或服务器错误，重试
                    retry_count += 1
                    wait_time = REQUEST_INTERVAL * (2 ** retry_count)  # 指数退避
                    logger.warning(f"⚠️ API 错误: {response.code} - {response.message}, 等待 {wait_time} 秒后重试")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API 错误: {response.code} - {response.message}")
            except Exception as e:
                retry_count += 1
                logger.error(f"❌ Embedding 调用失败: {str(e)}, 重试 {retry_count}/{MAX_RETRIES-1}")
                if retry_count < MAX_RETRIES:
                    wait_time = REQUEST_INTERVAL * (2 ** retry_count)
                    time.sleep(wait_time)
                else:
                    raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入（自动分批）"""
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            logger.info(f"🌐 调用通义千问 Embedding (批次: {batch_num}/{(len(texts)-1)//BATCH_SIZE + 1}, 数量: {len(batch)})")
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
            continue
        content = doc.get("content", "").strip()
        if not content or len(content) < 10:
            continue

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            clean_chunk = chunk.replace("\r", "").strip()
            if len(clean_chunk) < 5:
                continue
            all_chunks.append(Document(
                page_content=clean_chunk,
                metadata={"source": doc["filename"], "chunk_id": i}
            ))
        logger.info(f"✂️  {doc['filename']} → {len(chunks)} 个块")

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
        # 配置 Chroma 客户端，禁用 telemetry
        settings = Settings(anonymized_telemetry=False)
        logger.info("🔧 已通过 Settings 禁用 Chroma telemetry")

        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH,
            collection_name="qwen_rag",
            client_settings=settings
        )
        logger.info(f"✅ 向量库已创建并持久化: {VECTOR_DB_PATH}")
        # 显式持久化
        vectorstore.persist()
        logger.info(f"✅ 向量库已显式持久化")
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