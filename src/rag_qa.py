# src/rag_qa.py - 完整的 RAG 问答系统（检索 + 生成 + 引用）
import os
import logging
import time
import requests
from typing import List, Optional, Dict, Any

from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.schema import Document
import dashscope
import dotenv

# 加载环境变量
dotenv.load_dotenv()

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ================== 通义千问文本生成类 ==================
class QwenTextGeneration:
    """封装通义千问文本生成API调用"""
    def __init__(self, model: str = "qwen3-30b-a3b-thinking-2507"):
        self.model = model
        self.MAX_RETRIES = 3
        self.REQUEST_INTERVAL = 12
        
        # 从环境变量获取API Key
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
        
    def generate(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """生成文本响应
        Args:
            prompt: 用户输入的提示
            history: 对话历史
        Returns:
            生成的文本响应
        """
        retry_count = 0
        while retry_count < self.MAX_RETRIES:
            try:
                logger.info(f"📤 发送文本生成请求 (模型: {self.model}, 重试: {retry_count}/{self.MAX_RETRIES-1})")
                
                url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # 构建消息列表
                messages = []
                if history:
                    messages.extend(history)
                messages.append({"role": "user", "content": prompt})
                
                data = {
                    "model": self.model,
                    "input": {"messages": messages}
                }
                
                response = requests.post(url, json=data, headers=headers)
                logger.info(f"📥 收到文本生成响应: 状态码 {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"📝 API响应内容: {result}")  # 打印完整响应，用于调试
                    
                    # 尝试不同的响应解析方式
                    if "output" in result:
                        # 检查是否直接有text字段
                        if "text" in result["output"]:
                            return result["output"]["text"]
                        # 检查是否有choices字段
                        elif "choices" in result["output"] and len(result["output"]["choices"]) > 0:
                            if "message" in result["output"]["choices"][0] and "content" in result["output"]["choices"][0]["message"]:
                                return result["output"]["choices"][0]["message"]["content"]
                            elif "text" in result["output"]["choices"][0]:
                                return result["output"]["choices"][0]["text"]
                    elif "result" in result:
                        return result["result"]
                    
                    # 如果所有尝试都失败，抛出异常
                    raise Exception(f"API响应格式不正确，预期结构未找到。响应: {result}")
                elif response.status_code in [429, 500, 502, 503, 504]:
                    # 限流或服务器错误，重试
                    retry_count += 1
                    wait_time = self.REQUEST_INTERVAL * (2 ** retry_count)  # 指数退避
                    logger.warning(f"⚠️ API错误: {response.status_code}, 等待 {wait_time} 秒后重试")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API错误: {response.status_code} - {response.text}")
            except Exception as e:
                retry_count += 1
                logger.error(f"❌ 文本生成调用失败: {str(e)}, 重试 {retry_count}/{self.MAX_RETRIES-1}")
                if retry_count < self.MAX_RETRIES:
                    wait_time = self.REQUEST_INTERVAL * (2 ** retry_count)
                    time.sleep(wait_time)
                else:
                    raise


# ================== 通义千问 Embedding 类 ==================
class QwenEmbeddings:
    """封装通义千问 Embedding API 调用"""
    def __init__(self, model: str = "text-embedding-v4"):
        self.model = model
        self.MAX_RETRIES = 3
        self.REQUEST_INTERVAL = 12
        self.BATCH_SIZE = 5

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 DashScope Embedding API，仅使用HTTP调用方式"""
        retry_count = 0
        while retry_count < self.MAX_RETRIES:
            try:
                logger.info(f"📤 发送HTTP请求: {len(texts)} 个文本 (重试: {retry_count}/{self.MAX_RETRIES-1})")
                
                # 使用兼容模式API URL
                url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {dashscope.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
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
                        retry_count += 1
                        wait_time = self.REQUEST_INTERVAL * (2 ** retry_count)  # 指数退避
                        logger.warning(f"⚠️ HTTP API 错误: 状态码 {response.status_code}, 响应: {response.text}, 等待 {wait_time} 秒后重试")
                        time.sleep(wait_time)
                        if retry_count >= self.MAX_RETRIES:
                            raise Exception(f"HTTP API 重试次数达到上限 {self.MAX_RETRIES} 次")
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
            if i + self.BATCH_SIZE < len(texts):  # 不是最后一批才需要等待
                time.sleep(self.REQUEST_INTERVAL)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入"""
        return self._call_api([text])[0]

# ================== 配置 ==================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

dashscope.api_key = DASHSCOPE_API_KEY

# 向量库路径（与split_text.py保持一致）
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore_qwen")
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# 大模型配置
LLM_MODEL = "qwen-max"  # 可选: qwen-plus, qwen-turbo
EMBEDDING_MODEL = "text-embedding-v4"  # 与split_text.py保持一致

# ================== 加载向量库 ==================
def load_vectorstore():
    """加载之前创建的向量库"""
    try:
        # 打印向量库路径的绝对路径，用于调试
        abs_vector_path = os.path.abspath(VECTOR_DB_PATH)
        logger.info(f"🔍 尝试加载向量库，绝对路径: {abs_vector_path}")
        
        # 禁用 Chroma telemetry
        os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
        logger.info("🔧 已禁用 Chroma telemetry")
        
        # 与split_text.py保持一致，使用相同的嵌入模型
        embeddings = QwenEmbeddings()
        logger.info(f"✅ 初始化嵌入模型: {embeddings.model}")

        # 添加向量库配置调试信息
        logger.info(f"🔍 向量库配置: 绝对路径={abs_vector_path}, 集合名称=qwen_rag, 嵌入模型={embeddings.model}")
        
        # 检查目录是否存在
        if not os.path.exists(abs_vector_path):
            logger.error(f"❌ 向量库目录不存在: {abs_vector_path}")
            return None
        
        # 检查目录是否非空
        if len(os.listdir(abs_vector_path)) == 0:
            logger.error(f"❌ 向量库目录为空: {abs_vector_path}")
            return None
        
        # 打印目录内容，用于调试
        logger.info(f"📂 向量库目录内容: {os.listdir(abs_vector_path)}")
        
        # 使用PersistentClient直接连接向量库
        from chromadb import PersistentClient
        client = PersistentClient(path=abs_vector_path)
        logger.info(f"✅ 成功初始化PersistentClient")
        
        # 获取集合 - 不传入embedding_function，使用已存储的嵌入
        try:
            collection = client.get_collection(
                name="qwen_rag"
            )
            logger.info(f"✅ 成功访问集合: qwen_rag")
            
            # 检查文档数量
            doc_count = collection.count()
            logger.info(f"📊 集合文档数量: {doc_count}")
            
            # 创建Chroma向量库实例
            db = Chroma(
                client=client,
                collection_name="qwen_rag",
                embedding_function=embeddings
            )
            logger.info(f"✅ 向量库加载完成")
            
            # 打印Chroma客户端配置，用于调试
            logger.info(f"🔍 向量库客户端类型: {type(db._client)}")
            
            # 检查向量库连接是否正常
            try:
                # 执行一个简单的查询
                test_query = "测试查询"
                test_embedding = embeddings.embed_query(test_query)
                logger.info(f"✅ 生成测试查询嵌入成功，向量长度: {len(test_embedding)}")
            except Exception as e:
                logger.error(f"❌ 生成测试查询嵌入失败: {e}")
            
            return db
        except Exception as e:
            logger.error(f"❌ 无法访问集合 qwen_rag: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"❌ 加载向量库失败: {e}")
        return None

# ================== RAG 问答函数 ==================
def ask_rag_question(question: str, k: int = 2):
    """
    执行 RAG 问答
    :param question: 用户问题
    :param k: 返回 top-k 个相关片段
    """
    db = load_vectorstore()
    if not db:
        return

    logger.info(f"📝 用户提问: {question}")
    logger.info(f"🔍 正在检索最相关的 {k} 个文档片段...")

    # 1. 检索
    try:
        # 添加检索参数调试信息
        logger.info(f"🔍 检索参数: 问题='{question}', k={k}")
        
        # 生成查询嵌入
        embeddings = QwenEmbeddings(model=EMBEDDING_MODEL)
        query_embedding = embeddings.embed_query(question)
        logger.info(f"✅ 生成查询嵌入成功，向量长度: {len(query_embedding)}")
        
        # 执行检索
        docs = db.similarity_search_by_vector(query_embedding, k=k)
        logger.info(f"✅ 检索完成，找到 {len(docs)} 个相关文档")
        
        if not docs:
            print(f"⚠️ 未找到相关文档。")
            print(f"🔍 建议检查向量库内容: 运行 python src/split_text.py 确认向量库是否正确构建")
            return
        
        # 2. 生成回答（这里简化处理，仅展示检索结果）
        print(f"\n🔍 找到 {len(docs)} 个相关结果：")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. 来源: {doc.metadata.get('source', '未知来源')}")
            print(f"   内容: {doc.page_content[:100]}...\n")
            
    except Exception as e:
        logger.error(f"❌ 问答过程出错: {e}")
        print(f"⚠️ 问答过程出错: {str(e)}")

# ================== 调试函数 ==================
def debug_vectorstore():
    """直接使用chromadb客户端调试向量库"""
    try:
        import chromadb
        logger.info("✅ 导入chromadb成功")
        
        # 获取向量库绝对路径
        abs_vector_path = os.path.abspath(VECTOR_DB_PATH)
        logger.info(f"🔍 向量库绝对路径: {abs_vector_path}")
        
        # 直接初始化chromadb客户端
        client = chromadb.PersistentClient(path=abs_vector_path)
        logger.info(f"✅ 初始化chromadb客户端成功")
        
        # 列出所有集合
        collections = client.list_collections()
        logger.info(f"✅ 找到 {len(collections)} 个集合")
        for coll in collections:
            logger.info(f"   - 集合名称: {coll.name}, 文档数量: {coll.count()}")
            
        # 尝试直接访问qwen_rag集合
        try:
            collection = client.get_collection("qwen_rag")
            logger.info(f"✅ 成功访问集合: {collection.name}")
            logger.info(f"📊 集合统计信息: 文档数量={collection.count()}")
            
            # 如果有文档，尝试获取一些文档
            if collection.count() > 0:
                results = collection.get(limit=5)
                logger.info(f"✅ 获取到 {len(results['ids'])} 个文档ID: {results['ids'][:5]}")
            else:
                logger.warning("⚠️ 集合中没有文档")
        except Exception as e:
            logger.error(f"❌ 无法访问集合: {e}")
    except Exception as e:
        logger.error(f"❌ 调试向量库失败: {e}")

# ================== 测试函数 ==================
if __name__ == "__main__":
    logger.info("🚀 RAG 问答系统启动")
    
    # 先运行调试函数
    debug_vectorstore()
    
    # 测试问题
    test_questions = [
        "报告写了什么？",
        "sample.pdf 是什么内容？",
        "LangChain 可以读取什么文件？"
    ]
    
    for question in test_questions:
        print("--------------------------------------------------")
        ask_rag_question(question)
        print("--------------------------------------------------")
        time.sleep(2)  # 避免请求过快
    
    logger.info("🏁 RAG 问答系统测试完成")