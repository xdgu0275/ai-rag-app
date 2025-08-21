import chromadb
import time
import os
from langchain_community.embeddings import DashScopeEmbeddings
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置 API Key 和禁用 Chroma 遥测
os.environ["DASHSCOPE_API_KEY"] = "sk-7940c68d86644583bc69778b8651d7e2"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# 初始化 embedding 模型
embeddings = DashScopeEmbeddings(model="text-embedding-v1")

# 初始化 Chroma 内存客户端（通过环境变量禁用遥测）
logger.info("正在初始化 Chroma 客户端...")
client = chromadb.EphemeralClient()
logger.info("✅ Chroma 客户端初始化成功")

# 创建集合
collection = client.create_collection(name="test_collection")
logger.info("✅ 集合创建成功")

# 测试数据
text = "这是一个测试文档。"
logger.info(f"测试文本: {text}")

# 生成 embedding
logger.info("开始生成 embedding...")
start_embed = time.time()
embedding = embeddings.embed_query(text)
end_embed = time.time()
logger.info(f"✅ embedding 生成成功，长度: {len(embedding)}，耗时: {end_embed - start_embed:.2f}秒")

# 尝试添加到集合
logger.info("开始执行 collection.add()...")
start_add = time.time()
try:
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=["test_id_1"]
    )
    end_add = time.time()
    logger.info(f"✅ collection.add() 执行成功，耗时: {end_add - start_add:.2f}秒")

    # 验证添加结果
    count = collection.count()
    logger.info(f"✅ 集合中的文档数量: {count}")

    # 检索文档
    result = collection.get(ids=["test_id_1"])
    logger.info(f"✅ 检索成功，文档内容: {result['documents'][0]}")
except Exception as e:
    end_add = time.time()
    logger.error(f"❌ collection.add() 执行失败，耗时: {end_add - start_add:.2f}秒")
    logger.error(f"错误类型: {type(e).__name__}")
    logger.error(f"错误信息: {str(e)}")

logger.info("测试完成")