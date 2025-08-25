import os
from chromadb import PersistentClient

# 设置向量库路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = "vectorstore_qwen"
abs_vector_path = os.path.join(BASE_DIR, VECTOR_DB_DIR)
COLLECTION_NAME = "qwen_rag"

print(f"🔍 向量库绝对路径: {abs_vector_path}")

# 使用PersistentClient连接向量库
try:
    client = PersistentClient(path=abs_vector_path)
    print(f"✅ 成功初始化PersistentClient")

    # 列出所有集合
    collections = client.list_collections()
    print(f"✅ 找到 {len(collections)} 个集合")
    for coll in collections:
        print(f"   - 集合名称: {coll.name}, 文档数量: {coll.count()}")

    # 尝试获取指定集合
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ 成功访问集合: {COLLECTION_NAME}")
        print(f"📊 集合文档数量: {collection.count()}")

        # 如果有文档，尝试获取一些文档ID
        if collection.count() > 0:
            print("🔍 尝试获取文档ID...")
            results = collection.get(limit=5)
            print(f"✅ 获取到 {len(results['ids'])} 个文档ID: {results['ids'][:5]}")
        else:
            print("⚠️ 集合中没有文档")

    except Exception as e:
        print(f"❌ 无法访问集合 {COLLECTION_NAME}: {e}")

except Exception as e:
    print(f"❌ 连接向量库失败: {e}")