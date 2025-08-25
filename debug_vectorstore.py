import chromadb
import os

# 获取向量库路径
vector_db_path = os.path.abspath("vectorstore_qwen")
print(f"🔍 向量库绝对路径: {vector_db_path}")

# 初始化chromadb客户端
client = chromadb.PersistentClient(path=vector_db_path)
print(f"✅ 初始化chromadb客户端成功")

# 列出所有集合
collections = client.list_collections()
print(f"✅ 找到 {len(collections)} 个集合")
for coll in collections:
    print(f"   - 集合名称: {coll.name}, 文档数量: {coll.count()}")
    
# 尝试访问qwen_rag集合
try:
    collection = client.get_collection("qwen_rag")
    print(f"✅ 成功访问集合: {collection.name}")
    print(f"📊 集合统计信息: 文档数量={collection.count()}")
    
    # 如果有文档，获取一些文档
    if collection.count() > 0:
        results = collection.get(limit=5)
        print(f"✅ 获取到 {len(results['ids'])} 个文档ID: {results['ids'][:5]}")
    else:
        print("⚠️ 集合中没有文档")

except Exception as e:
    print(f"❌ 无法访问集合: {e}")