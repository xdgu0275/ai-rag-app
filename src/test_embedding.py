from langchain_community.embeddings import DashScopeEmbeddings

# 设置模型
embeddings = DashScopeEmbeddings(model="text-embedding-v1")

# 测试文本
text = "Hello, world!"

# 生成向量
try:
    result = embeddings.embed_query(text)
    print("✅ 成功！向量长度:", len(result))
    print("👉 前10个数值:", result[:10])
except Exception as e:
    print("❌ 失败！错误:", e)