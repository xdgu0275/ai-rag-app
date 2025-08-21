import onnxruntime
import chromadb
import sys
from chromadb.utils import embedding_functions

# 将输出重定向到文件
with open('test_output.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    
    try:
        print("测试ONNX Runtime版本:", onnxruntime.__version__)
        
        # 禁用遥测
        chromadb.api.client.TELEMETRY = False
        
        # 初始化内存模式客户端
        client = chromadb.EphemeralClient()
        print("✅ Chroma客户端初始化成功")
        
        # 创建集合
        collection = client.create_collection(name="test_collection")
        print("✅ 集合创建成功")
        
        # 添加文档
        collection.add(
            documents=["这是一个测试文档"],
            metadatas=[{"source": "test"}],
            ids=["test_id"]
        )
        print("✅ 文档添加成功")
        
        # 验证添加结果
        count = collection.count()
        print(f"✅ 集合中的文档数量: {count}")
        
        print("测试完成！")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

# 恢复标准输出
sys.stdout = sys.__stdout__
print("测试结果已写入test_output.txt文件")