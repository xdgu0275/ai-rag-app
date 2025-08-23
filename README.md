# AI-RAG应用

基于检索增强生成（RAG）技术的人工智能应用，用于文档问答和知识检索。

## 功能特点
- 文档加载与解析（支持PDF、DOCX等格式）
- 文本分割与向量化存储
- 基于向量相似度的检索
- 结合大语言模型的智能问答
- 简单易用的Web界面

## 技术栈
- **核心框架**：LangChain
- **向量存储**：ChromaDB
- **大语言模型**：通过DashScope API接入
- **文档处理**：PyPDF、docx2txt
- **Web框架**：FastAPI、Streamlit
- **数据验证**：Pydantic
- **数据库ORM**：SQLAlchemy
- **配置管理**：python-dotenv

## 安装步骤

### 1. 克隆仓库
```bash
# 假设这是一个Git仓库
# git clone <仓库URL>
# cd ai-rag-app
```

### 2. 创建并激活虚拟环境
```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/Mac
source venv/bin/activate
# Windows
# venv\Scripts\activate
```

### 3. 安装依赖
```bash
# 使用清华源加速安装
pip install -r requirements.txt
```

### 4. 配置环境变量
创建`.env`文件，添加以下配置：
```
# DashScope API密钥
DASHSCOPE_API_KEY=your_api_key

# 其他配置
# 向量存储路径
VECTOR_STORE_PATH=./vectorstore_qwen

# 日志级别
LOG_LEVEL=INFO
```

## 使用方法

### 1. 加载文档
```bash
python src/load_documents.py --docs_path ./docs
```

### 2. 启动API服务
```bash
uvicorn src.api:app --reload
```

### 3. 启动Web界面
```bash
streamlit run src/web_app.py
```

## 项目结构
```
ai-rag-app/
├── .env                 # 环境变量配置
├── .gitignore           # Git忽略文件
├── README.md            # 项目说明
├── requirements.txt     # 依赖列表
├── docs/                # 文档目录
│   ├── notes.txt        # 笔记
│   ├── report.docx      # 报告文档
│   └── sample.pdf       # 示例PDF
├── src/                 # 源代码
│   ├── api.py           # API接口
│   ├── docs.json        # 文档元数据
│   ├── load_documents.py # 文档加载脚本
│   ├── split_text.py    # 文本分割脚本
│   ├── vector_store.py  # 向量存储操作
│   └── web_app.py       # Web应用
└── vectorstore_qwen/    # 向量存储目录
    ├── chroma.sqlite3   # ChromaDB数据库
    └── ...              # 向量数据
```

## 配置说明
所有配置项都可以在`.env`文件中设置：
- `DASHSCOPE_API_KEY`: 阿里云DashScope服务的API密钥
- `VECTOR_STORE_PATH`: 向量存储的路径
- `LOG_LEVEL`: 日志级别（DEBUG, INFO, WARNING, ERROR）

## 贡献指南
1. Fork本仓库
2. 创建特性分支（`git checkout -b feature/xxx`）
3. 提交修改（`git commit -am 'Add some feature'`）
4. 推送到分支（`git push origin feature/xxx`）
5. 创建Pull Request

## 许可证
本项目采用MIT许可证，详情请见LICENSE文件。
