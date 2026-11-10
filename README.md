# 🛡️ 网络安全智能问答助手

基于大语言模型与 RAG（检索增强生成）的网络安全知识问答系统，青岛理工大学网络空间安全242班《生成式人工智能》课程期末项目。

## 👥 组员
- 张宜彤（代码实现）
- 刘思语（PPT制作）
- 王茜（上台演讲）

## ✨ 功能特点
- 基于 RAG 检索增强，回答严格基于知识库内容，不会胡说八道
- 内置 10 篇网络安全核心知识点文档
- Streamlit 精美聊天界面，支持侧边栏、快捷问题、清空对话
- 回答附带参考来源，可追溯原文

## 🛠️ 技术栈
| 组件 | 选型 |
|------|------|
| 大语言模型 | 通义千问 qwen-plus |
| Embedding | text-embedding-v3 |
| 向量数据库 | Chroma |
| 应用框架 | LangChain + Streamlit |
| 运行环境 | Python 3.10+ |

## 📚 知识库内容
1. SQL注入攻击与防御
2. XSS跨站脚本攻击
3. 密码学基础（对称/非对称加密）
4. 防火墙与网络安全
5. CSRF跨站请求伪造
6. DDoS拒绝服务攻击
7. Linux系统安全
8. HTTPS与TLS安全
9. 社会工程学攻击
10. 入侵检测与防御

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install streamlit langchain langchain-openai langchain-community langchain-text-splitters chromadb pypdf python-dotenv
```

### 2. 配置API Key
在项目根目录创建 `.env` 文件：
```
DASHSCOPE_API_KEY=你的通义千问API Key
```

### 3. 建立向量索引
```bash
python ingest.py
```

### 4. 启动应用
```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

## 📁 项目结构
```
网络安全RAG问答助手/
├── app.py              # Streamlit 主界面
├── ingest.py           # 建索引脚本
├── .env                # API Key配置
├── docs/               # 知识库文档（txt格式）
└── vector_store/       # Chroma向量库（自动生成）
```

## 📝 新增文档
把新的 `.txt` 或 `.pdf` 文件放进 `docs/` 目录，重新运行 `python ingest.py` 即可。

## ⚠️ 免责声明
本项目仅用于教学研究，知识库中涉及的攻击手法内容请勿用于非法用途。
