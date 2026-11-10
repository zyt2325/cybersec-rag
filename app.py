"""
网络安全智能问答助手 - Streamlit 界面
运行: streamlit run app.py
"""
import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 本地运行用.env，云端部署用st.secrets
try:
    DASHSCOPE_API_KEY = st.secrets["DASHSCOPE_API_KEY"]
except:
    DASHSCOPE_API_KEY = "sk-ws-H.PYMLRMX.Fu2x.MEQCIF4oG83ILR-C5tREEjKrhueW8Z17wwtMRTc3c1dG4w4tAiAlHCHMXJMzv7bFMdbObAQpb5yl4Qz08ut_7uLsO-cA8g"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "vector_store")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

st.set_page_config(
    page_title="网络安全智能问答助手",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); }
    .main .block-container { padding-top: 2rem; padding-bottom: 6rem; }
    h1 { color: #1e3c78 !important; }
    .project-card {
        background: linear-gradient(135deg, #1e3c78 0%, #5385bd 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .project-card h3 { color: white !important; margin: 0 0 0.5rem 0; }
    .project-card p { margin: 0; opacity: 0.9; }
    .stChatMessage { border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #5385bd;
        color: #1e3c78;
        background: white;
        font-size: 0.85rem;
    }
    .stButton > button:hover {
        background: #5385bd;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v3",
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )
    vs = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        streaming=True,
    )
    return vs, llm


def ask_stream(question, vs, llm):
    t0 = time.time()
    docs = vs.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严谨的网络安全知识问答助手。请严格根据下面提供的参考资料回答用户问题。

要求：
1. 只根据参考资料回答，不要编造资料里没有的内容。
2. 如果资料里没有相关信息，直接说"根据现有知识库暂未找到相关内容"。
3. 回答要准确、简洁、有条理，用markdown格式排版。
4. 如果问题涉及攻击手法、漏洞利用，在回答末尾加上"⚠️ 以上内容仅供网络安全教学研究，请勿用于非法用途"。

参考资料：
{context}"""),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    elapsed = time.time() - t0
    return chain.stream({"context": context, "question": question}), docs, elapsed


if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
    st.warning("⚠️ 向量库尚未建立，请先运行 python ingest.py 建库")
    st.stop()

vs, llm = load_resources()

# 侧边栏
with st.sidebar:
    st.title("🛡️ 系统信息")
    st.markdown("**技术栈**")
    st.markdown("- 大模型：通义千问 qwen-plus")
    st.markdown("- 向量库：Chroma")
    st.markdown("- Embedding：text-embedding-v3")
    st.markdown("- 框架：LangChain + Streamlit")
    st.divider()
    st.markdown("**知识库分类快速提问**")
    categories = {
        "Web安全": "SQL注入的原理和防御方法是什么？",
        "密码学": "对称加密和非对称加密有什么区别？",
        "网络安全": "防火墙的工作原理是什么？",
        "系统安全": "Linux系统安全加固有哪些要点？",
        "云安全": "云安全共享责任模型是什么？",
        "新兴技术": "什么是零信任安全架构？",
        "攻防实战": "渗透测试的流程是什么？",
        "合规法律": "等保2.0是什么？",
    }
    def on_cat_change():
        if st.session_state.cat_select != "选择分类...":
            st.session_state.pending_query = categories[st.session_state.cat_select]
            st.session_state.cat_select = "选择分类..."
    st.selectbox("选择分类快速提问：", ["选择分类..."] + list(categories.keys()), key="cat_select", on_change=on_cat_change)
    st.divider()
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是网络安全知识助手，内置海量网络安全核心知识点文档，可以问我任何相关问题。"}
        ]
        st.rerun()
    if st.button("📥 导出对话记录"):
        chat_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="下载对话记录",
            data=chat_text,
            file_name="对话记录.txt",
            mime="text/plain",
        )
    st.divider()
    with st.expander("ℹ️ 关于项目"):
        st.markdown("""
**网络安全智能问答助手**

基于RAG技术构建的网络安全知识问答系统。

**组员**：
- 张宜彤（代码实现）
- 刘思语（PPT制作）
- 王茜（上台演讲）

《生成式人工智能》课程项目
网安241/242/243班
        """)

# 主界面
st.title("🛡️ 网络安全智能问答助手")
st.caption("基于大语言模型与 RAG 检索增强生成 · 《生成式人工智能》课程项目")

# 项目介绍卡片
st.markdown("""
<div class="project-card">
    <h3>🎯 项目简介</h3>
    <p>本系统基于检索增强生成（RAG）技术，将大语言模型与专业网络安全知识库结合，实现准确、可溯源的网络安全知识问答。内置200+网络安全核心知识点，覆盖Web安全、系统安全、密码学、云安全、攻防实战等全方向。</p>
</div>
""", unsafe_allow_html=True)

# RAG流程说明
with st.expander("🔍 RAG工作原理（点击展开）", expanded=False):
    st.markdown("""
    **用户提问** → ① 向量检索知识库 → ② 拼接相关文档作为上下文 → ③ 大模型基于上下文生成回答 → 返回答案
    """)
    st.info("💡 RAG的优势：回答基于真实知识库内容，不会胡说八道，并且可以追溯来源。")

# 快捷问题
st.markdown("**💡 试试这些问题：**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("SQL注入怎么防御？"):
        st.session_state.pending_query = "SQL注入的防御方法有哪些？"
with col2:
    if st.button("XSS有哪几种类型？"):
        st.session_state.pending_query = "XSS跨站脚本攻击有哪几种类型？"
with col3:
    if st.button("什么是零信任？"):
        st.session_state.pending_query = "什么是零信任安全架构？"

st.divider()

# 初始化消息
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是网络安全知识助手，内置海量网络安全核心知识点文档，可以问我任何相关问题。"}
    ]

# 渲染所有历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 官方标准输入，先获取手动输入
prompt = st.chat_input("请输入你的网络安全问题...")

# 如果有按钮点的pending_query，覆盖掉
if "pending_query" in st.session_state:
    prompt = st.session_state.pending_query
    del st.session_state.pending_query

# 统一走同一个处理路径，不管是手动输入还是按钮点的
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并思考..."):
            stream_gen, sources, elapsed = ask_stream(prompt, vs, llm)
            response = st.write_stream(stream_gen)
        st.caption(f"⏱️ 检索+生成耗时 {elapsed:.1f} 秒 · 检索了 {len(sources)} 条知识库片段")
        if sources:
            with st.expander("📚 参考来源（来自内置知识库文档）"):
                for i, doc in enumerate(sources, 1):
                    src = doc.metadata.get("source", "未知来源")
                    st.write(f"[{i}] {os.path.basename(src)}")
                    st.caption(doc.page_content[:150] + "...")
    st.session_state.messages.append({"role": "assistant", "content": response})
