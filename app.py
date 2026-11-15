"""
网络安全智能问答助手 - Streamlit 界面
运行: streamlit run app.py
"""
import os
import json
import time
import uuid
import html
from datetime import datetime

import streamlit as st
import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 本地运行用 .env，云端部署用 st.secrets
try:
    DASHSCOPE_API_KEY = st.secrets["DASHSCOPE_API_KEY"]
except Exception:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-ws-H.PYMLRMX.Fu2x.MEQCIF4oG83ILR-C5tREEjKrhueW8Z17wwtMRTc3c1dG4w4tAiAlHCHMXJMzv7bFMdbObAQpb5yl4Qz08ut_7uLsO-cA8g")

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INDEX_FILE = os.path.join(BASE_DIR, "docs_index.json")
HIST_DIR = os.path.join(BASE_DIR, "data", "history")

SIM_THRESHOLD = 1.25     # 相似度距离超过该值认为知识库没有覆盖（text-embedding-v2 实测分界）
MAX_TURNS = 3             # 带进提示词的历史轮数
K_RETRIEVE = 5            # 检索候选数量
EMBED_MODEL = "text-embedding-v2"

WELCOME = "你好！我是网络安全知识助手，内置覆盖十一个方向的专业网络安全知识文档，可以问我任何网络安全相关问题。支持连续追问，回答都会标注参考来源。"

SYS_PROMPT = """你是"网络安全智能问答助手"，专门回答网络安全相关的问题。请严格依据下面提供的参考资料作答。

回答要求：
1. 只使用参考资料中的内容回答，不要引入资料之外的知识，也不要编造。
2. 如果参考资料不足以回答这个问题，直接回复"根据现有知识库暂未找到相关内容"，并说明知识库主要覆盖哪些方向，不要硬编答案。
3. 回答准确、简洁、有条理，使用 Markdown 排版，适当使用小标题与编号。
4. 结合上文对话理解用户省略的主语，但不要重复之前已经讲过的内容。
5. 涉及攻击手法或漏洞利用的问题，只讲原理与防御思路，并在回答末尾加上一句"以上内容仅用于网络安全教学与研究，请勿用于非法用途"。

参考资料：
{context}"""

SUGGEST_PROMPT = """根据下面这一轮网络安全问答，给出 3 个用户可能接着追问的问题。

要求：
1. 每行只写一个问题，不超过 20 个字。
2. 直接写问题本身，不要编号、不要引号、不要任何解释。
3. 问题要围绕网络安全知识，可以从原理、区别、防御、场景等角度展开。

用户问题：{question}
回答要点：{answer}"""

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
    .project-card p { margin: 0; opacity: 0.92; line-height: 1.7; }
    .stChatMessage { border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #5385bd;
        color: #1e3c78;
        background: white;
        font-size: 0.85rem;
    }
    .stButton > button:hover { background: #5385bd; color: white; }
    .doc-view {
        white-space: pre-wrap;
        font-size: 0.82rem;
        line-height: 1.75;
        color: #2c3444;
        background: #ffffff;
        border: 1px solid #dbe3ec;
        border-radius: 10px;
        padding: 0.8rem;
        max-height: 22rem;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )
    # 向量库按 kb_0 / kb_1 ... 分集合存放，这里把每个集合都打开
    client = chromadb.PersistentClient(path=VECTOR_DIR)
    names = [c.name for c in client.list_collections() if c.name.startswith("kb_")]
    names.sort(key=lambda n: int(n.split("_")[-1]))
    stores = [
        Chroma(client=client, collection_name=n, embedding_function=embeddings)
        for n in names
    ]
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        streaming=True,
    )
    return stores, llm


@st.cache_resource
def load_index():
    """读取分类索引，缺失时按文件名兜底生成"""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    cats = ["Web安全", "系统安全", "密码学", "网络安全", "云安全",
            "数据安全", "攻防实战", "合规法律", "新兴技术", "个人隐私保护"]
    idx = {c: [] for c in cats}
    if os.path.isdir(DOCS_DIR):
        for fn in sorted(os.listdir(DOCS_DIR)):
            if fn.lower().endswith(".txt"):
                idx["Web安全"].append({"file": fn, "title": os.path.splitext(fn)[0]})
    return idx


def title_map(idx):
    m = {}
    for cat, items in idx.items():
        for it in items:
            m[it["file"]] = (cat, it["title"])
    return m


TITLES = title_map(load_index())


# ---------- 会话持久化 ----------
def get_sid():
    """把会话 ID 放在 URL 里，刷新页面后仍能找回历史"""
    try:
        sid = st.query_params.get("sid")
        if isinstance(sid, list):
            sid = sid[0] if sid else None
        if not sid:
            sid = uuid.uuid4().hex[:10]
            st.query_params["sid"] = sid
        st.session_state.sid = sid
    except Exception:
        if "sid" not in st.session_state:
            st.session_state.sid = uuid.uuid4().hex[:10]
    return st.session_state.sid


def history_path(sid):
    return os.path.join(HIST_DIR, f"{sid}.json")


def load_history(sid):
    try:
        p = history_path(sid)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
    except Exception:
        pass
    return [{"role": "assistant", "content": WELCOME, "time": ""}]


def save_history(sid, messages):
    try:
        os.makedirs(HIST_DIR, exist_ok=True)
        p = history_path(sid)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(messages[-40:], f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = load_history(get_sid())
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def set_query(q):
    st.session_state.pending_query = q


# ---------- 检索与生成 ----------
def last_user_question(messages, skip_current=True):
    items = messages[:-1] if skip_current else messages
    for m in reversed(items):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def retrieval_query(question, messages):
    """只有带指代的问题才把上一轮问题拼进检索语句，避免短问题被上文带偏"""
    refs = ("它", "这个", "那个", "上述", "该", "其", "这种", "这些", "这几", "那",
            "以上", "此", "两者", "分别")
    if any(h in question for h in refs):
        prev = last_user_question(messages)
        if prev and prev != question:
            return f"{prev} {question}"
    return question


def build_history(messages):
    """取最近几轮的问答，转成消息对象"""
    turns = []
    for m in messages[:-1]:
        if m.get("role") in ("user", "assistant"):
            turns.append((m["role"], m.get("content", "")))
    hist = []
    pending_user = False
    for role, content in turns[-MAX_TURNS * 2:]:
        if role == "user":
            hist.append(HumanMessage(content=content))
            pending_user = True
        elif pending_user:
            hist.append(AIMessage(content=content))
            pending_user = False
    return hist


def retrieve(question, stores):
    """在各个分集合里检索，按距离合并取最相关的几条"""
    pairs = []
    for vs in stores:
        try:
            pairs.extend(vs.similarity_search_with_score(question, k=K_RETRIEVE))
        except Exception:
            continue
    pairs.sort(key=lambda x: float(x[1]))
    return pairs[:K_RETRIEVE]


def answer_stream(question, context, hist, llm):
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", SYS_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )
    return chain.stream({"context": context, "history": hist, "question": question})


def gen_suggestions(question, answer, llm):
    """回答结束后推荐 3 个相关问题"""
    try:
        msg = llm.invoke(SUGGEST_PROMPT.format(question=question, answer=answer[:400]))
        text = msg.content if hasattr(msg, "content") else str(msg)
        lines = [l.strip(" -·\t。") for l in text.splitlines() if l.strip()]
        out = []
        for l in lines:
            l = l.lstrip("0123456789.、）) ")
            if 4 <= len(l) <= 24 and l not in out:
                out.append(l)
        return out[:3]
    except Exception:
        return []


def sources_of(pairs):
    out = []
    seen = set()
    for doc, score in pairs:
        src = doc.metadata.get("source", "")
        fn = os.path.basename(src)
        if fn in seen:
            continue
        seen.add(fn)
        cat, title = TITLES.get(fn, ("综合知识", os.path.splitext(fn)[0]))
        out.append({
            "title": title,
            "cat": cat,
            "snippet": doc.page_content[:160].replace("\n", " "),
            "score": round(float(score), 3),
        })
    return out


# ---------- 页面 ----------
init_state()

if not os.path.exists(os.path.join(VECTOR_DIR, "chroma.sqlite3")):
    st.warning("⚠️ 向量库尚未建立，请先运行 python ingest.py 建库")
    st.stop()

vs, llm = load_resources()
idx = load_index()

with st.sidebar:
    st.title("🛡️ 系统信息")
    st.markdown("**技术栈**")
    st.markdown("- 大模型：通义千问 qwen-plus")
    st.markdown("- 向量库：Chroma")
    st.markdown("- Embedding：text-embedding-v2")
    st.markdown("- 框架：LangChain + Streamlit")
    st.divider()

    st.markdown("**知识库分类快速提问**")
    categories = {
        "Web安全": "SQL注入的原理和防御方法是什么？",
        "系统安全": "Linux系统安全加固有哪些要点？",
        "密码学": "对称加密和非对称加密有什么区别？",
        "网络安全": "防火墙的工作原理是什么？",
        "云安全": "云安全共享责任模型是什么？",
        "数据安全": "数据分类分级一般怎么划分？",
        "攻防实战": "渗透测试的标准流程是什么？",
        "合规法律": "等保2.0是什么？",
        "新兴技术": "什么是零信任安全架构？",
        "个人隐私保护": "手机App权限应该怎么管理？",
        "安全意识与防护": "怎么识别和防范电信网络诈骗？",
    }
    st.selectbox(
        "选择分类快速提问：",
        ["选择分类..."] + list(categories.keys()),
        key="cat_select",
        on_change=lambda: set_query(categories[st.session_state.cat_select]) if st.session_state.cat_select != "选择分类..." else None,
    )
    st.divider()

    with st.expander("📖 知识库浏览"):
        st.caption("直接翻阅内置知识库原文，支持按分类检索")
        cat_names = [c for c in idx.keys() if idx[c]]
        kb_cat = st.selectbox("分类", cat_names, key="kb_cat")
        items = idx.get(kb_cat, [])
        if items:
            titles = [f"{i+1}. {it['title']}" for i, it in enumerate(items)]
            ki = st.selectbox("文档", titles, key=f"kb_doc_{cat_names.index(kb_cat)}")
            pos = titles.index(ki)
            fn = items[pos]["file"]
            path = os.path.join(DOCS_DIR, fn)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    body = f.read()
                st.markdown(f'<div class="doc-view">{html.escape(body)}</div>', unsafe_allow_html=True)
        else:
            st.caption("该分类暂无文档")

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME, "time": ""}]
        save_history(st.session_state.sid, st.session_state.messages)
        st.rerun()

    chat_text = "\n\n".join(
        [f"[{m.get('role')}] {m.get('content')}" for m in st.session_state.messages]
    )
    st.download_button(
        "📥 导出对话记录",
        data=chat_text,
        file_name=f"对话记录_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()
    with st.expander("ℹ️ 关于项目"):
        st.markdown("""
**网络安全智能问答助手**

基于 RAG 检索增强生成技术构建的网络安全知识问答系统。

**组员**：
- 张宜彤（代码实现）
- 刘思语（PPT制作）
- 王茜（上台演讲）

《生成式人工智能》课程项目
网安241/242/243班
        """)

st.title("🛡️ 网络安全智能问答助手")
st.caption("基于大语言模型与 RAG 检索增强生成 · 《生成式人工智能》课程项目")

st.markdown("""
<div class="project-card">
    <h3>🎯 项目简介</h3>
    <p>本系统基于检索增强生成（RAG）技术，将大语言模型与专业网络安全知识库结合，实现准确、可溯源的网络安全知识问答。知识库覆盖 Web 安全、系统安全、密码学、网络安全、云安全、数据安全、攻防实战、合规法律、新兴技术、个人隐私保护、安全意识与防护十一个方向，支持多轮追问与知识库原文查阅。</p>
</div>
""", unsafe_allow_html=True)

with st.expander("🔍 RAG工作原理（点击展开）", expanded=False):
    st.markdown("""
    1. **用户提问**：输入自然语言问题，结合上一轮对话补全指代
    2. **向量检索**：把问题转成向量，在 Chroma 知识库中找出最相关的文档片段
    3. **相关度判断**：检索结果相关度不足时直接说明知识库未覆盖，不硬编答案
    4. **拼接上下文**：把检索到的原文片段与历史对话一起放进提示词
    5. **生成回答**：通义千问基于这些资料组织答案，流式输出并标注参考来源
    """)
    st.info("💡 RAG 的优势：回答基于真实知识库内容，可追溯到具体文档，能有效抑制大模型幻觉。")

st.markdown("**💡 试试这些问题：**")
q_cols = st.columns(3)
quick = ["SQL注入怎么防御？", "XSS有哪几种类型？", "什么是零信任？"]
quick_map = {
    "SQL注入怎么防御？": "SQL注入的防御方法有哪些？",
    "XSS有哪几种类型？": "XSS跨站脚本攻击有哪几种类型？",
    "什么是零信任？": "什么是零信任安全架构？",
}
for c, label in zip(q_cols, quick):
    with c:
        st.button(label, key=f"quick_{label}", on_click=set_query, args=(quick_map[label],), use_container_width=True)

st.divider()

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))
        if msg.get("elapsed"):
            st.caption(f"⏱️ 检索+生成耗时 {msg['elapsed']} 秒")
        if msg.get("sources"):
            with st.expander(f"📚 参考来源（{len(msg['sources'])} 条知识库片段）"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}] {s['title']}** · *{s['cat']}*")
                    st.caption(s["snippet"] + "...")

prompt = st.chat_input("请输入你的网络安全问题...")
if st.session_state.get("pending_query"):
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "time": datetime.now().strftime("%H:%M"),
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    srcs = []
    used = []
    answer = ""
    elapsed = 0.0

    with st.chat_message("assistant"):
        t0 = time.time()
        rq = retrieval_query(prompt, st.session_state.messages)
        embed_error = False
        try:
            pairs = retrieve(rq, vs)
        except Exception:
            pairs = []
            embed_error = True
        top_score = float(pairs[0][1]) if pairs else 9.9

        if embed_error:
            answer = ("向量检索服务暂时不可用（接口额度或网络异常），请稍后再试一次。\n\n"
                      "如果反复出现，可以在页面顶部刷新后重试。")
            st.markdown(answer)
            elapsed = round(time.time() - t0, 1)
            st.caption(f"⏱️ 检索失败 {elapsed} 秒 · 已停止生成，避免编造回答")
        elif not pairs or top_score > SIM_THRESHOLD:
            answer = ("根据现有知识库暂未找到相关内容。\n\n"
                      "我是专门回答网络安全问题的助手，知识库主要覆盖 Web 安全、系统安全、密码学、"
                      "网络安全、云安全、数据安全、攻防实战、合规法律、新兴技术、个人隐私保护、"
                      "安全意识与防护十一个方向。你可以换个问法，或者点左侧的分类快速提问试试。")
            st.markdown(answer)
            elapsed = round(time.time() - t0, 1)
            st.caption(f"⏱️ 检索耗时 {elapsed} 秒 · 相关度不足，已拒绝编造回答")
        else:
            used_pairs = [(d, s) for d, s in pairs if float(s) <= SIM_THRESHOLD][:3]
            used = [d for d, _ in used_pairs]
            context = "\n\n".join([d.page_content for d in used])
            hist = build_history(st.session_state.messages)
            with st.spinner("正在检索知识库并组织回答..."):
                answer = st.write_stream(answer_stream(prompt, context, hist, llm))
            elapsed = round(time.time() - t0, 1)
            st.caption(f"⏱️ 检索+生成耗时 {elapsed} 秒 · 参考了 {len(used)} 条知识库片段")

            srcs = sources_of(used_pairs)
            if srcs:
                with st.expander(f"📚 参考来源（{len(srcs)} 条知识库片段）"):
                    for i, s in enumerate(srcs, 1):
                        st.markdown(f"**[{i}] {s['title']}** · *{s['cat']}*")
                        st.caption(s["snippet"] + "...")

    new_sugs = []
    if used:
        with st.spinner("正在整理相关问题..."):
            new_sugs = gen_suggestions(prompt, answer, llm)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "time": datetime.now().strftime("%H:%M"),
        "elapsed": elapsed if used else None,
        "sources": srcs,
        "suggestions": new_sugs,
    })
    save_history(st.session_state.sid, st.session_state.messages)

# 相关问题推荐（每次对话结束后刷新，点击由回调写入 pending_query）
last = st.session_state.messages[-1] if st.session_state.messages else {}
sugs = last.get("suggestions") or []
if last.get("role") == "assistant" and sugs:
    st.markdown("**💡 你可能还想问：**")
    s_cols = st.columns(len(sugs))
    for i, (c, q) in enumerate(zip(s_cols, sugs)):
        with c:
            st.button(q, key=f"sug_{i}_{q[:6]}", on_click=set_query, args=(q,), use_container_width=True)

