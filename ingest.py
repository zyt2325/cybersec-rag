"""
ingest.py - 把 docs 目录下的文档切分、向量化，存入本地 Chroma
运行一次即可，之后新增文档再重新运行
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "vector_store")


def load_documents():
    """加载 docs 目录下所有 PDF 和 TXT"""
    loaders = [
        DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
        DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True,
                        loader_kwargs={"autodetect_encoding": True}),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    print(f"共加载 {len(docs)} 个文档片段")
    return docs


def split_documents(docs):
    """切分成块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "；", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"切分成 {len(chunks)} 个文本块")
    return chunks


def build_vectorstore(chunks):
    """用通义千问 text-embedding-v3 做向量化，存入 Chroma"""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v3",
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR,
    )
    print(f"向量库已保存到 {VECTOR_DIR}")
    return vs


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("docs 目录下没有文档，请先放一些 PDF 或 TXT")
    else:
        chunks = split_documents(docs)
        build_vectorstore(chunks)
        print("建库完成！")
