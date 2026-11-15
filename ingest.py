"""
ingest.py - 把 docs 目录下的文档切分、向量化，存入本地 Chroma
运行一次即可，之后新增文档再重新运行

说明：Chroma 1.5 在单个集合内文本块数量超过一定规模（约近 1000）时，
磁盘索引会出现无法重载的问题，因此这里按批切成多个集合 kb_0 / kb_1 ...
存放，检索时在 app.py 里合并各集合的结果。
"""
import os
import json
import shutil

import numpy as np
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")

EMBED_MODEL = "text-embedding-v2"
EMBED_BATCH = 10          # 单次向量化的文本条数
MAX_PER_COLLECTION = 800  # 单个集合的文本块上限


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


def embed_chunks(chunks):
    """调用通义千问 text-embedding-v3 批量向量化"""
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=EMBED_BATCH,
    )
    vectors = []
    total = len(chunks)
    for i in range(0, total, EMBED_BATCH):
        batch = [c.page_content for c in chunks[i:i + EMBED_BATCH]]
        vectors.extend(embeddings.embed_documents(batch))
        if (i // EMBED_BATCH) % 20 == 0:
            print(f"向量化进度 {i + len(batch)}/{total}", flush=True)
    print(f"向量化完成，共 {len(vectors)} 条")
    return np.array(vectors, dtype=np.float32)


def build_vectorstore(chunks, vectors):
    """分集合写入 Chroma"""
    if os.path.isdir(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
        print(f"已清除旧向量库 {VECTOR_DIR}")

    client = chromadb.PersistentClient(path=VECTOR_DIR)
    n = len(chunks)
    for start in range(0, n, MAX_PER_COLLECTION):
        end = min(start + MAX_PER_COLLECTION, n)
        name = f"kb_{start // MAX_PER_COLLECTION}"
        col = client.get_or_create_collection(name)
        metas = []
        for c in chunks[start:end]:
            m = dict(c.metadata or {})
            m["source"] = os.path.basename(m.get("source", "")) or "unknown"
            metas.append(m)
        col.add(
            ids=[f"c{i}" for i in range(start, end)],
            embeddings=vectors[start:end].tolist(),
            documents=[c.page_content for c in chunks[start:end]],
            metadatas=metas,
        )
        print(f"集合 {name}: {col.count()} 个文本块")

    total = sum(client.get_collection(f"kb_{i}").count()
                for i in range((n + MAX_PER_COLLECTION - 1) // MAX_PER_COLLECTION))
    print(f"向量库已保存到 {VECTOR_DIR}，共 {total} 个文本块")
    return total


def save_snapshot(chunks, vectors):
    """额外存一份向量快照，方便重建时不用重新调用接口"""
    path = os.path.join(BASE_DIR, "vector_store_data.npz")
    np.savez_compressed(
        path,
        vectors=vectors,
        documents=np.array([c.page_content for c in chunks], dtype=object),
        sources=np.array([os.path.basename((c.metadata or {}).get("source", "")) for c in chunks], dtype=object),
    )
    with open(os.path.join(BASE_DIR, "vector_store_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"chunks": len(chunks), "max_per_collection": MAX_PER_COLLECTION}, f, ensure_ascii=False)
    print(f"向量快照已保存 {path}")


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("docs 目录下没有文档，请先放一些 PDF 或 TXT")
    else:
        chunks = split_documents(docs)
        vectors = embed_chunks(chunks)
        build_vectorstore(chunks, vectors)
        save_snapshot(chunks, vectors)
        print("建库完成！")
