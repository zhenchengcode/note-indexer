import os
import jieba
import tiktoken
import faiss
import pickle
import openai
from typing import List
import re

# 设置你的 OpenAI Key
openai.api_key = "your-api-key"

# ========= 1. 中文分块函数 =========
def split_chinese_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[。！？\?！])', text)
    return [s.strip() for s in sentences if s.strip()]

def estimate_tokens(text: str, encoding_name="cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def chunk_text_zh(text, max_tokens=500, overlap=50) -> List[str]:
    paragraphs = re.split(r'\n+', text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        sentences = split_chinese_sentences(para)
        for sentence in sentences:
            tok = estimate_tokens(sentence)
            if current_tokens + tok > max_tokens:
                chunks.append("".join(current_chunk))
                # 重叠策略
                if overlap > 0:
                    overlap_chunk = []
                    tok_count = 0
                    for s in reversed(current_chunk):
                        tok_s = estimate_tokens(s)
                        tok_count += tok_s
                        overlap_chunk.insert(0, s)
                        if tok_count >= overlap:
                            break
                    current_chunk = overlap_chunk
                    current_tokens = sum(estimate_tokens(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += tok

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks

# ========= 2. 调用 GPT 生成摘要 =========
def summarize_chunk(chunk: str, model="gpt-3.5-turbo") -> str:
    messages = [
        {"role": "system", "content": "你是一个中文笔记摘要助手。请提炼以下段落的核心内容，用简洁的方式表达。"},
        {"role": "user", "content": chunk}
    ]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"摘要出错：{e}")
        return ""

# ========= 3. 生成 embedding =========
def get_embedding(text: str, model="text-embedding-3-small") -> List[float]:
    try:
        response = openai.Embedding.create(
            model=model,
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding出错：{e}")
        return []

# ========= 4. 向量数据库保存 =========
def save_to_faiss(embeddings: List[List[float]], metadata: List[str], faiss_path="faiss_index", meta_path="metadata.pkl"):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    faiss.write_index(index, faiss_path)
    print("已保存向量数据库。")

# ========= 5. 主流程 =========
import numpy as np

def index_notes_from_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text_zh(text)
    summaries = [summarize_chunk(c) for c in chunks]
    embeddings = [get_embedding(s) for s in summaries]

    save_to_faiss(embeddings, summaries)
