# -*- coding: utf-8 -*-
"""
File: rag_chatbot.py
Author: Tomoyuki H. <deartomo@gmail.com>
Created: 2025-09-02
Last Modified: 2025-09-02
Description: RAG チャットボット
             - PDF/HTML/テキスト/CSV ドキュメントを読み込み
             - LangChain + FAISS でベクトル検索
             - Ollama + HuggingFaceEmbeddings を用いた回答生成
License: Apache License 2.0
"""

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

import os
import sys
import time
import faiss
import torch
import numpy as np
import magic

def load_documents():
    docs = []    # LangChain用の Document オブジェクトの空リスト（[Document(...), Document(...), Document(...), ...]）
    loaders = {
        "application/pdf": PyMuPDFLoader,
        "text/html": UnstructuredHTMLLoader,
        "text/plain": lambda fp: TextLoader(fp, encoding="utf-8"),
        "text/csv": CSVLoader
        }
#    for filename in os.listdir("docs"):
#        filepath = os.path.join("docs", filename)
    for filename in os.listdir("docs_employee"):
        filepath = os.path.join("docs_employee", filename)
        mime = magic.from_file(filepath, mime=True)
        print(f"{filename} → MIME: {mime}")
        loader_class = loaders.get(mime)
        if loader_class:
            loader = loader_class(filepath) if callable(loader_class) else loader_class
            docs.extend(loader.load())
        else:
            print(f"{filename} は未対応のファイル形式です。スキップします。")
    return docs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,    # 1つのチャンクの大きさを 250 文字とする。
        chunk_overlap=50,    # separatoesで区切られずに、chunk_sizeの上限に達したことにより分割された場合、前のチャンクと 50 文字重複する。
        separators=['。', '\n', ''],    #  まず句点'。'で分割しようとし、句点がない場合は改行'\n'で分割する。句点も改行もなければchunk_sizeで分割する。
        keep_separator="end"
        )
    return splitter.split_documents(documents)    # スプリッターに Document オブジェクトのリストを渡してチャンクに分割し、呼び出し元に返す。

def create_vector_db(chunks, embeddings):
    print("ベクトルストアを生成中...")
    vectors = np.array(embeddings.embed_documents([doc.page_content for doc in chunks])).astype("float32")
    print(f"基本基本ベクトル数: {vectors.shape[0]} 次元: {vectors.shape[1]}")
    
    faiss.normalize_L2(vectors)    # コサイン類似度計算用に正規化する。
    
    d = vectors.shape[1]
    if vectors.shape[0] < 80:
        print("IndexFlatIPを使用します。")
        index = faiss.IndexFlatIP(d)
    else:
        nlist = min(100, max(1, vectors.shape[0] // 40))
        print(f"IndexIVFFlatを使用します。クラスタ数 nlist: {nlist}")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)    # IVF(IndexIVFFlat) は、事前に全ベクトルをクラスタリングして、クエリと近いクラスタだけ探索する。
        index.train(vectors)    # .train() で事前にクラスタを作る（これがIVFの肝）
    index.add(vectors)
    
    # LangChainのFAISS Wrapperを使ってベクトルストアを作成する。
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(chunks)})
    index_to_docstore_id = {i: str(i) for i in range(len(chunks))}
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
        )
        
    vectorstore.save_local("vectorstore")    # vectorstoreを保存する。
    return vectorstore

def load_vector_db(embeddings):
    print("既存のベクトルストアを読み込み中...")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return vectorstore
    
def answer_with_filtered_docs(query, retriever, llm, embeddings, sim_threshold=0.6):
    docs = retriever.invoke(query)     
    if not docs:
        result = llm.invoke(query)  # あるいは「情報が不足しています。」と返す
        return {
            "result": result,
            "source_documents": []
        }
        
    query_vec = np.array(embeddings.embed_query(query)).reshape(1, -1)
    doc_texts = [doc.page_content for doc in docs]
    doc_vecs = np.array(embeddings.embed_documents(doc_texts)).reshape(len(doc_texts), -1)
    sims = cosine_similarity(query_vec, doc_vecs)[0]    # コサイン類似度の計算
     
    filtered_docs = []
    print("--------------------------------------------------")
    print("【検索結果】:")
    for i, (doc, sim) in enumerate(zip(docs, sims), start=1):
        print(f"--- [{i}] ---")
        print(f"類似度: {sim:.4f}")
        print(f"資料: {doc.metadata.get('source', '').strip()}")
        print(f"内容: {doc.page_content[:100]} ...")
        if sim >= sim_threshold:
            doc.metadata['score'] = sim
            filtered_docs.append(doc)
    print("--------------------------------------------------")
    
    if not filtered_docs:
        prompt = query
    else:
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        prompt = f"以下の情報に基づいて質問に答えてください。\n\n{context}\n\n質問: {query}"

    result = llm.invoke(prompt)
    return {
        "result": result,
        "source_documents": filtered_docs
        }

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA非対応環境です。CPUを使用します。")
        device = "cpu"
        
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE",     # 文をベクトル化（埋め込み）するモデルを指定する。
        model_kwargs={"device": device},              # "cuda"または"cpu"
        encode_kwargs={"normalize_embeddings": True}  # cosine類似度に適した正規化
        )
    
    if "--rebuild" in sys.argv:
        docs = load_documents()
        print(f"読み込んだドキュメント数: {len(docs)}")
        chunks = chunk_documents(docs)
        print(f"分割後のチャンク数: {len(chunks)}")
        vectorstore = create_vector_db(chunks, embeddings)
        print("ベクトルストア作成完了！")
    else:
        vectorstore = load_vector_db(embeddings)
        print("既存のベクトルストアを読み込み完了！")

    if device == "cuda" and faiss.get_num_gpus():
        print("インデックスをGPUへ転送中...")
        res = faiss.StandardGpuResources()    # FAISS に CUDA の GPU リソース（メモリプールやストリーム）を管理させるためのオブジェクト
        vectorstore.index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)   # vectorstore.indexをCUDA デバイス 0 に転送する。
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 5})
    llm = OllamaLLM(model="hf.co/mmnga/llm-jp-3.1-1.8b-instruct4-gguf:Q8_0")    # 他のモデルの例: "gpt-oss:20b"
    
    print("チャットを開始します。終了するには exit と入力してください。")
    
    while True:
        query = input("【質問】: ")
        if query.lower() in ["exit", "quit"]:
            break

        start = time.time()
        response = answer_with_filtered_docs(query, retriever, llm, embeddings, sim_threshold=0.5)
        elapsed = time.time() - start

        print("【回答】:", response["result"])
        print("【処理時間】:{:.2f} 秒".format(elapsed))
#        print("【出典】:")
#        for i, doc in enumerate(response["source_documents"], start=1):
#            print(f"--- [{i}] ---")
#            print(f"評価: {doc.metadata.get('score', ''):.4f}")
#            print(f"資料: {doc.metadata.get('source', '').strip()}")
#            print(f"内容: {doc.page_content[:100]} ...")
        print("==================================================")

