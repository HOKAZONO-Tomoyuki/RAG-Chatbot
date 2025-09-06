# RAGチャットボット

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![FAISS](https://img.shields.io/badge/vectorstore-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-lightgrey.svg)](https://ollama.ai/)

シンプルな構成で **Retrieval-Augmented Generation (RAG)** を実現する Python プログラムです。  
Ollama を利用してローカル環境で LLM を実行し、PDF・HTML・テキスト・CSV などを検索対象として活用できます。  
商用利用可能な **Apache License 2.0** で公開しています。

---

## ✨ 特徴
- LangChain + FAISS によるベクトル検索
- HuggingFace Embeddings (LaBSE) に対応
- Ollama を使ったローカル LLM 実行
- CUDA 対応 (GPU で高速化)
- PDF / TXT / CSV / HTML を入力可能

---

## 💿 インストール

「技術解説記事」 (docs/overview.md)の「4 構築」および「5 仮想環境」をお読みください。

## 💻 使い方

### 1. ベクトルストアの作成
```
python rag_chatbot.py --rebuild
```

### 2. チャットボットの起動
```
python rag_chatbot.py
```

### 実行例
```
【質問】: こんにちは
【回答】: こんにちは！ご質問があればどうぞ。
```

## 📂 ディレクトリ構成
```
.
├── LICENSE                   # Apache 2.0 ライセンス
├── README.md                 # 簡易説明
├── rag_chatbot.py            # メインプログラム
├── third_party_licenses.txt  # 3rdパーティ ライセンス
└── docs/
    └── overview.md           # 技術解説記事（詳細版）

```

## 📚 ドキュメント
* 技術解説記事 (docs/overview.md)
* LICENSE

## 📜 ライセンス
このプロジェクトは Apache License 2.0 の下で公開されています。

