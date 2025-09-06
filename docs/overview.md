# RAGチャットボット

初版: 2025年8月28日

## もくじ

[1 はじめに](#1-はじめに)

[1.1 市場動向](#11-市場動向)

[1.2 生成AIに対する懸念](#12-生成aiに対する懸念)

[1.3 この文書の目的](#13-この文書の目的)

[2 概要](#2-概要)

[2.1 LLMの種類](#21-llmの種類)

[2.2 RAGの概要](#22-ragの概要)

[3 動作環境](#3-動作環境)

[3.1 ハードウェア](#31-ハードウェア)

[3.2 ソフトウェア](#32-ソフトウェア)

[4 構築](#4-構築)

[4.1 作業の概要](#41-作業の概要)

[4.2 OSとデスクトップ環境](#42-osとデスクトップ環境)

[4.3 GPUデバイスドライバーとCUDA](#43-gpuデバイスドライバーとsuda)

[4.3.1 デバイスの確認](#431-デバイスの確認)

[4.3.2 デバイスドライバーのインストール](#432-デバイスドライバーのインストール)

[4.3.3 CUDAのインストール](#433-cudaのインストール)

[4.4 OllamaのインストールとLLMのダウンロード](#44-ollamaのインストールとllmのダウンロード)

[4.4.1 Ollamaのインストール](#441-ollamaのインストール)

[4.4.2 LLMのダウンロード](#442-llmのダウンロード)

[4.5 Pythonのインストール](#45-pythonのインストール)

[4.6 Minicondaのインストールと仮想環境の作成](#46-minicondaのインストールと仮想環境の作成)

[4.6.1 Minicondaのインストール](#461-minicondaのインストール)

[4.6.2 仮想環境の作成](#462-仮想環境の作成)

[5 仮想環境](#5-仮想環境の整備)

[5.1 Pytorch](#51-pytorch)

[5.2 faiss-gpu](#52-faiss-gpu)

[5.3 LangChain](#53-langchain)

[5.4 NumPy](#54-numpy)

[5.5 sentence-transformers](#55-sentence-transformers)

[5.6 scikit-learn](#56-scikit-learn)

[5.7 文書ファイルを分析するライブラリー群](#57-文書ファイルを分析するライブラリー群)

[6 RAGアプリケーション](#6-ragアプリケーション)

[6.1 アプリケーションの概要](#61-アプリケーションの概要)

[6.2 主制御部(\_\_main\_\_)](#62-主制御部\_\_main\_\_)

[6.3 埋め込み(Embedding)](#63-埋め込みembedding)

[6.4 ベクトルストア(Vector Store)](#64-ベクトルストアvector-store)

[6.4.1 ベクトルストアの概要](#641-ベクトルストアの概要)

[6.4.2 文字列の抽出](#642-文字列の抽出)

[6.4.3 文字列の分割](#643-文字列の分割)

[6.4.4 文字列のベクトル化](#644-文字列のベクトル化)

[6.4.5 インデックスの作成](#645-インデックスの作成)

[6.4.6 ラッパーによる統合](#646-ラッパーによる統合)

[6.4.7 ベクトルストアの使用](#647-ベクトルストアの使用)

[6.5 LLMの使用](#65-llmの使用)

[7 調整](#7-調整)

[7.1 ドキュメントの重複排除](#71-ドキュメントの重複排除)

[7.2 類似度のフィルター](#72-類似度のフィルター)

[7.3 検索データの補完](#73-検索データの補完)

[8 おわりに](#8-おわりに)

[A 補足](#a-補足)

[A.1 脚注・出典](#a1-脚注出典)


## 1 はじめに

### 1.1 市場動向

近年、大規模言語モデル(LLM: Large Language Model)を使って文を生成する人工知能(AI)が脚光を浴びています。
実際のサービスとして次のようなクラウド型生成AIサービスが提供されています。
* ChatGPT (OpenAI)
* Gemini (Google)
* Copilot (Microsoft)

総務省が発表した令和7年版情報通信白書[(※1)](#1-令和7年版情報通信白書-総務省)によれば、日本における生成AIの利用者比率は海外に比べて低い水準に留まっています。
企業における生成AIの活用方針策定状況を見ると、「積極的に活用する方針である」と「活用する領域を限定して利用する方針である」と回答した企業の比率は合わせて約50%でした。

### 1.2 生成AIに対する懸念

企業が生成AIの活用に慎重になる理由として技術的リスク、社会的リスクが挙げられます。
例えば、技術的リスクとして生成AIの回答は一貫していないということが挙げられます。生成AIの回答(文)は同じ質問であっても質問する度に異なります。
その他の技術的リスクとして生成AIは誤った回答文を生成する場合があります。この現象はハルシネーションと呼ばれます。
一方、社会的リスクとしては、知的財産権を侵害する可能性や機密情報の流出などが挙げられます。

先に挙げた令和7年版情報通信白書のデータ集「第Ⅰ部第1章第2節 21. 生成AI導入に際しての懸念事項（国別）」[(※2)](#2-令和7年版情報通信白書-総務省-データ集)によると、
最も多い意見は「効果的な活用方法がわからない」(30.1%)で、次に多い意見は「社内情報の漏洩などセキュリティリスクがある」(27.6%)です。

先ず「効果的な活用方法がわからない」という点ですが、これは言い換えると「生成AIが業務改善にどのように役立つか想像できない」ということでしょう。
これはごもっともな意見だと思います。多くの生成AIはインターネット上に公開されている膨大な情報を学習していますが、個々の企業の内部事情は学習していません。
また、生成AIは学習した情報の最終日(AI Knowledge Cutoff Date)以降の出来事を知りません。
つまり、業務内容や最新の市場動向について生成AIに質問しても適切な回答が得られません。

次に「社内情報の漏洩などセキュリティリスクがある」という点ですが、これもごもっともな意見だと思います。
生成AIサービスを提供する企業は、利用者が入力した文を蓄積して生成AIサービスの改良に利用します。
つまり、入力した内容が不特定の第三者に知られる可能性があります。
そのため、企業は業務内容や顧客情報などをクラウド型の生成AIサービスに入力できません。

### 1.3 この文書の目的

上の節「1.2 生成AIに対する懸念」で取り上げた問題(企業での効果的な活用と情報漏洩などのセキュリティリスク)に対して2つのアプローチがあります。
1つは、検索拡張生成(RAG: Retrieval-Augmented Generation)を用いて、企業が蓄積している情報を生成AIで利用することです。
もう1つは、オープンクラウドの文章生成AIサービスを使うのではなく、LLMを自社の環境(On-Premises)で運用することです。

この記事では価格が10万円～20万円程度の消費者向けのパソコンにLLMをダウンロードし、RAGを用いて独自の情報を活用する文章生成AI(以降「RAGチャットボット」と言います)を実装する方法について述べます。

この記事はRAGチャットボットの導入を検討している方、特に本格的導入に先立ち、高額な費用を掛けずにRAGチャットボットに触れてみたいと考えている技術者を対象にした記事です。

この記事の中ではPythonプログラミングで用いられる「クラス」や「インスタンス」などの用語が現れます。ですので、この記事を読むためにはPythonプログラミングの基本的な知識を有していることが求められます。

## 2 概要

### 2.1 LLMの種類

LLMは文章生成AIの中核的技術です。2018年にGoogleからBARTが、OpenAIからGPTが発表され、現在では数多くのLLMが公開されています。
この記事では国立情報学研究所の大規模言語モデル研究開発センターが公開しているLLM-jp[(※3)](#3-llm\-jp-国立情報学研究所-大規模言語モデル研究開発センター)を利用します。

LLMの性能を測る指標としてパラメータ数(モデルサイズ)が挙げられます。
パラメータ数の大きなLLMは、パラメータ数の小さなLLMと比べて、質問に対してより適切で精度の高い回答文を生成することができます。
例えば、LLM-jpのバージョン3.1では3つのモデルが公開されています。

* LLM-jp-3.1-8x13b (パラメータ数: 約730億)
* LLM-jp-3.1-13b (パラメータ数: 約140億)
* LLM-jp-3.1-1.8b (パラメータ数: 約20億)

ちなみに、ChatGPTやGeminiなどの商用LLMのパラメータ数は公表されていません。

モデルサイズの小さいLLMのほうが、モデルサイズの大きいLLMより処理時間が短くなります。
つまり、質問を入力した後、早く回答が出力されます。また、GPUを使って処理するほうがCPUで処理するより処理時間が短くなります。

この記事では使用するGPUに搭載されているメモリーが8GBであることからLLM-jp-3.1-1.8bを使います。
もちろん、GoogleのGemma3を使うこともできます。詳しくは後の章でお話しします。

### 2.2 RAGの概要

RAGとは、LLMに個別の事情に適した回答を生成させるための手法です。以下に流れの概要を示します。

1. 個別の事情に関するテキストファイルやPDFファイル、HTMLファイルなどを集める。
2. 集めた文書から検索可能なデータベースを作成する。
3. 利用者が質問文を入力したあと、その文を検索キーとしてデータベースを検索し、内容の近い文書をいくつか選び出す。
4. 検索結果を得たあと、質問文と合わせてLLMに渡し、回答文を得る。

データベースを作成するのは初回、または新しい文書を追加したいときだけです。
アプリケーションを停止して再度実行したときは、作成したデータベースを読み込んで③と④を繰ります。

文書からデータベースを作成するソフトウェアやそのデータベースを検索するソフトウェアはライブラリーとして既に公開されていますので、最初から全てを作る必要はありません。
詳しくは後の章でお話しします。

## 3 動作環境

### 3.1 ハードウェア

使用するコンピュータ(パソコン)の主なハードウェア仕様を以下に示します。

* CPU: AMD Ryzen 9 7900
* GPU: NVIDIA Geforce RTX 4060 8GB
* Memory: 64GB (DDR5 PC5-44800 32GB x2)
* Storage: 1TB SSD M.2 PCIe4.0

マザーボードやモニターを合わせてもおよそ20万円で一式揃えることができます。
このパソコンは仮想基盤の検証用にも使っていますのでメモリーを64GB搭載していますが、LLMとRAGの検証だけならメモリーは32GBでも良いと思います。
また、LLMとRAGの検証のためには予算の許す限り高性能なGPUを使うことをお勧めします。特にGPUに搭載しているメモリーの容量が大きいものが良いでしょう。

### 3.2 ソフトウェア

OSはDebianを使います。

* debian 12 (Bookworm)

インストールする主なソフトウェアは次の通りです。

* GPUデバイスドライバー
* CUDA (注1)
* Ollama
* Python
* Miniconda (注2)  
(注1) CUDAはNVIDIAのGPUを活用するためのソフトウェアです。  
(注2) Minicondaの代わりにAnacondaを使うこともできます。

これらの他、複数のPythonライブラリーをインストールします。Pythonのライブラリーについては後の章でお話しします。

## 4 構築

### 4.1 作業の概要

RAGチャットボットを構築する作業の概要を以下に示します。

1. OSとデスクトップ環境のインストール
2. GPUデバイスドライバーとCUDAのインストール
3. OllamaのインストールとLLMのダウンロード
4. Pythonのインストール
5. Minicondaのインストールと仮想環境の作成
6. アプリケーションの開発

この章では、OSのインストールからPythonの仮想環境の作成までを説明します。

### 4.2 OSとデスクトップ環境

この記事ではOSおよびデスクトップ環境のインストール手順については触れません。Debianの公式サイト[(※4)](#4-debian-gnulinux-インストールガイド-devian)などを見て作業してください。

デスクトップ環境は、RAGチャットボットを実行する上で必要なものではありません。
ただ、システムを構築する過程でインターネットにあるホームページの内容を読んだりすることがあるので、デスクトップ環境とウェブブラウザーはインストールしておいたほうが良いです。

また、この記事ではシステムのセキュリティについては全く考慮していません。
root権やファイルおよびディレクトリのパーミッションについては読者ご自身でご注意ください。

### 4.3 GPUデバイスドライバーとCUDA

GPUのデバイスドライバーはNVIDIAの公式サイトからダウンロードしてインストールすることができますが、この記事ではDebianのリポジトリーからインストールする方法を説明します。
GPUを活用するRAGチャットボットでは、デバイスドライバーのバージョンとCUDAのバージョン、そしてPythonのライブラリーのバージョンの組み合わせが重要になります。
これらの組み合わせによってはソフトウェアが動作しないトラブルに見舞われます。
NVIDIAの公式サイトから手動で任意のファイルをダウンロードしてインストールしたあと、作業を進めて行く中でPythonのライブラリーがエラーで動かないなどの問題が生じると復旧が大変です。ですので、パッケージの依存関係が管理されているDebianのリポジトリーからaptを使ってインストールすることをお勧めします。

デバイスドライバーをインストールする手順はDebianの公式サイト[(※5)](#5-nvidiagraphicsdrivers-devian--wiki-)を参考にします。概要は次の通りです。

1. デバイスが認識されているか確認する。
2. ドライバーをインストールする。
3. CUDAをインストールする。

#### 4.3.1 デバイスの確認

OSがデバイスを認識しているかを確認する方法は二つあります。
1つはコマンド"lspci"で確認する方法、もう1つはコマンド"nvidia-detect"で確認する方法です。

**A) コマンド"lspci"で確認する方法**

```
 ~$ lspci -nn | egrep -i "3d|display|vga"
 01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD107 [GeForce RTX 4060] [10de:2882] (rev a1)
 0c:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Raphael [1002:164e] (rev c4)
```

上の実行結果では、NVIDIA GeForce RTX 4060の他、AMD Raphaelという出力があります。
Raphael（ラファエル）はRyzen 7000シリーズのデスクトップCPUのコードネームです。
AMD Raphaelはグラフィックス機能を持っているため、NVIDIAのGPUとあわせて2行出力されています。

**B) コマンド"nvidia-detect"で確認する方法**

コマンド"nvidia-detect"はリポジトリーからインストールします。次に手順を示します。

sources.listに"contrib"と"non-free"を追加します。

```
 ~$ sudo vi /etc/apt/sources.list
 ---
 #deb http://deb.debian.org/debian/ bookworm main non-free-firmware
 deb http://deb.debian.org/debian/ bookworm main contrib non-free non-free-firmware
 ---
```

パッケージリストを更新します。

```
 ~$ sudo apt update
```

カーネルヘッダーをインストールします。

```
 ~$ sudo apt install linux-headers-amd64
```

パッケージをインストールします。

```
 ~$ sudo apt install nvidia-detect
```

インストールしたあと、コマンド"nvidia-detect"を実行すると、デバイスが認識されているか確認できます。

```
 ~$ nvidia-detect
 Detected NVIDIA GPUs:
 01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD107 [GeForce RTX 4060] [10de:2882] (rev a1)
 
 Checking card:  NVIDIA Corporation AD107 [GeForce RTX 4060] (rev a1)
 Your card is supported by the default drivers.
 It is recommended to install the
     nvidia-driver
 package.
```

#### 4.3.2 デバイスドライバーのインストール

GPUのデバイスドライバをDebianのリポジトリーからインストールします。以下に手順を示します。
上の項「4.3.1 デバイスの確認」で"nvidia-detect"をインストールしている場合は、重複している作業をスキップしてください。

sources.listに"contrib"と"non-free"を追加します。

```
 ~$ sudo vi /etc/apt/sources.list
 ---
 #deb http://deb.debian.org/debian/ bookworm main non-free-firmware
 deb http://deb.debian.org/debian/ bookworm main contrib non-free non-free-firmware
 ---
```

パッケージリストを更新します。

```
 ~$ sudo apt update
```

カーネルヘッダーをインストールします。

```
 ~$ sudo apt install linux-headers-amd64
```

パッケージをインストールします。

```
 ~$ sudo apt install nvidia-driver firmware-misc-nonfree
```

インストールしたあと、リブートしてください。

```
 ~$ sudo reboot
```

OSが起動したら、コマンド"nvidia-smi"を実行します。以下のようにGPUの名前やメモリー容量が表示されれば成功です。

```
 ~$ nvidia-smi
 Thu Jul 24 10:13:24 2025
 +---------------------------------------------------------------------------------------+
 | NVIDIA-SMI 535.247.01             Driver Version: 535.247.01   CUDA Version: 12.2     |
 |-----------------------------------------+----------------------+----------------------+
 | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
 | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
 |                                         |                      |               MIG M. |
 |=========================================+======================+======================|
 |   0  NVIDIA GeForce RTX 4060        On  | 00000000:01:00.0 Off |                  N/A |
 |  0%   34C    P8              N/A / 115W |     11MiB /  8188MiB |      0%      Default |
 |                                         |                      |                  N/A |
 +-----------------------------------------+----------------------+----------------------+
 
 +---------------------------------------------------------------------------------------+
 | Processes:                                                                            |
 |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
 |        ID   ID                                                             Usage      |
 |=======================================================================================|
 |    0   N/A  N/A      1760      G   /usr/lib/xorg/Xorg                            4MiB |
 +---------------------------------------------------------------------------------------+
```

コマンド"nvidia-smi"を実行した結果、デバイスドライバーのバージョンは535.247.01、デバイスドライバーがサポートしているCUDAのバージョンは12.2であることがわかります。

#### 4.3.3 CUDAのインストール

CUDA(Compute Unified Device Architecture)は、NVIDIAが開発したGPUを使った汎用的な並列処理のための仕組みです。
CUDAもDebianリポジトリーからインストールします。手順を以下に示します。

パッケージをインストールします。

```
 ~$ sudo apt install nvidia-cuda-dev nvidia-cuda-toolkit
```

コマンド"nvcc -v"でバージョンを確認します。

```
 ~$ nvcc -V
 nvcc: NVIDIA (R) Cuda compiler driver
 Copyright (c) 2005-2022 NVIDIA Corporation
 Built on Wed_Sep_21_10:33:58_PDT_2022
 Cuda compilation tools, release 11.8, V11.8.89
 Build cuda_11.8.r11.8/compiler.31833905_0
```

コマンド"nvidia-smi"の結果からデバイスドライバーがサポートしているCUDAのバージョンは12.2であることが判明しています。
一方、インストールされたCUDAツールのバージョンは11.8です。バージョンが一致していませんが、この組み合わせに問題はありません。

### 4.4 OllamaのインストールとLLMのダウンロード

Ollamaは手元にあるパソコンで簡単にLLMを実行するためのプラットフォームです。Ollamaを使えば、いろいろなLLMをコマンド一つで試すことが出来ます。

#### 4.4.1 Ollamaのインストール

Ollamaをインストールする手順は公式サイト[(※6)](#6-download-ollama-ollama)を参考にします。

パッケージ"curl"をインストールします。curlはURLを指定してデータを転送するCLI(コマンドラインインターフェース)のツールです。

```
 ~$ sudo apt install curl
```

インストーラー(シェルスクリプト)をダウンロードして実行します。

```
 ~$ curl -fsSL https://ollama.com/install.sh | sh
```

状態を確認します。

```
 ~$ sudo systemctl status ollama
 ● ollama.service - Ollama Service
      Loaded: loaded (/etc/systemd/system/ollama.service; enabled; preset: enabled)
      Active: active (running) since Mon 2025-08-18 10:22:23 JST; 2h 38min ago
    Main PID: 1592 (ollama)
       Tasks: 14 (limit: 76031)
      Memory: 68.2M
         CPU: 350ms
      CGroup: /system.slice/ollama.service
              └─1592 /usr/local/bin/ollama serve
```

#### 4.4.2 LLMのダウンロード

コマンド"ollama run"を使うとLLMをローカルホストで実行することが出来ます。指定したモデルがローカルホストにない場合は、自動的にライブラリー"ollama.com/library"からダウンロードします。

```
 ~$ ollama run hf.co/mmnga/llm-jp-3.1-1.8b-instruct4-gguf:Q8_0
```

モデルのダウンロードが完了すると、直ぐにユーザーインターフェースが起動します。

```
 ~$ ollama run hf.co/mmnga/llm-jp-3.1-1.8b-instruct4-gguf:Q8_0
 ...
 >>> Send a message (/  for help)
```

挨拶をしたり、自己紹介を求めたり、自由に試してみてください。終了するときは"/bye"を入力します。

```
 ~$ ollama run hf.co/mmnga/llm-jp-3.1-1.8b-instruct4-gguf:Q8_0
 ...
 >>> こんにちは。はじめまして。
 こんにちは！はじめまして。ご質問やお困りのことがあれば、どうぞ何でもお知らせください。できる限り丁寧にお答えいたしますので、どうぞよろしくお願い
 いたします。
```

### 4.5 Pythonのインストール

pythonとpipをインストールします。

```
 ~$ sudo apt install python3 python3-pip
```

### 4.6 Minicondaのインストールと仮想環境の作成

この記事ではminicondaによる仮想環境にRAGチャットボットを構築します。
RAGチャットボットに挑戦する人にとって、仮想環境を使うことの最も大きな利点は「初めからやり直すのが簡単」ということでしょう。

#### 4.6.1 Minicondaのインストール

Minicondaをインストールする手順は公式サイト[(※7)](#7-installing-miniconda-anaconda)を参考にします。インストール手順を以下に示します。

インストール用のシェルスクリプトをダウンロードする。

```
 ~$ mkdir -p ~/miniconda3
 ~$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh-O ~/miniconda3/miniconda.sh
```

シェルスクリプトを実行する。

```
 ~$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```

シェルスクリプトを削除する。

```
 ~$ rm ~/miniconda3/miniconda.sh
```

ターミナルを一旦閉じて再起動するか、つぎのコマンドを実行してターミナルをリフレッシュすると、condaの仮想環境(base)が有効になります。

```
 ~$ source ~/miniconda3/bin/activate
```

仮想環境が有効になっている(仮想環境に入っている)と、プロンプトの前に仮想環境の名前が付きます。

```
 (base):~$ 
```

仮想環境を無効にする(仮想環境から出る)ためには、次のコマンドを実行します。仮想環境が無効になるとプロンプトが通常に戻ります。

```
 (base):~$ conda deactivate
 ~$ 
```

すべてのシェルでcondaが使えるように初期化する。

```
 ~$ conda init --all
```

condaを初期化した後、新にターミナルを開くと自動的にcondaの仮想環境(base)が有効になります。これを止めたい場合は、次のコマンドを実行して自動実行を無効にします。

```
 ~$ conda config --set auto_activate no
```

#### 4.6.2 仮想環境の作成

Pythonの仮想環境を作成します。仮想環境を作るときは、仮想環境の名前とPythonのバージョンを指定します。
この記事では仮想環境の名前を"rag-chatbot"とし、Pythonのバージョンは3.11を指定します。

```
 ~$ conda create -n rag-chatbot python=3.11
```

作成した仮想環境を有効にします。

```
 ~$ conda activate rag-chatbot
```

作成した仮想環境のディレクトリに移動します。

```
 (rag-chatbot):~$ cd miniconda3/envs/rag-chatbot/
```

## 5 仮想環境の整備

RAGチャットボットのプログラムで使用するライブラリーを仮想環境でインストールします。

### 5.1 Pytorch

Pytorchとは、機械学習のためのPythonのライブラリーです。pipコマンドを使ってtorchをインストールします。
このとき、CUDAのバージョンが11.8であることが判明しているので、それに対応したURLを指定します。

```
 (rag-chatbot):~$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

インストールが終了したら、以下に示す確認用のPythonコードを実行します。バージョンと"True"が表示されれば正常です。

```
 (rag-chatbot):~$ cat torch-test.py
 import torch
 print(torch.version.cuda)
 print(torch.cuda.is_available())
 (rag-chatbot):~$
 (rag-chatbot):~$ python3 torch_test.py
 11.8
 True
```

### 5.2 faiss-gpu

FAISSとは、ベクトルデータの類似性検索を行うためのライブラリーです。
FAISSにはCPUで動作するfaiss-cpuと、GPUで動作するfaiss-gpuがあります。この記事ではfaiss-gpuを使います。

faiss-gpuはコマンド"conda install"でインストールします。

```
 (rag-chatbot):~$ conda install -c pytorch faiss-gpu
```

インストールが終了したら、以下に示す確認用のPythonコードを実行します。バージョンと"True"が表示されれば正常です。

```
 (rag-chatbot):~$ cat faiss-gpu-test.py
 import faiss
 print(faiss.__version__)
 print(hasattr(faiss, "StandardGpuResources"))
 (rag-chatbot):~$ 
 (rag-chatbot):~$ python3 faiss-gpu-test.py
 1.9.0
 True
```

### 5.3 LangChain

LangChainとは、LLMを活用するアプリケーション開発に役立つライブラリーです。

```
 (rag-chatbot):~$ pip install langchain langchain-community langchain-ollama langchain-huggingface
```

### 5.4 NumPy

NumPyは、多次元の数値配列の計算を効率的に行うライブラリーです。

NumPyをインストールするときは、1.X系のNumPyをインストールします。これは、LangChainが1.X系のNumPyに依存しているためです。

```
 (rag-chatbot):~$ pip install "numpy>=1.25,<2.0"
```

もし、既に2.X系のNumPyがインストールされている場合は、以下に示すコマンドを実行して、NumPyを再インストールしてください。

```
 (rag-chatbot):~$ pip uninstall numpy -y
 (rag-chatbot):~$ pip cache purge
 (rag-chatbot):~$ pip install "numpy>=1.25,<2.0"
```

### 5.5 sentence-transformers

sentence-transformers(別名、SBERT)は、最新のEmbeddingモデルを使ったりトレーニングしたりするためのPythonモジュールです。

日本語で「埋め込み」を意味するEmbeddingとは、文(文字の列)を数値の列に変換することです。
変換後の数値の列を「ベクトル」と呼びます。そのため、Embeddingする(埋め込みする)ことを「ベクトル化する」と表現することがあります。

```
 (rag-chatbot):~$ pip install sentence-transformers
```

この記事ではsentence-transformersで定義されているクラス"SentenceTransformer"を使用していませんが、パッケージは必要ですのでインストールしてください。

### 5.6 scikit-learn

scikit-learnは、データ分析や機械学習のためのライブラリーです。
この記事では二つの文がどのくらい似ているかを判定するためにscikit-learnの中にある関数"cosin\_similarity"を使います。

```
 (rag-chatbot):~$ pip install scikit-learn
```

### 5.7 文書ファイルを分析するライブラリー群

RAGで用いるために集めた各種ファイルから検索可能なデータベースを作成するために幾つかのモジュールをインストールします。

* python-magic ァイルタイプを判別するライブラリー
* unstructured PDF、DOCX、HTMLなど、各種非構造化データからテキストデータを抽出するためのツール
* pymupd PDFを分析するライブラリー

上の3つのパッケージをpipでインストールします。

```
 (rag-chatbot):~$ pip install python-magic unstructured pymupdf
```

## 6 RAGアプリケーション

### 6.1 アプリケーションの概要

この記事で紹介するプログラムは大きく二つのパートに分かれます。

① データベースの準備
実行コマンドに添えられているオプションを確認して、データベースを準備する。 | |
a) オプションがある：各種ファイルから、RAG用のデータベースを作成して読み込む。 |
b) オプションがない：作成済みのデータベースを読み込む。 |

② 入出力ループ
特定の文字列が入力されるまでループする。 | |
a) 特定の文字列(例:"exit")が入力されたらループを抜けて、プログラムを終了する。 |
b) 文字列からデータベース内を検索して、類似性の高い文を選択する。 |
c) 入力された文字列と検索結果をLLMに渡して回答文を得る。 |

この記事ではプログラムの名前を\"rag-chatbot.py"としています。初めてプログラムを動かすとき、または新しい資料を追加してデータベースを作り直すときはオプション"--rebuild"を添えます。

```
 (rag-chatbot):~$ python3 rag-chatbot.py --rebuild
```

### 6.2 主制御部(\_\_main\_\_)

主制御部(\_\_main\_\_)の概要を以下に示します。

load\_documents()やchunk\_documents()、answer\_with\_filtered\_docs()はプログラム内で定義している関数です。あとの節で詳しくお話しします。

```
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
     llm = OllamaLLM(model="hf.co/mmnga/llm-jp-3.1-1.8b-instruct4-gguf:Q8_0")    # 他のモデルの例: "gpt-oss:20b"
     
     print("チャットを開始します。終了するには exit と入力してください。")
     
     while True:
         query = input("【質問】: ")
         if query.lower() in ["exit", "quit"]:
             break
 
         response = answer_with_filtered_docs(query, retriever, llm, embeddings, sim_threshold=0.5)
 
         print("【回答】:", response["result"])
```

#### 6.3 埋め込み(Embedding)

Emneddingのインスタンスを作るためのコードを以下に示します。

```
 from langchain_huggingface import HuggingFaceEmbeddings
 
 embeddings = HuggingFaceEmbeddings(
         model_name="sentence-transformers/LaBSE",     # 文をベクトル化(埋め込み)するモデルを指定する。
         model_kwargs={"device": device},              # "cuda"または"cpu"
         encode_kwargs={"normalize_embeddings": True}  # cosine類似度に適した正規化をする。
      )
```

HuggingFaceEmbeddingsは、LangChainのEmbeddingのためのクラスの一つです。
他に、OpenAIEmbeddingsやOllamaEmbeddingsというクラスもあります。

```
 from langchain_openai import OpenAIEmbeddings
 
 embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

```
 from langchain_ollama import OllamaEmbeddings
 
 embeddings = OllamaEmbeddings(model="llama3")
```

LangChainのクラスの他、先に紹介したsentence-transformersのSentenceTransformerもあります。
LangChainのクラスとsentence-transformersのクラスとでは使い方(メソッド)が違うので、注意が必要です。

```
 from sentence_transformers import SentenceTransformer
 
 model = SentenceTransformer("all-MiniLM-L6-v2")
```

それぞれどのような違いがあるかはこの記事の範疇ではありませんので、詳しくはそれぞれのドキュメントをご覧ください。

Embeddingのインスタンスを作るとき、Embedding Model(埋め込みモデル)を指定します。
インターネット上に幾つもの埋め込みモデルが公開されていますので、好きな埋め込みモデルを選んでください。
この記事では多言語に対応した"sentence-transformers/LaBSE"を使用します。

### 6.4 ベクトルストア(Vector Store)

#### 6.4.1 ベクトルストアの概要

RAGチャットボットが入力された質問に対して自社の事情を加味して応答できるようにする為に、専門的な内容や組織の規則などに関するテキストファイルやHTMLファイル、PDFファイルなどを集めて検索可能なデータベースを作成します。
このデーターベースは"vector store"(ベクトルストア)と呼ばれます。

プログラムがあるディレクトリーと同じディレクトリに"docs"という名前のディレクトリーを作成し、その中にテキストファイルやHTMLファイル、PDFファイルなどを格納しておきます。

ベクトルストアを作成する大まかな流れは次のとおりです。

1. 各種ファイルから文字列を抽出する。
2. 抽出した文字列を分割する。
3. 分割した文字列をベクトルに変換する。
4. ベクトルデータのインデックスを作成する。
5. Wrapper(ラッパー)を使って統合する。

#### 6.4.2 文字列の抽出

ファイルを格納しているディレクトリーから各種ファイルを順次読み出してファイルの種別を判別し、その種別に応じたDocument Loader(ドキュメントローダー)を使ってファイルからデータを抽出します。

* TextLoader
* PyMuPDFLoader
* CSVLoader

ドキュメントローダーにファイルパスをセットし、load()メソッドを呼び出すことでファイルからデータを抽出してDocumentオブジェクトに格納します。

Documentオブジェクトは、二つの重要な要素を持っています。

* page\_content: ファイルから抽出したテキストの文字列 |
* metadata: コンテンツに関係付けられた任意のメタデータ（例：ファイルパス、ウェブページのURLなど） |

```
 def load_documents():
     docs = []    # LangChain用の Document オブジェクトの空リスト（[Document(...), Document(...), Document(...), ...]）
     loaders = {
         "application/pdf": PyMuPDFLoader,
         "text/html": UnstructuredHTMLLoader,
         "text/plain": lambda fp: TextLoader(fp, encoding="utf-8"),
         "text/csv": CSVLoader
        }
     for filename in os.listdir("docs"):
         filepath = os.path.join("docs", filename)
         mime = magic.from_file(filepath, mime=True)
         print(f"{filename} → MIME: {mime}")
         loader_class = loaders.get(mime)
         if loader_class:
             loader = loader_class(filepath) if callable(loader_class) else loader_class
             docs.extend(loader.load())
         else:
             print(f"{filename} は未対応のファイル形式です。スキップします。")
     return docs
```

#### 6.4.3 文字列の分割

Documentオブジェクトの列"docs[]"にある文字列をルールに従って分割します。
文字列を分割することをchunking(チャンキング)、分割された文字列をchunk(チャンク)といいます。

```
 def chunk_documents(documents):
     splitter = RecursiveCharacterTextSplitter(
         chunk_size=250,    # 1つのチャンクの大きさを 250 文字とする。
         chunk_overlap=50,    # separatoesで区切られずに、chunk_sizeの上限に達したことにより分割された場合、前のチャンクと 50 文字重複する。
         separators=['。', '\n', ''],    #  まず句点'。'で分割しようとし、句点がない場合は改行'\n'で分割する。句点も改行もなければchunk_sizeで分割する。
         keep_separator="end"
         )
     return splitter.split_documents(documents)
```

関数chunk\_documents()はスプリッター"splitter"にDocumentオブジェクトのリストを渡してチャンクに分割し、呼び出し元に返します。

クラス"RecursiveCharacterTextSplitter"による分割は、文単位で分割されます。これは"sentence-based chunking"と呼ばれます。
splitter.split\_documentsは、documents(文書)からsentence(文)を切り出します。文はseparatorsで指定された文字で文で区切られます。
上に示したコードの例では、まず句点(。)で文を切り出すことを試みます。もし、句点が検出されないままchunk\_sizeに達した場合、次に改行('\n')で区切ることを試みます。句点も改行も無い場合は、chunk\_sizeで区切ります。

Documentから最初に切り出した文を1つ目のチャンクに格納します(ここではチャンクは箱のようなものと考えてください。)

次に、Documentから2つ目の文を切り出して1つ目のチャンクに追加しようとします。もし、1つ目のチャンクに十分な空き領域があれば2つ目の文を1つ目のチャンクに追加します。ここでは、1つ目のチャンクに2つ目の文を格納したとします。

さらに、Documentから3つ目の文を切り出して1つ目のチャンクに格納しようとします。もし、1つ目のチャンクに3つ目の文を格納する空き領域が無い場合は、3つ目の文は新しいチャンク(2つ目のチャンク)に格納されます。

新しく切り出した文(n番目の文)を新しいチャンク(m番目のチャンク)に格納するとき、chunk\_overlapに値がセットされているなら、新しく切り出した文を新しいチャンクに格納する前にオーバーラップの処理をします。

まず、chunk\_sizeから新しく切り出した文の長さ"len(sentence[n])"を引いた値(空き領域の大きさ)を求めます。仮にこの値を"rest\_size"とします。

``` 
 rest_size = chunk_size - len(sentence[n])
```

次に、rest\_sizeの値とchunk\_overlapの値のうち、小さい方の値を求めます。
この値ががオーバーラップに使用できる領域です。仮にこの値を"overlap\_size"とします。

``` 
 overlap_size = min([rest_size, chunk_overlap])
```

最後のチャンク(m-1番目のチャンク)に格納されている文を後ろから調べて、overlap\_sizeに収まる文を取り出します。
もし、最後の文(n-1番目の文)と最後の文の1つ前の文(n-2番目の文)の長さの合計がoverlap\_sizeより短ければ、二つの文が新しいチャンク(m番目のチャンク)にコピーされ、そのあと新たに切り出した文(n番目の文)を新しいチャンクに追加します。

#### 6.4.4 文字列のベクトル化

埋め込みモデルを使ってチャンクの文字列をベクトル(数値の列)に変換し、NumPyの配列に追加します。

```
 def create_vector_db(chunks, embeddings):
     vectors = np.array(embeddings.embed_documents([doc.page_content for doc in chunks])).astype("float32")
     print(f"基本基本ベクトル数: {vectors.shape[0]} 次元: {vectors.shape[1]}")
```

NumPyの配列に格納するデータ型を32ビット浮動小数点にしているのは、あとで、FAISSを使ってベクトル間の類似度を計算するときに都合が良いからです。

#### 6.4.5 インデックスの作成

ベクトルデータを検索するときIndex(インデックス)を使います。ベクトルデータを検索する方法はいくつかあり、それに合わせてインデックスを作成します。

* IndexFlatL2  
	すべてのベクトルを走査し、クエリとベクトルストア内にあるベクトル間の距離を計算して類似度を評価します。
* IndexFlatIP  
	すべてのベクトルを走査し、クエリとベクトルストア内にあるベクトル間の内積を計算して類似性を評価します。
* IndexIVFFlat  
	事前にすべてのベクトルをいくつかのグループ(クラスタ)に分けて、クエリに近いクラスタだけを探索します。

IndexIVFFlatは効率的に検索することができますが、データ数が少なすぎると効果は期待できません。
以下に示すコードでは、ベクトルデータの数が少ない場合はIndexFlatIPを使用し、そうでない場合はIndexIVFFlatを使用するようになっています。

```
     faiss.normalize_L2(vectors)    # コサイン類似度計算用に正規化する。
     
     d = vectors.shape[1]
     if vectors.shape[0] < 80:
         print("IndexFlatIPを使用します。")
         index = faiss.IndexFlatIP(d)
     else:
         nlist = min(100, max(1, vectors.shape[0] // 40))
         print(f"IndexIVFFlatを使用します。クラスタ数 nlist: {nlist}")
         quantizer = faiss.IndexFlatIP(d)
         index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
         index.train(vectors)    # .train() で事前にクラスタを作る（これがIVFの肝）
     index.add(vectors)
```

#### 6.4.6 ラッパーによる統合

埋め込みモデル、チャンク、ベクトルデータとインデックスをWrapper(ラッパー)を使って一つにまとめます。これがベクトルストアです。

```
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
```

docstoreの構造を以下に示します。

``` 
 {
     "0": Document(page_content="..."),
     "1": Document(page_content="..."),
     ...
 }
```

index\_to\_docstore\_idの構造を以下に示します。

``` 
 {
     0: "0",
     1: "1",
     ...
 }
```

例えば、vectorstore.similarity\_search(query)と呼ぶと、
(Step 1) 内部のindex.search(...)を呼び出して、戻り値としてqueryのベクトルと近いベクトルの番号のリストを得て、
(Step 2) 近い文書IDをindex\_to\_docstore\_idで引き、(index\_to\_docstore\_id[3] -> "3")
(Step 3) 実際のテキストをdocstoreから取り出して返してくれます。

#### 6.4.7 ベクトルストアの使用

\_\_main\_\_ の中でCUDAが利用可能であるかを調べています。もし、CUDAが利用できるなら、ベクトルストアのindexをGPUに送ります。

```
     if torch.cuda.is_available():
         device = "cuda"
     else:
         print("CUDA非対応環境です。CPUを使用します。")
         device = "cpu"
     
     embeddings = HuggingFaceEmbeddings(
         ... (省略) ...
         )
     
     if "--rebuild" in sys.argv:
           ... (省略) ...
         vectorstore = create_vector_db(chunks, embeddings)
         print("ベクトルストア作成完了！")
     else:
         vectorstore = load_vector_db(embeddings)
         print("既存のベクトルストアを読み込み完了！")
 
     if device == "cuda" and faiss.get_num_gpus():
         print("インデックスをGPUへ転送中...")
         res = faiss.StandardGpuResources()
         vectorstore.index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
```

上のコードの中では、faiss.StandardGpuResources()メソッドで、FAISSにCUDAのGPUリソース(メモリプールやストリーム)を管理させるためのオブジェクトresを作成しています。
そのあと、faiss.index\_cpu\_to\_gpu()メソッドで、インデックスをCUDAデバイス0に転送します。

vectorstore.as_retriever()メソッドを使ってベクトルストアを検索するためのインスタンス"retriever"を作成します。
retriever(レトリーバー)のinvokeメソッドは検索を実行するメソッドです。

```
 retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
 
 docs = retriever.invoke(query)
```

上のコードでは、検索の結果として質問文"query"に近い(類似度が高い)チャンクの上位5つがDocumentオブジェクトを配列の形式で返ります。
as\_retrieverメ()ソッドでパラメータ"search\_kwargs={"k": 5}"を指定しない場合、デフォルトでは上位4つのDocumentオブジェクトのリストが返ります。

retrieverを作成しているのは、\_\_main\_\_ の中ですが、invoke()メソッドで検索を実行しているのは関数answer\_with\_filtered\_docs()の中です。
関数answer\_with\_filtered\_docs()の全容は後の章でお話しします。

```
 def answer_with_filtered_docs(query, retriever, llm, embeddings, sim_threshold=0.6):
     docs = retriever.invoke(query)
```

retriever.invoke(query)を使わずに、上の「6.4.6 ラッパーによる統合」で示したvectorstore.similarity\_search(query)を使うこともできます。
retrieverを使うと、検索結果をフィルターしたり、ランキングを調整したりできます。

### 6.5 LLMの使用

検索結果と質問文を合わせてプロンプトを作成し、LLMに渡します。

```
 llm = OllamaLLM(model="hf.co/mmnga/llm-jp-3.1-1.8b-instruct4-gguf:Q8_0")    # 他のモデルの例: "gpt-oss:20b"
```

LLMのインスタンスを作成しているのは、\_\_main\_\_ の中ですが、LLMのインスタンスにクエリーを渡しているのは関数answer\_with\_filtered\_docs()の中です。

```
 context = "\n\n".join([doc.page_content for doc in docs])
 prompt = f"以下の情報に基づいて質問に答えてください。\n\n{context}\n\n質問: {query}"
 
 result = llm.invoke(prompt)
```

## 7 調整

### 7.1 ドキュメントの重複排除

ベクトルストアを準備するとき、自社のウェブサイトで閲覧できるファイルを収集することが予想されます。
このとき、Scraping(スクレイピング)するツールによっては、リンクされているファイルが重複して複製されることがあります。
収集したファイルは次のような名前で保管されているかもしれません。

```
 ~$ ls -l
 -rw-r--r-- 1 hokazono hokazono 1254127  7月 14 18:35 2025005.pdf
 ...
 -rw-r--r-- 1 hokazono hokazono 1254127  7月 14 18:35 www.domain.jp_document_2025005.pdf.html
 ...
```

データが重複されている状況は、ベクトルストアを検索したあとに結果を表示するとで発見することができます。

```
 docs = retriever.invoke(query)
 
 for i, doc in enumerate(docs, start=1):
     print(f"[{i}] 資料: {doc.metadata.get('source', '').strip()}")
     print(f"      内容: {doc.page_content[:100]} ...")
```

同じ内容が複数表示された場合は、ファイルが重複している可能性があります。確認して重複しているファイルがあればそれを除外してください。

### 7.2 類似度のフィルター

RAGでは、質問文を入力したあと、ベクトルストアから類似度の高い内容(チャンク)をいくつか取り出して、質問文と共にLLMに渡します。
このとき、質問文と関係ない内容がベクトルストアから取り出される場合があります。質問文との類似度が低くても上位5つに入っていると、その内を取り出して質問文と合わせてLLMに渡します。
試しに、以下のような社員情報のCSVファイルからベクトルストアを作成して、社員情報と関係のない文を入力してみます。  
【注意】 以下のSCVファイルに記載している名前は全て架空の人物の名前です。実在する人物とは一切関係ありません。

```
 (rag-chatbot):~$ cat docs/employee.csv
 社員番号,名前,よみがな,役職,所属(部),所属(課)
 10001,大阪太郎,おおさかたろう,部長,情報システム部,,
 10002,堺一郎,さかいいちろう,課長,情報システム部,１課
 10003,池田次郎,いけだじろう,課長,情報システム部,２課
 10004,和泉花子,いずみはなこ,主任,情報システム部,１課
 10005,守口剛,もりぐちつよし,主任,情報システム部,２課
 10006,八尾ひとみ,やおひとみ,,情報システム部,１課
 10007,枚方昇,ひらかたのぼる,,情報システム部,１課
 10008,高石健太,たかいしけんた,,情報システム部,２課
 10009,貝塚優子,かいづかゆうこ,,情報システム部,２課
 (rag-chatbot):~$ 
```

プログラムを動かして、挨拶を入力してみます。

```
 【質問】: こんにちは。
 【回答】: こんにちは！ご提供いただいた情報に基づいて、堺一郎さんに関する情報を以下のようにまとめました。
 
 社員番号: 10002  
 名前: 堺一郎  
 役職: 課長  
 所属(部): 情報システム部  
 所属(課): １課  
 
 堺一郎さんは「情報システム部１課」の課長として勤務されています。
```

単に挨拶をしただけなのに、堺一郎に関する情報が提供されました。これは検索の結果、余計な情報がLLMに渡されたからです。

では、この堺一郎に関する情報と「こんにちは」はどれぐらい似ているのでしょうか。

デフォルトでは、レトリーバーが返すDocumentオブジェクトのリストに類似度の値は含まれていません。
そこで、ベクトルストアから取り出したDocumentオブジェクトの内容と質問文との類似度を計算して表示するようにします。

```
     #ベクトルストアを検索してDocumentオブジェクトのリストを得る。
     docs = retriever.invoke(query)
     
     # 質問文のベクトルを取得する。
     query_vec = np.array(embeddings.embed_query(query)).reshape(1, -1)
 
     # Documentオブジェクトのリストから、コンテンツ(内容)を取り出してリストにする。
     doc_texts = [doc.page_content for doc in docs]
 
     # コンテンツのベクトルを取得する。結果(doc_vecs)はリストであることに注意する。
     doc_vecs = np.array(embeddings.embed_documents(doc_texts)).reshape(len(doc_texts), -1)
 
     # 質問文のベクトルとコンテンツのベクトルとのcosine類似度を計算する。
     sims = cosine_similarity(query_vec, doc_vecs)[0]
     
     print("--------------------------------------------------")
     print("【検索結果】:")
     for i, (doc, sim) in enumerate(zip(docs, sims), start=1):
         print(f"--- [{i}] ---")
         print(f"類似度: {sim:.4f}")
         print(f"資料: {doc.metadata.get('source', '').strip()}")
         print(f"内容: {doc.page_content[:100]} ...")
     print("--------------------------------------------------")
```

検索結果の類似度を表示するようにプログラムを書き換えて再度試してみます。

```
 【質問】: こんにちは。
 --------------------------------------------------
 【検索結果】:
 --- [1] ---
 類似度: 0.0334
 資料: docs_employee/employee.csv
 内容: 社員番号: 10002
 名前: 堺一郎
 よみがな: さかいいちろう
 役職: 課長
 所属(部): 情報システム部
 所属(課): １課 ...
 --- [2] ---
 類似度: 0.0328
 資料: docs_employee/employee.csv
 内容: 社員番号: 10008
 名前: 高石健太
 よみがな: たかいしけんた
 役職: 
 所属(部): 情報システム部
 所属(課): ２課 ...
 --- [3] ---
 類似度: 0.0327
 資料: docs_employee/employee.csv
 内容: 社員番号: 10004
 名前: 和泉花子
 よみがな: いずみはなこ
 役職: 主任
 所属(部): 情報システム部
 所属(課): １課 ...
 --- [4] ---
 類似度: 0.0299
 資料: docs_employee/employee.csv
 内容: 社員番号: 10005
 名前: 守口剛
 よみがな: もりぐちつよし
 役職: 主任
 所属(部): 情報システム部
 所属(課): ２課 ...
 --- [5] ---
 類似度: 0.0263
 資料: docs_employee/employee.csv
 内容: 社員番号: 10003
 名前: 池田次郎
 よみがな: いけだじろう
 役職: 課長
 所属(部): 情報システム部
 所属(課): ２課 ...
 --------------------------------------------------
 【回答】: こんにちは。ご提供いただいた情報に基づき、いくつかの関連する質問にお答えいたします。
 
 1. 社員番号10002の堺一郎さんについて:
    - 社員番号10002の社員である堺一郎さんは、「課長」という役職にあります。また、所属は「情報システム部１課」となっています。
 
 2. 社員番号1008の高石健太さんの詳細については具体的な情報がありません。ただし、彼も同じく「情報システム部２課」に所属していることが
 わかります。追加の情報が必要であれば、彼の職務内容や役割についてさらに調査いたします。
 ～（以下、省略）
```

表示された、5つの結果の類似度はいずれも0.1以下です。このような類似度が低い内容をLLMに渡さないようにするために、類似度に閾値を設けてLLMに渡す内容を絞ります。

(これは余談ですが、同じ質問をしても回答文は毎回異なります。)

レトリーバーに閾値を設定して、検索をする段階で絞り込むこともできますが、この記事では検索結果を受けたあと、改めて類似度を計算します。
参考までにレトリーバーに閾値を設定するコードを以下に示します。

```
 retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 5})
```

質問文とベクトルストアーを検索した結果とのコサイン類似度(最大1.0)を求め、その値が"0.5"未満の内容を捨ててLLMに渡すプロンプトを生成するようにプログラムを改善します。

```
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
 
     embeddings = HuggingFaceEmbeddings(
         ... (省略) ...
         )
     
     if "--rebuild" in sys.argv:
           ... (省略) ...
         vectorstore = create_vector_db(chunks, embeddings)
         print("ベクトルストア作成完了！")
     else:
         vectorstore = load_vector_db(embeddings)
         print("既存のベクトルストアを読み込み完了！")
         
     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
     llm = OllamaLLM(model="hf.co/mmnga/llm-jp-3.1-1.8b-instruct4-gguf:Q8_0")    # 他のモデルの例: "gpt-oss:20b"
     
     print("チャットを開始します。終了するには exit と入力してください。")
     
     while True:
         query = input("【質問】: ")
         if query.lower() in ["exit", "quit"]:
             break
 
         response = answer_with_filtered_docs(query, retriever, llm, embeddings, sim_threshold=0.5)
         
         print("【回答】:", response["result"])
```

検索結果のうち類似度が0.5未満の内容を排除するコードを追加したあと、もう一度試してみます。

```
 【質問】: こんにちは。
 --------------------------------------------------
 【検索結果】:
 
 ～（省略）～
 
 --------------------------------------------------
 【回答】: こんにちは！今日はどのようなお手伝いが必要ですか？
```

### 7.3 検索データの補完

先に紹介した社員情報のデータファイルを持ったRAGチャットボットに次の質問をしてみます。

```
 【質問】: 情報システム部の部長は誰ですか。
 --------------------------------------------------
 【検索結果】:
 --- [1] ---
 類似度: 0.4866
 資料: docs_employee/employee.csv
 内容: 社員番号: 10007
 名前: 枚方昇
 よみがな: ひらかたのぼる
 役職: 
 所属(部): 情報システム部
 所属(課): １課 ...
 --- [2] ---
 
 ～（省略）～
 
 --- [5] ---
 類似度: 0.4708
 資料: docs_employee/employee.csv
 内容: 社員番号: 10001
 名前: 大阪太郎
 よみがな: おおさかたろう
 役職: 部長
 所属(部): 情報システム部
 所属(課): ...
 --------------------------------------------------
 【回答】: ご質問にお答えするためには、具体的な組織名や企業名が必要となります。 私は特定の企業や団体に関する内部情報にはアクセスできない
 ため、一般的な情報システム部門の構造についてお伝えします。 通常、日本の大手企業では、取締役会のもとに執行役員として情報システム部が設置
 されています。部長はその中でも特に上位の役職であり、多くの場合、IT戦略や技術方針を決定する責任を持っています。 具体的な人物名は、各企業
 の公式ウェブサイトやニュースリリースなどで確認することをお勧めします。
```
 
大阪太郎に関する情報の類似度が0.4708であるためLLMに渡っていません。

このような現象に対して2つの対策が考えられます。一つは、類似度の閾値を調整することです。
上の例だと類似度の閾値を0.47に設定すると適切な回答が得られます。実際に類似度の閾値を下げて実行したときの結果を以下に示します。

```
 【質問】: 情報システム部の部長は誰ですか。
 --------------------------------------------------
 【検索結果】:
 
 ～（省略）～
 
 --------------------------------------------------
 【回答】: 情報システム部の部長は、大阪太郎です。
```

もう1つの方法は、類似度の閾値は変更せずに、想定される質問文に近い文をデータベースに納めておくことです。
少し極端な例ですが、以下のようなテキストファイルを用意します。

```
 (rag-chatbot):~$ cat docs/employee.txt
 情報システム部の部長は大阪太郎です。
 (rag-chatbot):~$ 
```

類似度の閾値を0.5に戻して、試してみます。

```
 【質問】: 情報システム部の部長は誰ですか。
 --------------------------------------------------
 【検索結果】:
 --- [1] ---
 類似度: 0.7168
 資料: docs_employee/employee.txt
 内容: 情報システム部の部長は大阪太郎です。 ...
 --- [2] ---
 
 ～（省略）～
 
 --- [5] ---
 類似度: 0.4753
 資料: docs_employee/employee.csv
 内容: 社員番号: 10004
 名前: 和泉花子
 よみがな: いずみはなこ
 役職: 主任
 所属(部): 情報システム部
 所属(課): １課 ...
 --------------------------------------------------
 【回答】: 情報システム部の部長は大阪太郎さんです。
```

このように想定される質問文とよく似た構造の文書を予め用意しておくと、回答の精度を上げることができます。

employee.csvのような元データがある場合は、そのファイルを表計算ソフトなどで読み込み、新しいデータファイルを作ると良いでしょう。以下に例を示します。
以下の例では、G列とH列を追加して、その内容を式で自動的に生成するようにしています。

|     |A    |B    |C    |D     |E    |F    |G    |H    |
|-----|-----|-----|-----|------|-----|-----|-----|-----|
|1    |<span style="white-space: nowrap;">社員番号</span>|<span style="white-space: nowrap;">名前</span>|<span style="white-space: nowrap;">よみがな</span>|<span style="white-space: nowrap;">役職</span>|<span style="white-space: nowrap;">所属(部)</span>|<span style="white-space: nowrap;">所属(課)</span>|<span style="white-space: nowrap;">説明１</span>|<span style="white-space: nowrap;">説明２</span>|
|2    |<span style="white-space: nowrap;">10001</span>|<span style="white-space: nowrap;">大阪太郎</span>|<span style="white-space: nowrap;">おおさかたろう</span>|<span style="white-space: nowrap;">部長</span>|<span style="white-space: nowrap;">情報システム部</span>|<span style="white-space: nowrap;"></span>|<span style="white-space: nowrap;">=IF(D2&lt;&gt;"", CONCATENATE(B2,"は",E2,F2,"の",D2,"です"), "")</span>|<span style="white-space: nowrap;">=IF(D2&lt;&gt;"", CONCATENATE(E2,F2,"の",D2,"は",B2,"です"), "")</span>|
|3    |<span style="white-space: nowrap;">10002</span>|<span style="white-space: nowrap;">堺一郎　</span>|<span style="white-space: nowrap;">さかいいちろう</span>|<span style="white-space: nowrap;">課長</span>|<span style="white-space: nowrap;">情報システム部</span>|<span style="white-space: nowrap;">１課</span>|<span style="white-space: nowrap;">=IF(D3&lt;&gt;"", CONCATENATE(B3,"は",E3,F3,"の",D3,"です"), "")</span>|<span style="white-space: nowrap;">=IF(D3&lt;&gt;"", CONCATENATE(E3,F3,"の",D3,"は",B3,"です"), "")</span>|


## 8 おわりに

上の章「7 調整」でお話ししたとおり、RAGで重要なことは質問に対する回答文の生成に役立つ的確な情報をデータベースから取り出すことです。
そのために類似度の閾値を調整したり、想定される質問によく似た文を用意したりします。

もちろん、LLMの性能も無視できません。小さなパラメータ数のLLMより大きなパラメータ数のLLMを使う方がより精度が高く適切な回答が期待できます。
しかしながら、大きなパラメータ数のLLMを使って応答時間が2秒程度の実用的なRAGチャットボットを構築しようとすると、100万円以上する高性能なGPUが必要ですので簡単に試すことは難しいでしょう。
なお、近年ではNPU(Neural Network Processing Unit)搭載のCPUが発表されています。今後は低価格のハードウェア資源でも大きなパラメータ数のLLMを動かすことができるかもしれません。

それから、RAGを導入する前にRAGが本当に業務改善に資するのかをよく検討するべきだと考えます。
例えば、社内で蓄積されている情報が体系的に整理されていて、ウェブUIで多角的に検索できる仕組みが既にあるなら、そして社員がその仕組みに満足しているなら、RAGを導入してどのような業務改善が期待できるのかを慎重に検討するべきです。
一方、情報が非常に大量に蓄積されていて、体系的に整理してインデックスを付けることが難しい場合などは、RAGによる業務の効率化が期待できます。

最後に、RAGの導入を考えている方にとってこの記事が少しでもお役に立つことを願っています。

## A 補足

### A.1 脚注・出典

##### ※1 「令和7年版情報通信白書」 (総務省)
URL https://www.soumu.go.jp/johotsusintokei/whitepaper/index.html

##### ※2 「令和7年版情報通信白書」 (総務省) データ集
URL https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r07/html/datashu.html

##### ※3 LLM-jp (国立情報学研究所 大規模言語モデル研究開発センター)
URL https://llm-jp.nii.ac.jp/

##### ※4 Debian GNU/Linux インストールガイド (devian)
URL https://www.debian.org/releases/stable/amd64/index.ja.html

##### ※5 NvidiaGraphicsDrivers (devian / Wiki /)
URL https://wiki.debian.org/NvidiaGraphicsDrivers

##### ※6 Download Ollama (Ollama)
URL https://ollama.com/download/linux

##### ※7 Installing Miniconda (ANACONDA)
URL https://www.anaconda.com/docs/getting-started/miniconda/install
