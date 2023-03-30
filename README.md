# CerebrasモデルZoo

## 導入

このリポジトリには、Cerebrasハードウェアで学習できる一般的なディープラーニングモデルの例が含まれています。これらのモデルは、Cerebrasハードウェアをターゲットとしたモデルをコーディングするためのベストプラクティスであり、この新しく強力な計算エンジンを最大限に活用できます。

CSシステム上でモデルの実行を開始するためには、READMEファイルと一緒に[開発用ドキュメント](https://dics.cerebras.net/en/latest/index.html)を参照してください。

**備考**：　Cerebras Model ZooをCerebras Hardware（CS-2システム）でお試しいただく場合、以下のオプションをご用意しています。

- 教育機関 - パートナーハードウェアアクセスのリクエストフォーム[こちら](https://www.cerebras.net/developers/partner-hardware-access-request/)にご記入ください。私たちのパートナーからシステムへのアクセスについてご連絡いたします。
- 営利目的 - 私たちのチームがデモを提供し、私たちのシステムのアクセスについて話し合いができるように、当社の[デモのご案内フォーム](https://www.cerebras.net/get-demo/)にご記入ください。
- 上記に該当しない方 - developer@cerebras.netまでご連絡ください。

対応機種一覧は、[このリポジトリにある機種](#models-in-this-repository)をご確認ください。

## サポートするフレームワーク

[PyTorch](https://pytorch.org/)と[TensorFlow](https://www.tensorflow.org/)で開発されたモデルをサポートしています。フレームワーク固有のワークフローの詳細については、以下の開発者向けドキュメントを参照してください：

- [PyTorch CS workflow](https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-cs-workflow.html)
- [TensorFlow CS workflow](https://docs.cerebras.net/en/latest/tensorflow-docs/cs-tf-workflow.html)

## Basic workflow

Cerebras CSシステムでニューラルネットワークの仕事をする場合、以下の開発者向けドキュメントにあるクイックスタートリンクに従って、このModelZooのモデルをお好みのフレームワーク用にコンパイル、検証、トレーニングしてください。

- [PyTorch Quickstart](https://docs.cerebras.net/en/latest/getting-started/pytorch/index.html)
- [TensorFlow Quickstart](https://docs.cerebras.net/en/latest/getting-started/tensorflow/index.html)

高度なユースケースや、TensorFlowやPyTorchから既存のコードをCerebras互換に移植する場合、高レベルのワークフローは次のように記述できます：

1. [サポートされているフレームワーク](#suppoeted-frameworks)のいずれかで、コードをCSに移植してください。
	- PyTorchの場合は、`cerebras.framework.torch`を使用します。
	- TensorFlowでは、`CerebrasEstimator`を使用します。
2. シャーディング、シャッフル、プリフェッチ、インターリーブ、リピート、バッチなど、入力データを適切な順序で前処理することを保証する入力データを準備する。
3. CPU上でコードをコンパイルし、早い段階で特定のCSシステムにコードを最適化する。
4. コンパイルしたコードをCSシステム上で実行する。

## 実行モード

Cerebrasウェハースケールエンジン（WSE）では、様々なモデルサイズのニューラルネットワークを実行することができます。Cerebras Softwareは、このような多様なモデルを効率的に実行するために、様々な実行モードをサポートしています。

実行モードとは、CerebrasランタイムがニューラルネットワークモデルをCerebrasウェハースケールエンジン（WSE）にロードする方法を指します。2つの実行モードがサポートされています。：

- **Layer pupeline（レイヤーパイプライン）**：このモードでは、ネットワークの全てのレイヤーがCerebrasWSEに一括してロードされます。このモードは、少～中規模のモデル（パラメータが10億個以下）のニューラルネットワークモデルに選択されます。

- **Weight Streaming（ワイヤーストリーミング）**：このモードでは、ニューラルネットワークモデルの1つのレイヤーが一度にロードされます。このレイヤーごとのモードは、非常に大きなモデル（パラメータが数十億～数兆個規模）を実行するために使用されます。

これについては、開発者ページの[Cerebras実行モード](https://docs.cerebras.net/en/latest/cerebras-basics/cerebras-execution-modes.html#cerebras-execution-modes)の項目で詳しく解説しています。

## Cerebrasハードウェアのための最適化

Cerebras社のハードウェアの特性を生かし、トレーニングの高速化を実現する様々な機能を提供しています。以下は、その主な機能です：

- [CerebrasモデルZoo](#cerebrasモデルzoo)
  - [導入](#導入)
  - [サポートするフレームワーク](#サポートするフレームワーク)
  - [Basic workflow](#basic-workflow)
  - [実行モード](#実行モード)
  - [Cerebrasハードウェアのための最適化](#cerebrasハードウェアのための最適化)
    - [PyTorch Variable Tensor Shape(VTS)](#pytorch-variable-tensor-shapevts)
    - [TensorFlow Variable Sequence Length(VSL)](#tensorflow-variable-sequence-lengthvsl)
    - [Mult-replice mode](#mult-replice-mode)
  - [リポジトリ内のモデル](#リポジトリ内のモデル)
  - [ライセンス](#ライセンス)

一般的な最適化技術については、[Performance Best Practices Page](https://docs.cerebras.net/en/latest/general/performance-optimization.html)をご参照ください。

### PyTorch Variable Tensor Shape(VTS)

Variable Tensor Shape(VTS)は、パイプラインモードで動作するCSシステム上の計算で、バッチの要素ごとに形状が異なるテンソルを処理できるようにする機能です。これにより、配列の長さが異なる入力データに対応することができ、小さな配列で大きなパディングサンプルを取り除くことができるようになります。これにより、無駄な計算が減り、モデルの学習時間が改善されます。

VTSの詳細については、同じトピックの開発者向けドキュメントページ[こちら](https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-vts.html)を参照してください。

### TensorFlow Variable Sequence Length(VSL)

VTSと同じ概念で、VSTはTensorFlowのフレームワークでサポートしているレガシーネームです。VSLは汎用性に限界があり、現在のVTSに置き換えられつつある。
VSLについての詳細は、同じトピックの開発者向けドキュメントページ[こちら](https://docs.cerebras.net/en/latest/tensorflow-docs/tf-vsl.html)を参照してください。

### Mult-replice mode

マルチレプリカデータ並列トレーニングは、Cerebrasコンパイラが、同じモデルのコピー（レプリカ）を複数作成し、データ並列トレーニングを実行するための機能である。これは、複数のGPUを使用して1つのモデルの学習を高速化する方法と似ています。

バックグラウンドでは、コンパイラはこれらのレプリカが同じ重みで初期化されることを保証し、トレーニング中は、すべてのレプリカの重みがバッチごとに同期される。

マルチレプリカデータ並列トレーニングの終了時には、1つのトレーニング済みモデルが利用可能になります。このマルチレプリカデータ並列機能は、モデルのトレーニングにのみ使用することができます。

multi-replicaの詳細については、同じトピックの開発者向けドキュメントページ[こちら](https://docs.cerebras.net/en/latest/general/multi-replica-data-parallel-training.html)を参照してください。


## リポジトリ内のモデル

| Model                                       | Layer Pipeline mode                                                                                                                                                               | Weight Streaming mode                                                                                        |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| BERT                                        | [TensorFlow code](./modelzoo/transformers/tf/bert/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/)                                                                      | -                                                                                                            |
| BERT (fine-tuning) Classifier               | [TensorFlow code](./modelzoo/transformers/tf/bert/fine_tuning/classifier/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/fine_tuning/classifier/)                        | -                                                                                                            |
| BERT (fine-tuning) Named Entity Recognition | [TensorFlow code](./modelzoo/transformers/tf/bert/fine_tuning/token_classifier/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/fine_tuning/token_classifier/)            | -                                                                                                            |
| BERT (fine-tuning) Summarization            | [TensorFlow code](./modelzoo/transformers/tf/bert/fine_tuning/extractive_summarization/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/fine_tuning/extractive_summarization/) | -                                                                                                            |
| BERT (fine-tuning) Question Answering       | [TensorFlow code](./modelzoo/transformers/tf/bert/fine_tuning/qa/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/fine_tuning/qa/)                                        | -                                                                                                            |
| GPT-2                                       | [TensorFlow code](./modelzoo/transformers/tf/gpt2/)<br>[PyTorch code](./modelzoo/transformers/pytorch/gpt2/)                                                                      | [TensorFlow code](./modelzoo/transformers/tf/gpt2/)<br>[PyTorch code](./modelzoo/transformers/pytorch/gpt2/) |
| GPT-3                                       | -                                                                                                                                                                                 | [TensorFlow code](./modelzoo/transformers/tf/gpt3/)<br>[PyTorch code](./modelzoo/transformers/pytorch/gpt3/) |
| GPT-J                                       | -                                                                                                                                                                                 | [TensorFlow code](./modelzoo/transformers/tf/gptj/) <br>[PyTorch code](./modelzoo/transformers/pytorch/gptj/) |
| GPT-NeoX                                    | -                                                                                                                                                                                 | [TensorFlow code](./modelzoo/transformers/tf/gptj/) <br>[PyTorch code](./modelzoo/transformers/pytorch/gptj/) |
| GPT-J (fine-tuning) Summarization           | -                                                                                                                                                                                 | [TensorFlow code](./modelzoo/transformers/tf/gptj/fine_tuning/abstractive_summarization/)                    |
| Linformer                                   | [TensorFlow code](./modelzoo/transformers/tf/linformer/)                                                                                                                          | -                                                                                                            |
| RoBERTa                                     | [TensorFlow code](./modelzoo/transformers/tf/bert/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/)                                                                      | -                                                                                                            |
| T5                                          | [TensorFlow code](./modelzoo/transformers/tf/t5/)<br>[PyTorch code](./modelzoo/transformers/pytorch/t5/)                                                                          | -                                                                                                            |
| Transformer                                 | [TensorFlow code](./modelzoo/transformers/tf/transformer/)<br>[PyTorch code](./modelzoo/transformers/pytorch/transformer/)                                                        | -                                                                                                            |
| MNIST (fully connected)                     | [TensorFlow code](./modelzoo/fc_mnist/tf/)<br>[PyTorch code](./modelzoo/fc_mnist/pytorch/)                                                                                        | -                                                                                                            |
| 2D UNet (experimental in TensorFlow)        | [TensorFlow code](./modelzoo/unet/tf/)                                                                                           | [PyTorch code](./modelzoo/vision/pytorch/unet/)                                                                                                               |

## ライセンス

[Apache License 2.0](./LICENSE)