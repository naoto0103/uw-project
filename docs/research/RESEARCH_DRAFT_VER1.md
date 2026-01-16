# Hierarchical Action Models with 2D Paths and Consistency Flow Training for General Robot Manipulation

（2Dパス表現と一貫性フロー学習を用いた階層型生成モデルによる汎用的なロボット操作）

**著者**: 太田尚都

**指導教員**: 田中文英, 宇津呂武仁, 山口友之

---

## Abstract

Hierarchical vision-language-action (VLA) models that separate high-level semantic planning from low-level motion generation have emerged as a promising approach for robot manipulation. This paper integrates ManiFlow, a Consistency Flow Matching-based policy enabling high-quality action generation in 1-2 inference steps, as the low-level policy guided by 2D end-effector paths from a vision-language model (VLM) fine-tuned on robot path data. Through systematic experiments in RoboTwin 2.0 across 3 tasks and 6 conditions, we reveal three key findings. First, models trained on clean environments consistently outperform those trained on cluttered environments when evaluated on cluttered scenes (average success rate 20.3-28.0% vs 9.3-10.3%), suggesting that simplified training conditions facilitate learning essential motion patterns. Second, VLM path guidance improves performance in cross-domain settings (+7.7% average), indicating that semantic guidance can bridge domain gaps. Third, the Memory Function combining initial and current paths unexpectedly degrades performance, revealing that additional input information requires careful design. These results highlight both the potential and limitations of VLM-guided path conditioning for generalizable manipulation policies.

**Keywords**: Robot Manipulation, Hierarchical Model, Consistency Flow Matching, Path Guidance, Generalization

---

## 1. はじめに

汎用的なロボット操作ポリシーの開発は、ロボット工学における長年の課題である。近年、大規模な視覚言語モデル（vision-language model; VLM）の発展により、これらのモデルが持つ豊富な世界知識をロボット制御に活用する試みが活発化している[1, 2, 25, 26]。VLMを直接ファインチューニングしてロボットアクションを予測するvision-language-action（VLA）モデルは、タスク記述と画像観測からエンドツーエンドで制御信号を生成できるため、オープンワールド環境での汎化が期待されている[2, 27, 28]。

しかし、このようなモノリシックなVLAモデルには課題が存在する。第一に、高品質なロボット操作データの収集には多大なコストがかかり、データの規模・多様性が限定される[3, 29, 30]。第二に、VLAモデルは動作速度および精密な動作制御において、古典的なロボット制御のレベルには達していない。一方、小規模ポリシーモデルはモノリシックなVLAモデルに比べて精密な制御が可能だが、学習環境と異なる条件への汎化が困難である[2, 4]。これらの課題に対し、VLMによる高レベルの意味理解と、小規模ポリシーによる精密な動作生成を分離する階層的アプローチが提案されている[3, 5, 11]。

階層的アプローチの一つであるHAMSTERは、ロボット経路データでファインチューニングされたVLM（VILA-1.5-13B）を用いて2次元のエンドエフェクタ経路を生成し、これを低レベルポリシー（RVT-2や3D Diffuser Actor）へのガイダンスとして与える[3]。この設計により、VLMはセマンティックな軌道計画に専念し、低レベルポリシーは3次元空間での精密な制御に集中できる。HAMSTERは、シミュレーションデータや異なるエンボディメントのデータなど、アクションラベルを含まないオフドメインデータを活用できる点でも優れている。

一方、ロボット操作のための生成モデルとして、拡散モデル（Diffusion Model）に基づくポリシーが高い性能を示している[6, 13, 31]。しかし、拡散モデルは推論時に多数のデノイジングステップを必要とし、リアルタイム制御への適用に課題がある。ManiFlowは、Consistency Flow Matchingを採用することで、わずか1〜2ステップで高品質なアクションを生成可能な手法である[7]。ManiFlowはDiT-X（Diffusion Transformer with Cross-Attention）アーキテクチャを用いて、視覚・言語・固有受容感覚などの多様な入力を効率的に統合する。

本研究では、HAMSTERの階層的アーキテクチャにおける低レベルポリシーとしてManiFlowをベースとした機構を採用し、VLMが生成した2次元パスをManiFlowの入力に統合する手法を提案する。従来のManiFlowは学習環境と同一の条件で評価されてきたが、本研究では、VILAによるパスガイダンスを導入することで、（1）同一環境における単一タスク精度の向上、および（2）学習環境と異なる評価環境への汎化性能の向上を検証する。

具体的には、パスの入力方法として2つのパターンを提案する。第一のパターンでは、現在フレームに対するパス予測のみを使用する。第二のパターンでは、エピソード開始時のパス予測を追加で保持し、現在パスと併用する。後者を我々はMemory Functionと呼ぶ。この設計は、オクルージョンなどにより現在フレームでのパス生成が不安定な場合でも、過去の安定したパス情報を参照可能にすることを意図している。

評価実験は、RoboTwin 2.0シミュレーション環境[8]上で実施する。学習環境（clean table条件またはcluttered table条件）と評価環境（cluttered table条件）の組み合わせにより6つの実験条件を設定し、パスガイダンスが同一環境での精度およびクロスドメイン条件での汎化性能に与える効果を定量的に分析する。

## 2. 関連研究

### 2.1 階層的ロボット操作ポリシー

ロボット操作における階層的アプローチは、高レベルの計画と低レベルの制御を分離することで、複雑なタスクの解決を目指す。SayCanは、大規模言語モデル（LLM）の持つ知識とロボットのアフォーダンス関数を組み合わせ、実行可能かつ意味的に適切な行動計画を生成する手法を提案した[5]。Code as Policiesは、LLMを用いてロボット制御のためのプログラムコードを生成し、複雑な空間推論を実現する階層的コード生成アプローチを示した[9]。Inner Monologueは、LLMに環境からのフィードバックを与えることで閉ループ推論を実現し、成功検出やシーン記述を統合した[32]。

VLMを用いた階層的VLAモデルにおいて、高レベルモデルが生成する中間表現の選択は重要な設計決定である。キーポイントベースのアフォーダンス予測[10]は、操作対象の位置や配置場所を点として予測する手法であり、VLMのファインチューニングにより実現される。RT-Trajectoryは、2次元軌道スケッチを中間表現として用いることで、柔軟なタスク指定とポリシーの汎化を実現した[11]。低レベルポリシーとしては、ボクセル化した3D観測を用いるPerceiver-Actor[33]や、仮想視点からのマルチビュー画像を用いるRVT[34]、RVT-2[35]などの3D対応ポリシーが高い精度を示している。

HAMSTERは、VLMが2次元のエンドエフェクタ経路を予測し、これを低レベルの3D対応ポリシーへのガイダンスとして使用する階層的アーキテクチャを提案した[3]。この設計の利点は、高レベルVLMがアクションラベルを含まないオフドメインデータ（シミュレーションデータ、異なるエンボディメントのデータなど）で学習可能な点にある。HAMSTERでは、経路情報を画像上にオーバーレイする方法と、別次元の入力として連結する方法が比較され、後者がより高い性能を示した。本研究では、パスを画像上にオーバーレイする方法を採用し、ManiFlowの入力として使用する。

### 2.2 拡散モデルとフローマッチングに基づくロボット制御

拡散モデルは、ノイズからデータへの段階的な変換を学習する生成モデルであり、画像生成分野で大きな成功を収めた[12, 36]。Diffusion Policyは、この拡散モデルをロボットのビジュオモータポリシーに適用し、マルチモーダルな行動分布の表現や高次元行動空間への対応において優れた性能を示した[6]。3D Diffusion Policyは、スパースな点群から抽出した3次元表現を用いることで、視覚的汎化性能をさらに向上させた[13]。3D Diffuser Actorは、3Dシーン表現とDiffusion Policyを組み合わせ、RLBenchベンチマークにおいて高い性能を達成した[31]。しかし、拡散モデルは推論時に多数のデノイジングステップ（典型的には10〜100ステップ）を必要とするため、リアルタイム制御への適用には課題がある。

この推論効率の問題に対し、複数のアプローチが提案されている。DDIMは、DDPMの決定的サンプリング版として、より少ないステップで高品質な生成を可能にした[37]。Consistency Modelsは、拡散過程における任意の点から直接データへのマッピングを学習することで、1ステップでの生成を可能にした[14]。ロボット制御への応用として、Consistency Policyは事前学習済みのDiffusion Policyから知識蒸留を行い、推論を高速化する手法を提案した[19]。ManiCMは、3D Diffusion PolicyにConsistency Modelを適用し、リアルタイムでの3次元操作を実現した[20]。

一方、Flow Matchingは拡散モデルとは異なるアプローチで生成モデルを構築する。Flow Matchingは、ノイズ分布からデータ分布への連続的な変換（フロー）を、シミュレーションフリーで直接学習する手法である[15]。Rectified Flowは、このフローを直線的にすることで、より効率的なサンプリングを実現した[21]。これらの手法は、拡散モデルと比較して学習が安定し、少ないステップ数で高品質な生成が可能である。π₀は、VLAモデルにFlow Matchingを組み合わせた手法であり、単腕・双腕・モバイルマニピュレータなど多様なロボット構成での汎用制御を実現した[22]。

ManiFlowは、Flow MatchingとConsistency Trainingを統合したConsistency Flow Matchingを採用している[7]。具体的には、標準的なFlow Matching損失に加えて、フロー軌道上の異なる点が同一のターゲットデータ点に収束するよう制約する連続時間Consistency損失を同時に最適化する。この設計により、ManiFlowは事前学習済みの教師モデルを必要とせず、1〜2ステップでの高品質なアクション生成を実現する。ManiFlowのアーキテクチャの中核であるDiT-Xは、Diffusion Transformer[16]を基盤とし、クロスアテンションとAdaLN-Zero条件付けにより、視覚・言語・固有受容感覚などの多様な入力モダリティを効率的に統合する。本研究では、このManiFlowを低レベルポリシーとして採用し、VLMが生成したパスを追加の視覚入力として統合する。

### 2.3 ドメイン適応と汎化

ロボット操作ポリシーの実環境への展開において、学習環境と評価環境の間のドメインギャップは重要な課題である。Domain Randomizationは、シミュレーション環境でのレンダリングパラメータをランダム化することで、実環境への転移を促進する手法として広く用いられている[17]。Colosseumは、20タスク×14軸の環境摂動によりポリシーの汎化を評価するベンチマークであり、色・テクスチャ・サイズ・照明・背景などの変化に対するロバスト性を測定する[38]。

視覚表現の事前学習も汎化性能向上の有効なアプローチである。R3Mは、大規模な人間の動画データを用いて視覚表現を事前学習し、時間対照学習と動画言語アライメントにより、ロボット操作タスクへの転移性能を向上させた[18]。CLIPは、4億の画像-テキストペアで対照学習を行い、ゼロショットでの転移能力を実現した[39]。DINOv2は、1.42億画像で自己教師あり学習を行い、ファインチューニングなしで汎用的な視覚特徴を提供する[40]。

RoboTwin 2.0は、731オブジェクト、50タスク、5種類のロボットエンボディメントを含む大規模ベンチマークであり、Clutter、Lighting、Backgroundなど5軸のDomain Randomizationを実装している[8]。RLBench[41]やMetaWorld[42]などの標準ベンチマークと比較して、より現実的な視覚的変動を含む評価が可能である。本研究では、このRoboTwin 2.0環境を用いて、clean table条件で学習したポリシーがcluttered table条件でどの程度汎化できるかを評価する。VLMによるパスガイダンスが、このようなドメインギャップを橋渡しする効果を持つかどうかが、本研究の主要な検証項目の一つである。

## 3. 提案手法

本章では、VLMによる2次元パスガイダンスをManiFlowに統合した提案システムについて述べる。

### 3.1 システム概要

提案システムは、高レベルのパス生成モジュールと低レベルのアクション生成モジュールから構成される階層的アーキテクチャを採用する。図1に提案システムの全体構成を示す。

【図1: システム全体図 - 左側にVILA-13B（高レベル）、右側にManiFlow（低レベル）を配置。入力として上部にRGB画像とタスク指示、VILAからは2Dパスが出力され、それがRGB画像にオーバーレイされてManiFlowに入力される。ManiFlowからは16ステップ分のアクションが出力される。initial overlayとcurrent overlayの2つの経路も図示。】

高レベルモジュールでは、VILA-1.5-13B[23]を使用する。VILAは、RGB画像とタスク記述を入力として受け取り、エンドエフェクタが辿るべき2次元経路を出力する。出力される経路は、正規化された2次元座標列（各座標は[0, 1]の範囲）とグリッパー状態（開/閉）のシーケンスとして表現される。VILAは、HAMSTERプロジェクト[3]においてロボット操作データでファインチューニングされたモデルを使用する。

低レベルモジュールでは、ManiFlow[7]を使用する。オリジナルのManiFlowは、過去2ステップ分のRGB画像とロボットの固有受容感覚状態（関節角度およびグリッパー状態）を入力として受け取り、16ステップ分のアクションを一括で生成する。本研究では、この入力画像にVILAが生成した2次元パスをオーバーレイすることで、高レベルの軌道ガイダンスを低レベルポリシーに伝達する。

提案システムでは、パスの入力方法として2つのパターンを提供する。

**パターン1: Current pathのみ**

現在フレームに対してVILAが生成したパスのみを使用する。具体的には、ManiFlowへの入力となる過去2ステップ分の観測画像それぞれに現在のパスをオーバーレイし、ロボット状態とともにManiFlowに与える。

**パターン2: Initial path + Current path（Memory Function）**

エピソード開始時に生成したパス（initial path）と、現在フレームに対して生成したパス（current path）の両方を使用する。この設計は、タスク実行中にオクルージョンなどにより現在フレームでのパス生成が不安定になった場合でも、初期の安定したパス情報を参照可能にすることを意図している。我々はこの機構をMemory Functionと呼ぶ。

推論時のフローは以下の通りである。ManiFlowは16ステップ分のアクションを一度に予測するため、パス生成は毎フレームではなく16ステップごとに行われる。

1. **Step 0**: VILAによるパス生成を実行し、initial overlayおよびcurrent overlayを作成する。これらをManiFlowに入力し、16ステップ分のアクションを一括予測する。
2. **Step 1-15**: 予測済みのアクションを順次実行する。
3. **Step 16**: 新たにVILAによるパス生成を実行し、current overlayを更新する。initial overlayはエピソード開始時のまま固定される。次の16ステップ分のアクションを一括予測する。
4. 以降、タスク完了まで16ステップごとに繰り返す。

この設計により、VILAによるパス生成のオーバーヘッドは16ステップに1回のみとなり、リアルタイム制御への影響を最小限に抑えることができる。

### 3.2 高レベルパス生成

高レベルモジュールでは、HAMSTERプロジェクト[3]においてロボット操作データでファインチューニングされたVILA-1.5-13B[23]を使用する。VILAは、SigLIP視覚エンコーダとVicuna言語モデルを組み合わせたvision-language modelであり、53億の画像テキストペアで事前学習されている。

#### 入力と出力

VILAへの入力は、RGB画像と自然言語によるタスク指示である。タスク指示は、例えば「there is a hammer and a block on the table, use the arm to grab the hammer and beat the block」や「click the bell's top center on the table」のような形式で与えられる。

VILAの出力は、以下の形式で構造化される：

```
<ans>[(x1, y1), (x2, y2), <action>Close Gripper</action>, (x3, y3), ...]</ans>
```

ここで、各座標(x, y)は[0, 1]の範囲に正規化された2次元位置を表し、`<action>`タグはグリッパー状態の変化点を示す。この出力をパースすることで、各ウェイポイントに対応するグリッパー状態（開/閉）を含む経路表現を得る。

#### パスの視覚化

生成されたパスは、元のRGB画像上にオーバーレイとして描画される。描画仕様はHAMSTER[3]のオリジナル実装に準拠する：

- **経路の色**: jetカラーマップを使用し、時間的進行を青→シアン→緑→黄→赤の色変化でエンコードする
- **線の太さ**: 画像サイズに応じてスケーリング（512×512画像を基準）
- **グリッパーマーカー**: グリッパー状態の変化点にのみ円形マーカーを描画
  - 開状態: 青色の円（BGR: 255, 0, 0）
  - 閉状態: 赤色の円（BGR: 0, 0, 255）

図2にパスオーバーレイの例を示す。

【図2: パスオーバーレイの例 - 3つのタスク（click_bell, move_can_pot, beat_block_hammer）について、clean条件とcluttered条件それぞれのオーバーレイ画像を並べて表示。jetカラーマップによる経路の色変化と、グリッパー状態変化点のマーカーが確認できる例。】

#### パス生成の信頼性

VILAはtemperature=0で設定されるため出力は決定的であるが、一部のフレームでは指定フォーマットに従わない出力が生成される場合がある。この問題に対処するため、以下の戦略を採用する：

1. **リトライ機構**: 各フレームに対して最大2回のパス生成を試行する
2. **フォールバックパース**: `<ans>`タグが省略された場合や、座標の括弧が欠落した場合でも、座標データとして解釈可能な出力に対して自動補完を行う
3. **フォールバックパス**: パス生成に失敗した場合、直前に成功したパスを代用する。エピソード冒頭で連続失敗した場合は、そのエピソード内で最初に成功したパスを遡及的に適用する

この設計により、学習時と評価時で同一のフォールバック戦略が適用され、入力分布の一貫性が保たれる。

### 3.3 低レベルアクション生成

低レベルモジュールでは、Consistency Flow Matchingを用いたロボット操作ポリシーであるManiFlow[7]を使用する。本節では、ManiFlowのアーキテクチャと、本研究におけるパス入力の統合方法について述べる。

#### ManiFlowの概要

ManiFlowは、Flow Matching[15]とConsistency Training[14]を統合したConsistency Flow Matchingにより、1〜2ステップでの高品質なアクション生成を実現する。従来のDiffusion Policy[6]が推論時に10〜100ステップのデノイジングを必要とするのに対し、ManiFlowは大幅に少ないステップ数で同等以上の品質を達成する。

ManiFlowの学習では、標準的なFlow Matching損失とConsistency損失を同時に最適化する。Flow Matching損失は、ノイズ点からデータ点への速度場を学習する：

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{x_0, x_1}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$$

ここで、$x_t = (1-t)x_0 + tx_1$はノイズ$x_0$とデータ$x_1$の線形補間である。Consistency損失は、フロー軌道上の異なる点が同一のターゲットに収束するよう制約する：

$$\mathcal{L}_{CT}(\theta) = \mathbb{E}_{t, \Delta t}[\|v_\theta(x_t, t, \Delta t) - \tilde{v}_{target}\|^2]$$

これらの損失を組み合わせることで、事前学習済みの教師モデルを必要とせずに、少ステップでの高品質生成を実現する。

#### DiT-Xアーキテクチャ

ManiFlowのアーキテクチャの中核は、DiT-X（Diffusion Transformer with Cross-Attention）である。DiT-Xは、Diffusion Transformer[16]を基盤とし、以下の特徴を持つ：

- **適応的クロスアテンション**: アクショントークンと観測特徴量（視覚・言語）の間で、トークンレベルでの細粒度な相互作用を実現する
- **AdaLN-Zero条件付け**: タイムステップやロボット状態などの低次元入力に基づいて、スケール・シフトパラメータを動的に生成し、ネットワークの挙動を適応的に調整する

これらの機構により、視覚入力、言語入力、固有受容感覚入力を効率的に統合するマルチモーダル処理を実現する。

#### 本研究におけるパス入力の統合

オリジナルのManiFlowは、過去2ステップ分のRGB画像とロボットの固有受容感覚状態（14次元：各アーム7自由度の関節位置およびグリッパー状態）を入力として受け取り、16ステップ分のアクションを一括で生成する。

本研究では、入力RGB画像をパスオーバーレイ画像に置き換えることで、VILAが生成した高レベルのガイダンスを統合する。具体的な入力構成を表1に示す。

**表1: 各条件における入力構成**

| 条件 | 入力画像 | 固有受容感覚 | 出力 |
|------|---------|-------------|------|
| オリジナルManiFlow | RGB画像（過去2ステップ） | 14次元 | 16ステップ分のアクション |
| Current pathのみ | current overlay（過去2ステップ） | 14次元 | 16ステップ分のアクション |
| Initial + Current | initial overlay + current overlay | 14次元 | 16ステップ分のアクション |

パスオーバーレイ画像は、元のRGB画像にVILAが生成した2次元経路を視覚的に描画したものである。ManiFlowは、オーバーレイされた経路情報を視覚特徴として学習し、タスク実行に活用することが期待される。

### 3.4 Memory Function

本研究では、パス入力の第二のパターンとして、エピソード開始時のパス（initial path）と現在フレームのパス（current path）を併用する方式を提案する。我々はこの機構をMemory Functionと呼ぶ。

タスク実行中、ロボットのエンドエフェクタや把持した物体が視野を遮ることで、オクルージョンが発生する場合がある。このような状況では、VILAが生成するパスの品質が低下し、不正確または不完全な経路が出力される可能性がある。エピソード開始時（frame 0）では、通常オクルージョンが発生しておらず、タスク対象物体が明確に視認できる状態にある。したがって、この時点で生成されるパスは、タスク全体の軌道計画として信頼性が高いと考えられる。Memory Functionは、この初期パスを保持し続けることで、タスク実行中に現在パスの品質が低下した場合でも、タスク全体の文脈情報を参照可能にする。

Memory Functionを使用する場合、ManiFlowへの入力は以下のように構成される：

- **Initial overlay**: エピソード開始時（frame 0）のRGB画像に、その時点で生成されたパスをオーバーレイした画像。エピソード終了まで固定される。
- **Current overlay**: 現在フレームのRGB画像に、そのフレームで生成されたパスをオーバーレイした画像。16ステップごとに更新される。
- **固有受容感覚状態**: 現在のロボット状態（14次元）

図3にMemory Functionの概念図を示す。

【図3: Memory Functionの概念図 - 時間軸に沿って、frame 0でinitial overlayが生成・固定される様子と、frame 16, 32, ...でcurrent overlayが更新される様子を図示。オクルージョン発生時にcurrent pathが不安定になっても、initial pathが参照可能であることを視覚的に表現。】

この機構により、オクルージョン耐性の向上、タスク文脈の保持、および長期的な計画情報の活用が期待される。Current pathのみを使用する場合（パターン1）と比較すると、入力画像が1枚増加するためやや計算コストが高くなるが、初期状態からの全体計画を保持できる点が利点である。実験では、これら2つのパターンを比較し、Memory Functionの効果を定量的に検証する。

## 4. 実験

本章では、提案手法の有効性を検証するための実験設定と結果について述べる。

### 4.1 実験設定

実験は、RoboTwin 2.0シミュレーション環境[8]上で実施した。RoboTwin 2.0は、SAPIEN 3.0.0b1物理エンジン[24]を基盤とし、AgileX Cobot Magicロボットを用いた双腕操作タスクのベンチマークを提供する。本研究ではシングルアームタスクに焦点を当て、テーブル上の環境設定として提供されるclean table条件とcluttered table条件を用いた。cluttered table条件では、タスクに直接関係しない物体がテーブル上にランダムに配置され、視覚的な複雑さが増加する。

評価タスクとして、RoboTwin 2.0のシングルアームタスクの中から、難易度と動作パターンの多様性を考慮して3タスクを選択した。click_bellは単純な接触動作、move_can_potは標準的なPick-and-Place、beat_block_hammerは道具使用を要する2段階動作である。各タスクについて、RoboTwin 2.0のスクリプト化されたエキスパートポリシーを用いて50本の成功軌道を収集し、学習データとした。

本研究では、3つのモデル構成（オリジナルManiFlow、VILA + ManiFlow (current path)、VILA + ManiFlow (initial + current path)）と2つの学習環境（cluttered、clean）の組み合わせにより、6つの実験条件（C1〜C6）を設定した。表2に各条件の詳細を示す。全ての条件において、評価はcluttered table条件で実施した。条件C1〜C3はcluttered環境で学習・評価を行うin-domain設定であり、条件C4〜C6はclean環境で学習しcluttered環境で評価を行うcross-domain設定である。

**表2: 実験条件の一覧**

| 条件 | 学習環境 | 評価環境 | パス入力 |
|------|----------|----------|----------|
| C1 | Cluttered | Cluttered | なし（オリジナルManiFlow） |
| C2 | Cluttered | Cluttered | Current path |
| C3 | Cluttered | Cluttered | Initial + Current path |
| C4 | Clean | Cluttered | なし（オリジナルManiFlow） |
| C5 | Clean | Cluttered | Current path |
| C6 | Clean | Cluttered | Initial + Current path |

この設計により、以下の4つの仮説を検証する：**H1**（VILAパスガイダンスはin-domain精度を向上させる）、**H2**（VILAパスガイダンスはcross-domain汎化を向上させる）、**H3**（Memory FunctionはCurrent pathのみより優れる）、**H4**（パスガイダンスの効果はcross-domain条件でより顕著である）。

評価指標として、各タスク100回の試行に基づくタスク成功率を主要指標とし、RoboTwin 2.0の内蔵判定エンジンを用いて成功判定を行った。補助指標として、VILAによるパス生成時間、ManiFlowによるアクション生成時間、およびパス生成成功率を測定した。ManiFlowの学習設定は全条件で統一し、観測ステップ数2、アクション予測ステップ数16、学習エポック数501とした。

### 4.2 Results and Analysis

#### 4.2.1 Effect of Input Representation Mode

Figure 4に条件別成功率、Table 3に詳細な数値を示す。本節では、パス入力モードの効果に関する仮説H1〜H4を検証する。

【図挿入: Figure 4 - 条件別成功率のグラフ - `/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/analysis/outputs/figures/main_results_all.png`を使用】

**表3: タスク成功率 (%)**

| Task | C1 | C2 | C3 | C4 | C5 | C6 |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Click Bell | 4.0 | 2.0 | 4.0 | 3.0 | **20.0** | 1.0 |
| Move Can to Pot | 21.0 | 20.0 | 26.0 | 27.0 | 27.0 | 27.0 |
| Beat Block with Hammer | 4.0 | 6.0 | 1.0 | 31.0 | **37.0** | 11.0 |
| **Average** | 9.7 | 9.3 | 10.3 | 20.3 | **28.0** | 13.0 |

**H1: In-domain条件でのパスガイダンス効果**

C1（ベースライン）とC2, C3（パスあり）を比較した。C1→C2は平均-0.3%、C1→C3は平均+0.7%であり、VILAパスガイダンスによる一貫した改善は見られなかった。この結果は、in-domain条件ではパスガイダンスが追加的な情報を提供していない可能性を示唆している。また、パスオーバーレイによる視覚的な変化が、cluttered学習環境ではむしろ視覚的ノイズを増加させ、ManiFlowの学習を困難にした可能性も考えられる。

**H2: Cross-domain条件でのパスガイダンス効果**

C4（ベースライン）とC5, C6（パスあり）を比較した。C4→C5は平均+7.7%の改善を示し、3タスク中2タスク（Click Bell: +17.0%、Beat Block with Hammer: +6.0%）で改善が確認された。一方、C4→C6は平均-7.3%と逆に性能が低下した。cross-domain条件でパスガイダンスが部分的に効果を発揮した理由として、学習環境（clean）と評価環境（cluttered）の視覚的な差異をVILAのパスが補完したことが考えられる。VILAは大規模なデータで事前学習されているため、cluttered環境でも適切なパスを生成でき、ManiFlowに対してタスク遂行に必要な軌道情報を提供できたと推測される。

**H3: Memory Function（Initial + Current path）の効果**

C2 vs C3、C5 vs C6を比較した。In-domain条件（C2→C3）では平均+1.0%のわずかな改善が見られたが、cross-domain条件（C5→C6）では平均-15.0%という大幅な性能低下が観察された。この対照的な結果は、initial pathの効果がドメイン条件に強く依存することを示している。In-domain条件では、initial pathがタスク全体の文脈情報として若干機能し、わずかながら性能向上に寄与した可能性がある。一方、cross-domain条件では、2つのオーバーレイ画像を入力することで処理すべき情報量が増加し、clean環境で学習したモデルにとって負担となった可能性がある。また、エピソードが進むにつれてinitial path（frame 0時点）の情報が現在の状況と乖離し、誤った誘導を与えた可能性も考えられる。この結果は、Memory Functionの有効性が学習環境と評価環境の組み合わせに依存することを示唆している。

**H4: パスガイダンス効果のドメイン依存性**

(C2-C1) vs (C5-C4)、(C3-C1) vs (C6-C4)を比較した。Current pathモードでは、in-domain改善幅（-0.3%）に対してcross-domain改善幅（+7.7%）が大きく、3タスク中2タスクでこの傾向が確認された。一方、initial+currentモードでは逆のパターンを示し、in-domain改善幅（+0.7%）に対してcross-domain改善幅（-7.3%）と大幅に低下した。この対照的な結果は、パス入力モードによってドメインギャップへの適応特性が異なることを示している。Current pathモードはcross-domain条件でVILAのセマンティックなガイダンスがドメインギャップを橋渡しする効果を発揮する一方、initial+currentモードはin-domain条件でのみ有効であり、ドメインギャップがある場合は情報の不整合により逆効果となる。

#### 4.2.2 Effect of Training Environment

Table 3の結果から、仮説として設定していなかった予想外の発見が得られた。clean環境で学習した条件（C4-C6、平均13.0〜28.0%）がcluttered環境で学習した条件（C1-C3、平均9.3〜10.3%）より一貫して高い成功率を示したことである。特にBeat Block with Hammerタスクでは、C4（31.0%）およびC5（37.0%）がC1-C3（1.0〜6.0%）を大幅に上回った。

この結果は、cluttered環境で学習したモデルが学習データ特有の散乱パターンにオーバーフィットし、評価時の異なる散乱パターンに対応できなかった可能性を示唆している。一方、clean環境で学習したモデルは、視覚的ノイズが少ない環境でタスクの本質的な動作パターンを効率的に学習でき、結果として未知のcluttered環境への汎化性能が向上したと考えられる。

#### 4.2.3 Task-Specific Analysis

タスク間で結果のパターンが異なることが観察された。Click Bellでは、C5で突出した成功率（20.0%）が見られた一方、他の条件では1.0〜4.0%と低迷した。Move Can to Potでは、条件間の差が比較的小さく（20.0〜27.0%）、パスガイダンスの効果が限定的であった。Beat Block with Hammerでは、C4（31.0%）およびC5（37.0%）で高い成功率を示したが、それ以外の条件では1.0〜11.0%と低迷した。

これらの結果は、VILAパスガイダンスの効果がタスクの特性に依存することを示唆している。精密な操作や明確な軌道計画を必要とするタスク（Click Bell、Beat Block with Hammer）では、適切に生成されたパスが動作の精度向上に寄与する。一方、Move Can to Potのような標準的なPick-and-Placeタスクでは、動作パターンが比較的単純であるため、パスガイダンスによる追加的な改善が限定的であったと考えられる。

#### 4.2.4 Training Dynamics

全18条件（3タスク×6条件）について501エポックの学習を実施し、学習曲線を分析した（Figure 5）。Table 4に学習統計を示す。

【図挿入: Figure 5 - 学習曲線詳細 - `/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/analysis/outputs/figures/training_curves_by_task.png`を使用】

**Table 4: Training Statistics**

| Task | Condition | Initial Loss | Final Loss | Min Loss |
|:-----|:---------:|:------------:|:----------:|:--------:|
| Click Bell | C1-C3 | 2.76-2.77 | 0.003-0.003 | 0.003-0.003 |
| | C4-C6 | 2.76-2.77 | 0.003-0.003 | 0.002-0.003 |
| Move Can to Pot | C1-C3 | 2.33-2.34 | 0.002-0.002 | 0.002-0.002 |
| | C4-C6 | 2.33-2.34 | 0.002-0.003 | 0.002-0.002 |
| Beat Block with Hammer | C1-C3 | 2.51-2.53 | 0.001-0.004 | 0.0003-0.001 |
| | C4-C6 | 2.25-2.28 | 0.002-0.002 | 0.001-0.001 |

全条件で損失削減率99.9%を達成し、学習は正常に収束した。タスク間で初期損失値に差異が見られ、Click Bell（約2.76）> Beat Block with Hammer（約2.25-2.53）> Move Can to Pot（約2.33）の順であった。Beat Block with Hammerでは、clean環境条件の初期損失がcluttered環境条件より約10%低く、これはclean環境でのデータがより学習しやすい表現を持つことを示唆している。

#### 4.2.5 Computational Cost

**Table 5: Inference Time (ms)**

| Condition | VILA | ManiFlow | Total Overhead |
|:---------:|:----:|:--------:|:--------------:|
| C1, C4 | - | 108-136 | - |
| C2, C5 | 4313-4703 | 118-136 | +4.3-4.7s |
| C3, C6 | 4361-4363 | 118-126 | +4.4s |

VILAによるパス生成には約4.3〜4.7秒のオーバーヘッドが発生する。ManiFlow単体の推論時間は約100〜140msと高速であるため、16ステップごとにVILA推論を行う現在の設計では、エピソード全体で約2倍の実行時間となる。cross-domain条件でのC5の改善（+7.7%）を考慮すると、汎化性能が重要な応用シナリオではこのオーバーヘッドは許容可能と考えられる。一方、リアルタイム性が求められる応用では、より軽量なVLM[参考文献が必要: 軽量VLMに関する文献、例えばNVILA等]やパス生成頻度の最適化が必要である。

### 4.3 Limitations

本研究にはいくつかの限界が存在する。

第一に、本研究の評価はRoboTwin 2.0シミュレーション環境のみで実施されており、実ロボットでの検証は行っていない。シミュレーションと実環境の間にはsim-to-realギャップが存在し、視覚的外観、物理特性、センサノイズなどの差異が性能に影響を与える可能性がある。VILAパスガイダンスがこのsim-to-realギャップに対してどの程度ロバストであるかは、今後の実ロボット実験により検証する必要がある。

第二に、評価に使用したタスクは3種類（click_bell、move_can_pot、beat_block_hammer）に限定されている。これらのタスクは難易度と動作パターンの多様性を考慮して選択したが、より複雑な長期タスク、両腕協調タスク、あるいは接触リッチな操作タスクにおいてパスガイダンスがどのような効果を示すかは未検証である。特に、パスガイダンスの効果がタスク特性に依存することが示唆されたことから、より広範なタスクセットでの評価が必要である。

第三に、Memory Function（initial + current path）が期待に反して性能低下を引き起こした点について、本研究ではその原因を十分に解明できていない。当初の設計意図は、オクルージョン発生時に初期パスを参照することでロバスト性を向上させることであった。しかし、本実験で使用したタスクにおいてオクルージョンがどの程度発生していたか、またそれがパス生成品質にどの程度影響を与えていたかは定量的に評価していない。今後は、オクルージョンの発生頻度とその影響を定量的に分析した上で、オクルージョン検出に基づく適応的なパス参照機構など、より洗練されたMemory機構の設計が必要である。

## 5. 結論

本研究では、HAMSTERの階層的アーキテクチャにおける低レベルポリシーとしてManiFlowを採用し、VLMが生成した2次元パスをManiFlowの入力に統合する手法を提案した。RoboTwin 2.0シミュレーション環境において、3タスク×6条件の体系的な実験を通じて、パスガイダンスが単一タスク精度および汎化性能に与える効果を検証した。

実験の結果、以下の知見が得られた。第一に、clean環境で学習したモデルがcluttered環境で学習したモデルより一貫して高い汎化性能を示した（平均成功率20.3〜28.0% vs 9.3〜10.3%）。これは、視覚的にシンプルな環境での学習がタスクの本質的な動作パターンの獲得を促進し、結果として未知環境への汎化を向上させることを示唆している。第二に、VILAパスガイダンスはcross-domain条件（clean学習→cluttered評価）において効果を発揮し、current pathモードで平均+7.7%の改善を達成した。これは、VLMのセマンティックなガイダンスがドメインギャップを橋渡しする可能性を示している。第三に、Memory Function（initial + current path）は当初の期待に反して性能低下を引き起こし、入力表現の設計において追加情報が必ずしも有益ではないことが明らかになった。

本研究は、VLMによる高レベルの軌道計画とConsistency Flow Matchingによる高速なアクション生成を組み合わせた階層的アプローチの可能性と課題を明らかにした。今後は、実ロボットでの検証、より広範なタスクセットでの評価、およびオクルージョンに適応したMemory機構の設計を通じて、提案手法の実用性をさらに検証していく必要がある。

## 謝辞

This work was supported by Cross-Pacific AI Initiative (X-PAI).

## 参考文献

[1] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich, "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," in *Proc. Conference on Robot Learning (CoRL)*, 2023.

[2] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn, "OpenVLA: An Open-Source Vision-Language-Action Model," in *Proc. Conference on Robot Learning (CoRL)*, 2024.

[3] Y. Li, Y. Deng, J. Zhang, J. Jang, M. Memmel, R. Yu, C. Garrett, F. Ramos, D. Fox, A. Li, A. Gupta, and A. Goyal, "HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation," *arXiv preprint arXiv:2502.05485*, 2025.

[4] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich, "RT-1: Robotics Transformer for Real-World Control at Scale," *arXiv preprint arXiv:2212.06817*, 2022.

[5] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, A. Herzog, D. Ho, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, E. Jang, R. Ruano, K. Jeffrey, S. Jesmonth, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, K. Lee, S. Levine, Y. Lu, L. Luu, C. Parada, P. Pastor, J. Quiambao, K. Rao, J. Rettinghouse, D. Reber, C. Samiento, N. Siebers, C. Tan, A. Toshev, V. Vanhoucke, F. Xia, T. Xiao, P. Xu, S. Xu, M. Yan, and A. Zeng, "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances," *arXiv preprint arXiv:2204.01691*, 2022.

[6] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," in *Proc. Robotics: Science and Systems (RSS)*, 2023.

[7] G. Yan, J. Zhu, Y. Deng, S. Yang, R. Qiu, X. Cheng, M. Memmel, R. Krishna, A. Goyal, X. Wang, and D. Fox, "ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training," *arXiv preprint arXiv:2509.01819*, 2025.

[8] T. Chen, J. Wang, Y. Mu, Z. Liu, Y. Yuan, Y. Zhang, C. Wang, R. Qiu, P. Lu, and H. Dong, "RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation," *arXiv preprint arXiv:2506.18088*, 2025.

[9] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng, "Code as Policies: Language Model Programs for Embodied Control," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, 2023.

[10] W. Yuan, T. Ren, M. Memmel, D. Fox, A. Gupta, and A. Goyal, "RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics," in *Proc. Conference on Robot Learning (CoRL)*, 2024.

[11] H. Gu, Y. Su, Y. Liu, S. Jiang, K. Pertsch, J. Luo, A. Mandlekar, D. Xu, L. Fei-Fei, and J. Wu, "RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches," *arXiv preprint arXiv:2311.01977*, 2023.

[12] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[13] Y. Ze, G. Zhang, K. Zhang, C. Hu, J. Wang, and H. Dong, "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations," in *Proc. Robotics: Science and Systems (RSS)*, 2024.

[14] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, "Consistency Models," in *Proc. International Conference on Machine Learning (ICML)*, 2023.

[15] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, "Flow Matching for Generative Modeling," in *Proc. International Conference on Learning Representations (ICLR)*, 2023.

[16] W. Peebles and S. Xie, "Scalable Diffusion Models with Transformers," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023.

[17] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," in *Proc. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2017.

[18] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, "R3M: A Universal Visual Representation for Robot Manipulation," in *Proc. Conference on Robot Learning (CoRL)*, 2022.

[19] P. Prasad, R. Hoque, S. Tung, I. Pinto, K. Kawaguchi, and Y. Shkurti, "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation," in *Proc. Robotics: Science and Systems (RSS)*, 2024.

[20] Y. Lu, H. Chen, Y. Huang, Z. Chen, X. Cheng, and H. Wang, "ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation," *arXiv preprint arXiv:2406.01586*, 2024.

[21] X. Liu, C. Gong, and Q. Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow," in *Proc. International Conference on Learning Representations (ICLR)*, 2023.

[22] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakber, T. Kelestemur, S. Levine, A. Lii, I. Liang, H. Luo, S. Nair, K. Pertsch, L. Qi, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, K. Tran, T. Vuong, F. Xia, Z. Xu, T. Xiao, H. Xu, M. Xu, and S. Yeola, "π₀: A Vision-Language-Action Flow Model for General Robot Control," *arXiv preprint arXiv:2410.24164*, 2024.

[23] J. Lin, H. Yin, W. Ping, P. Molchanov, M. Shoeybi, and S. Han, "VILA: On Pre-training for Visual Language Models," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.

[24] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su, "SAPIEN: A SimulAted Part-based Interactive ENvironment," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

[25] D. Driess, F. Xia, M. S. M. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, W. Huang, Y. Chebotar, P. Sermanet, D. Duckworth, S. Levine, V. Vanhoucke, K. Hausman, M. Tober, G. Welker, P. Wohlhart, J. Wu, and P. R. Florence, "PaLM-E: An Embodied Multimodal Language Model," in *Proc. International Conference on Machine Learning (ICML)*, 2023.

[26] D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, Y. L. Tan, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine, "Octo: An Open-Source Generalist Robot Policy," in *Proc. Robotics: Science and Systems (RSS)*, 2024.

[27] Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, 2024.

[28] X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu, H. Li, and P. Luo, "Vision-Language Foundation Models as Effective Robot Imitators," *arXiv preprint arXiv:2311.01378*, 2023.

[29] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, P. David, K. Black, A. Kumar, C. Xu, Q. Vuong, and C. Finn, "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset," in *Proc. Robotics: Science and Systems (RSS)*, 2024.

[30] H. Walke, K. Black, A. Lee, M. J. Kim, M. Du, C. Zheng, T. Zhao, P. Hansen-Estruch, Q. Vuong, A. He, V. Myers, K. Fang, C. Finn, and S. Levine, "BridgeData V2: A Dataset for Robot Learning at Scale," in *Proc. Conference on Robot Learning (CoRL)*, 2023.

[31] T.-W. Ke, N. Gkanatsios, and K. Fragkiadaki, "3D Diffuser Actor: Policy Diffusion with 3D Scene Representations," in *Proc. Conference on Robot Learning (CoRL)*, 2024.

[32] W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, P. Sermanet, N. Brown, T. Jackson, L. Luu, S. Levine, K. Hausman, and B. Ichter, "Inner Monologue: Embodied Reasoning through Planning with Language Models," in *Proc. Conference on Robot Learning (CoRL)*, 2022.

[33] M. Shridhar, L. Manuelli, and D. Fox, "Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation," in *Proc. Conference on Robot Learning (CoRL)*, 2022.

[34] A. Goyal, J. Xu, Y. Guo, V. Blukis, Y.-W. Chao, and D. Fox, "RVT: Robotic View Transformer for 3D Object Manipulation," in *Proc. Conference on Robot Learning (CoRL)*, 2023.

[35] A. Goyal, V. Blukis, J. Xu, Y. Guo, Y.-W. Chao, and D. Fox, "RVT-2: Learning Precise Manipulation from Few Demonstrations," in *Proc. Robotics: Science and Systems (RSS)*, 2024.

[36] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, "Score-Based Generative Modeling through Stochastic Differential Equations," in *Proc. International Conference on Learning Representations (ICLR)*, 2021.

[37] J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," in *Proc. International Conference on Learning Representations (ICLR)*, 2021.

[38] A. Pumacay, I. Singh, J. Duan, F. Xia, J. Thomason, and D. Fox, "THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation," in *Proc. Robotics: Science and Systems (RSS)*, 2024.

[39] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. International Conference on Machine Learning (ICML)*, 2021.

[40] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jégou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, "DINOv2: Learning Robust Visual Features without Supervision," *arXiv preprint arXiv:2304.07193*, 2023.

[41] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, "RLBench: The Robot Learning Benchmark," *IEEE Robotics and Automation Letters*, vol. 5, no. 2, pp. 3019-3026, 2020.

[42] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning," in *Proc. Conference on Robot Learning (CoRL)*, 2020.
