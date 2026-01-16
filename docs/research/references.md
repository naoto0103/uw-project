# 参考文献

**最終更新**: 2025-12-26

---

## 参考文献一覧

### 1. HAMSTER関連

| ID | タイトル | 著者 | 会議/年 | カテゴリ | 関連度 | 内容 | 本研究との関係 | リンク |
|----|---------|------|---------|---------|--------|------|---------------|--------|
| [HAMSTER25] | HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation | Li et al. | arXiv 2025 | VLA, Hierarchical | ★★★ | VLM(VILA)で2Dパスを生成し、低レベルポリシー(RVT-2等)で精密制御を行う階層的VLAモデル。パスのConcat入力がOverlayより優位(成功率0.83→1.00)であることを実証 | 本研究のベース手法。VLM部分(VILA)を流用し、低レベルポリシーをManiFlowに置換 | [arXiv](https://arxiv.org/abs/2502.05485) |
| [VILA23] | VILA: On Pre-training for Visual Language Models | Lin et al. | CVPR 2024 | VLM | ★★★ | SigLIP視覚エンコーダとVicuna言語モデルを組み合わせたVLM。interleaved事前学習データの重要性、LLM unfreezeの必要性を実証 | HAMSTERのVLM基盤。本研究でパス生成に使用するVILA-1.5-13Bのベース | [arXiv](https://arxiv.org/abs/2312.07533) |
| [NVILA24] | NVILA: Efficient Frontier Visual Language Models | NVlabs | arXiv 2024 | VLM | ★★ | VILAを基盤とし、高解像度画像・長時間動画を効率的に処理するための「scale-then-compress」アプローチを提案 | VILAファミリーの発展形。効率化手法の参考 | [arXiv](https://arxiv.org/abs/2412.04468) |
| [RVT23] | RVT: Robotic View Transformer for 3D Object Manipulation | Goyal et al. | CoRL 2023 | 3D Manipulation | ★★ | 仮想視点からのマルチビュー画像を用いた3D操作ポリシー。PerActを大幅に上回る性能を達成 | HAMSTERの低レベルポリシーの一つ。本研究の比較対象 | [arXiv](https://arxiv.org/abs/2306.14896) |
| [RVT-2_24] | RVT-2: Learning Precise Manipulation from Few Demonstrations | Goyal et al. | RSS 2024 | 3D Manipulation | ★★ | RVTの改良版。coarse-to-fine戦略とカスタムレンダラーにより6倍高速化、19%性能向上 | HAMSTERの低レベルポリシーの一つ。本研究の比較対象 | [arXiv](https://arxiv.org/abs/2406.08545) |
| [3DDA24] | 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations | Ke et al. | CoRL 2024 | Diffusion Policy, 3D | ★★ | 3Dシーン表現とDiffusion Policyを組み合わせた手法。RLBenchでSOTA達成(+18.1%) | HAMSTERの低レベルポリシーの一つ。本研究の比較対象 | [arXiv](https://arxiv.org/abs/2402.10885) |
| [RT-Traj23] | RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches | Gu et al. | arXiv 2023 | Hierarchical, Trajectory | ★★ | 2D軌道スケッチを中間表現として用いることでロボットタスクの汎化を実現。Hindsight trajectory生成によりタスク指定 | HAMSTERの2Dパス表現の先行研究。軌道スケッチによるタスク指定の参考 | [arXiv](https://arxiv.org/abs/2311.01977) |
| [RoboPoint24] | RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics | Yuan et al. | CoRL 2024 | VLM, Affordance | ★★ | VLMを用いた空間アフォーダンス予測。自動合成データ生成パイプラインでInstruction-tuning。GPT-4oを21.8%上回る | HAMSTERのVLM学習データセットの基盤。Pixel Point Predictionデータに使用 | [arXiv](https://arxiv.org/abs/2406.10721) |

### 2. ManiFlow関連

| ID | タイトル | 著者 | 会議/年 | カテゴリ | 関連度 | 内容 | 本研究との関係 | リンク |
|----|---------|------|---------|---------|--------|------|---------------|--------|
| [ManiFlow25] | ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training | Ge et al. | arXiv 2025 | Flow Matching, Imitation Learning | ★★★ | Consistency Flow Trainingにより1-2ステップで高品質なアクション生成を実現。DiT-Xアーキテクチャ、AdaLN-Zero条件付けを採用 | 本研究の低レベルポリシー基盤。パス入力を追加して拡張 | [arXiv](https://arxiv.org/abs/2509.01819) |
| [CM23] | Consistency Models | Song et al. | ICML 2023 | Generative Models | ★★ | ノイズからデータへの直接マッピングを学習し、1ステップ生成を可能にする新しい生成モデルファミリー。蒸留と直接学習の両方に対応 | ManiFlowのConsistency Flow Trainingの理論的基盤 | [arXiv](https://arxiv.org/abs/2303.01469) |
| [ICM23] | Improved Techniques for Training Consistency Models | Song & Dhariwal | arXiv 2023 | Generative Models | ★★ | Consistency Modelsの学習技術を改善。蒸留なしでの直接学習性能を向上 | ManiFlowの学習手法の参考 | [arXiv](https://arxiv.org/abs/2310.14189) |
| [FM23] | Flow Matching for Generative Modeling | Lipman et al. | ICLR 2023 | Generative Models | ★★ | CNFをシミュレーションフリーで学習するFlow Matchingを提案。Optimal Transport経路により効率的な学習・サンプリングを実現 | ManiFlowのFlow Matching部分の理論的基盤 | [arXiv](https://arxiv.org/abs/2210.02747) |
| [DiT23] | Scalable Diffusion Models with Transformers | Peebles & Xie | ICCV 2023 | Diffusion, Transformer | ★★ | U-NetをTransformerに置換したDiffusion Transformer (DiT)を提案。スケーラビリティを実証しImageNetでSOTA達成 | ManiFlowのDiT-Xアーキテクチャの基盤 | [arXiv](https://arxiv.org/abs/2212.09748) |
| [RectFlow22] | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | Liu et al. | ICLR 2023 | Generative Models | ★★ | 直線的な軌道を学習するRectified Flowを提案。潜在空間での効率的な学習・転移を実現 | ManiFlowで比較対象として使用されるFlow Matching手法 | [arXiv](https://arxiv.org/abs/2209.03003) |
| [ConsFM24] | Consistency Flow Matching: Defining Straight Flows with Velocity Consistency | Yang et al. | arXiv 2024 | Generative Models | ★★ | 速度一貫性によるstraight flowの定義。Consistency ModelとFlow Matchingの融合 | ManiFlowの比較対象。Consistency Flow Trainingの関連手法 | [arXiv](https://arxiv.org/abs/2407.02398) |
| [Shortcut25] | One Step Diffusion via Shortcut Models | Frans et al. | ICLR 2025 | Generative Models | ★ | ステップサイズ条件付けと自己一貫性による1ステップ生成。ManiFlowの比較対象 | ManiFlow論文での比較対象手法 | [arXiv](https://arxiv.org/abs/2410.12557) |
| [MDT24] | Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals | Reuss et al. | RSS 2024 | Diffusion, Transformer | ★★ | マルチモーダルゴールから多様な行動を学習するDiffusion Transformer | ManiFlowのDiT-Xアーキテクチャ設計の比較対象 | N/A |
| [ConsPolicy24] | Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation | Prasad et al. | RSS 2024 | Diffusion Policy | ★★ | Consistency蒸留によるDiffusion Policyの高速化。ManiFlow論文で引用 | Consistency学習のロボット応用の参考 | [arXiv](https://arxiv.org/abs/2405.07503) |
| [ManiCM24] | ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation | Lu et al. | arXiv 2024 | Diffusion Policy, Consistency | ★★ | Consistency Modelによる3D Diffusion Policyのリアルタイム化 | ManiFlow論文で引用。Consistency学習の応用 | [arXiv](https://arxiv.org/abs/2406.01586) |

### 3. RoboTwin関連

| ID | タイトル | 著者 | 会議/年 | カテゴリ | 関連度 | 内容 | 本研究との関係 | リンク |
|----|---------|------|---------|---------|--------|------|---------------|--------|
| [RoboTwin24] | RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins | Mu et al. | ECCV Workshop 2024 (Best Paper), CVPR 2025 (Highlight) | Benchmark, Simulation | ★★★ | 3D生成モデルとLLMを用いてデジタルツインを生成し、双腕ロボットのベンチマークを提供。Real-to-Simパイプラインを確立 | 本研究の評価環境の基盤(RoboTwin 1.0) | [arXiv](https://arxiv.org/abs/2409.02920) |
| [RoboTwin2.0_25] | RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation | Chen et al. | arXiv 2025 | Benchmark, Domain Randomization | ★★★ | 731オブジェクト、50タスク、5ロボット embodimentsを含む大規模ベンチマーク。Clutter/Lighting/Background等5軸のDomain Randomizationを実装 | 本研究の主要評価環境。Clean/Cluttered条件での汎化実験に使用 | [arXiv](https://arxiv.org/abs/2506.18088) |
| [SAPIEN20] | SAPIEN: A SimulAted Part-based Interactive ENvironment | Xiang et al. | CVPR 2020 | Simulation | ★★ | PhysXベースの物理シミュレータ。PartNet-Mobilityデータセットによる関節物体の大規模シミュレーション環境を提供 | RoboTwin 2.0の基盤シミュレータ(SAPIEN 3.0.0b1) | [arXiv](https://arxiv.org/abs/2003.08515) |
| [Colosseum24] | THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation | Pumacay et al. | RSS 2024 | Benchmark | ★★ | 20タスク×14軸の環境摂動でポリシー汎化を評価。色・テクスチャ・サイズ・照明・背景等を変化 | HAMSTERの評価に使用。汎化評価手法の参考 | [arXiv](https://arxiv.org/abs/2402.08191) |
| [RLBench20] | RLBench: The Robot Learning Benchmark & Learning Environment | James et al. | IEEE RA-L 2020 | Benchmark | ★★ | 100タスクのロボット学習ベンチマーク。CoppeliaSim上に構築。複数の観測モードをサポート | HAMSTER/ManiFlowの評価に使用される標準ベンチマーク | [arXiv](https://arxiv.org/abs/1909.12271) |
| [MetaWorld20] | Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning | Yu et al. | CoRL 2020 | Benchmark | ★★ | 50タスクのマルチタスク/メタ強化学習ベンチマーク。MT1/MT10/MT50評価モード | ManiFlowの言語条件付きマルチタスク評価に使用 | [arXiv](https://arxiv.org/abs/1910.10897) |
| [DexArt23] | DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects | Bao et al. | CVPR 2023 | Benchmark, Dexterous | ★★ | 関節物体のDexterous操作ベンチマーク。3D表現学習による汎化を評価 | ManiFlowの評価ベンチマークの一つ | [arXiv](https://arxiv.org/abs/2305.05706) |
| [Adroit18] | Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations | Rajeswaran et al. | RSS 2018 | Benchmark, Dexterous | ★★ | Shadow Handによるdexterous操作ベンチマーク(Door, Hammer, Pen等)。デモからの学習手法を提案 | ManiFlowの評価ベンチマークの一つ | [arXiv](https://arxiv.org/abs/1709.10087) |

### 4. VLA (Vision-Language-Action)

| ID | タイトル | 著者 | 会議/年 | カテゴリ | 関連度 | 内容 | 本研究との関係 | リンク |
|----|---------|------|---------|---------|--------|------|---------------|--------|
| [RT-1_22] | RT-1: Robotics Transformer for Real-World Control at Scale | Brohan et al. | arXiv 2022 | VLA | ★★ | 130kエピソード・700+タスクで学習したRobotics Transformer。FiLM-conditioned EfficientNet + TokenLearner + Transformerアーキテクチャ。97%成功率達成 | VLAの先駆的研究。RT-2, OpenVLAの基盤 | [arXiv](https://arxiv.org/abs/2212.06817) |
| [RT-2_23] | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | Brohan et al. | CoRL 2023 | VLA | ★★ | PaLI-X/PaLM-EをベースにロボットアクションをテキストトークンとしてCo-fine-tuning。Webから学習した知識をロボット制御に転移（32%→62%） | VLAの代表的研究。End-to-end VLAアプローチの確立 | [arXiv](https://arxiv.org/abs/2307.15818) |
| [OpenVLA24] | OpenVLA: An Open-Source Vision-Language-Action Model | Kim et al. | CoRL 2024 | VLA | ★★ | 7BパラメータのオープンソースVLA。Llama 2 + DINOv2/SigLIP。970k実世界デモで学習。RT-2-X(55B)を16.5%上回る性能 | オープンソースVLAの代表。ファインチューニング手法の参考 | [arXiv](https://arxiv.org/abs/2406.09246) |
| [Octo24] | Octo: An Open-Source Generalist Robot Policy | Ghosh et al. | RSS 2024 | VLA | ★★ | Open X-Embodimentの800k軌道で学習した汎用ロボットポリシー。言語/ゴール画像指示に対応。9つのロボットプラットフォームで検証 | 汎用ロボットポリシーの参考。データスケーリングの知見 | [arXiv](https://arxiv.org/abs/2405.12213) |
| [pi0_24] | π₀: A Vision-Language-Action Flow Model for General Robot Control | Black et al. | arXiv 2024 | VLA, Flow Matching | ★★ | PaliGemma(3B)ベースのVLA + Flow Matching。単腕・双腕・モバイルマニピュレータで検証。洗濯物畳み等の複雑タスクを達成 | VLA + Flow Matchingの組み合わせの参考 | [arXiv](https://arxiv.org/abs/2410.24164) |
| [GR-2_24] | GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation | ByteDance | arXiv 2024 | VLA, Video Generation | ★ | 大規模インターネット動画で事前学習し、動画生成とアクション予測を統合。強力な汎化能力を実現 | Video-Language-Actionアプローチの参考 | [arXiv](https://arxiv.org/abs/2410.06158) |
| [GR-1_23] | Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation | Wu et al. | arXiv 2023 | VLA, Video Generation | ★ | 大規模動画生成事前学習によるロボット操作。ManiFlow論文で引用 | Video生成ベースのVLAの参考 | [arXiv](https://arxiv.org/abs/2312.13139) |
| [RoboFlamingo23] | Vision-Language Foundation Models as Effective Robot Imitators | Li et al. | arXiv 2023 | VLA | ★ | OpenFlamingoベースのVLA。事前学習済みVLMを凍結し、ポリシーヘッドのみ学習。CALVINベンチマークでSOTA | VLM凍結+ポリシーヘッド学習アプローチの参考 | [arXiv](https://arxiv.org/abs/2311.01378) |
| [OpenX23] | Open X-Embodiment: Robotic Learning Datasets and RT-X Models | Open X Collaboration | ICRA 2024 | Dataset, VLA | ★★ | 22ロボット・21機関からの60データセット統合。527スキル・160,266タスク。RT-Xモデルで正の転移を実証 | 大規模ロボットデータセット。データスケーリングの知見 | [arXiv](https://arxiv.org/abs/2310.08864) |
| [DROID24] | DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset | Khazatsky et al. | RSS 2024 | Dataset | ★★ | 大規模野外ロボット操作データセット。HAMSTERのVLM学習に使用 | HAMSTERのVLM学習データの一部 | [arXiv](https://arxiv.org/abs/2403.12945) |
| [Bridge23] | BridgeData V2: A Dataset for Robot Learning at Scale | Walke et al. | CoRL 2023 | Dataset | ★★ | 大規模ロボット学習データセット。HAMSTERのVLM学習に使用 | HAMSTERのVLM学習データの一部 | [arXiv](https://arxiv.org/abs/2308.12952) |

### 5. Diffusion Policy

| ID | タイトル | 著者 | 会議/年 | カテゴリ | 関連度 | 内容 | 本研究との関係 | リンク |
|----|---------|------|---------|---------|--------|------|---------------|--------|
| [DP23] | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | Chi et al. | RSS 2023 | Diffusion Policy | ★★ | 条件付きDenoising Diffusionでロボットポリシーを表現。マルチモーダル行動分布の扱い、高次元行動空間への適用。12タスクで平均46.9%改善 | Diffusion-basedポリシーの代表的研究。ManiFlowとの比較対象 | [arXiv](https://arxiv.org/abs/2303.04137) |
| [DP3_24] | 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations | Ze et al. | RSS 2024 | Diffusion Policy, 3D | ★★ | スパース点群からコンパクトな3D表現を抽出しDiffusion Policyに入力。72シミュレーションタスクで24.2%相対改善、実機85%成功率 | 3D入力のDiffusion Policy。ManiFlowの3D版との比較 | [arXiv](https://arxiv.org/abs/2403.03954) |
| [DDPM20] | Denoising Diffusion Probabilistic Models | Ho et al. | NeurIPS 2020 | Generative Models | ★ | 拡散確率モデルによる高品質画像生成。CIFAR-10でFID 3.17達成 | Diffusion Policyの理論的基盤 | [arXiv](https://arxiv.org/abs/2006.11239) |
| [DDIM21] | Denoising Diffusion Implicit Models | Song et al. | ICLR 2021 | Generative Models | ★ | DDPMの決定的サンプリング版。より少ないステップで高品質生成 | ManiFlow論文での比較対象 | [arXiv](https://arxiv.org/abs/2010.02502) |
| [ScoreSDE21] | Score-Based Generative Modeling through Stochastic Differential Equations | Song et al. | ICLR 2021 (Outstanding Paper) | Generative Models | ★ | SDEによるスコアベース生成モデルの統一的枠組み。CIFAR-10でFID 2.20達成 | Diffusion/Flow Matchingの理論的基盤 | [arXiv](https://arxiv.org/abs/2011.13456) |
| [ACT23] | Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | Zhao et al. | RSS 2023 | Imitation Learning | ★★ | Action Chunking with Transformers (ACT)。CVAEベースでアクションシーケンスを生成。ALOHAハードウェアと組み合わせ80-90%成功率 | アクションチャンキングの参考。双腕操作の比較対象 | [arXiv](https://arxiv.org/abs/2304.13705) |
| [UMI24] | Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots | Chi et al. | RSS 2024 | Data Collection | ★ | 実環境でのロボット教示なしにデータ収集可能なインターフェース | ManiFlow論文で引用。データ収集手法の参考 | [arXiv](https://arxiv.org/abs/2402.10329) |

### 6. 階層的ロボット制御

| ID | タイトル | 著者 | 会議/年 | カテゴリ | 関連度 | 内容 | 本研究との関係 | リンク |
|----|---------|------|---------|---------|--------|------|---------------|--------|
| [SayCan22] | Do As I Can, Not As I Say: Grounding Language in Robotic Affordances | Ahn et al. | arXiv 2022 | Hierarchical, LLM | ★★ | LLMの知識(Say)とアフォーダンス関数(Can)を組み合わせた階層的制御。PaLM-SayCanで84%正解率、74%実行成功率 | 階層的VLA制御の先駆的研究。高レベル計画の参考 | [arXiv](https://arxiv.org/abs/2204.01691) |
| [CaP22] | Code as Policies: Language Model Programs for Embodied Control | Liang et al. | ICRA 2023 | Hierarchical, LLM | ★★ | LLMでロボットポリシーコードを生成。階層的コード生成により複雑な空間推論を実現 | LLMによるコード生成アプローチの参考。HAMSTERの比較対象 | [arXiv](https://arxiv.org/abs/2209.07753) |
| [PaLM-E23] | PaLM-E: An Embodied Multimodal Language Model | Driess et al. | ICML 2023 | VLM, Hierarchical | ★ | 562Bパラメータの具身化マルチモーダルLLM。連続的なセンサー入力を言語埋め込み空間に注入 | 大規模具身化LLMの参考 | [arXiv](https://arxiv.org/abs/2303.03378) |
| [VoxPoser23] | VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models | Huang et al. | CoRL 2023 | Hierarchical, LLM | ★ | LLM+VLMで3Dアフォーダンス/制約マップを合成。ゼロショットで軌道生成 | 3D価値マップによる制御の参考 | [arXiv](https://arxiv.org/abs/2307.05973) |
| [InnerMonologue22] | Inner Monologue: Embodied Reasoning through Planning with Language Models | Huang et al. | CoRL 2022 | Hierarchical, LLM | ★ | LLMに環境フィードバックを与えることで閉ループ推論を実現。成功検出、シーン記述、人間対話を統合 | 閉ループ言語推論の参考 | [arXiv](https://arxiv.org/abs/2207.05608) |
| [PerAct22] | Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation | Shridhar et al. | CoRL 2022 | Multi-task, 3D | ★★ | ボクセル化した3D観測・行動空間でマルチタスク学習。Perceiver Transformerで18タスク(249バリエーション)を学習 | マルチタスク3D操作の参考。RVTの前身 | [arXiv](https://arxiv.org/abs/2209.05451) |
| [GNFactor23] | GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields | Ze et al. | CoRL 2023 | 3D, Multi-task | ★ | Neural Feature Fieldsによるマルチタスク実機学習 | ManiFlow論文で引用。3D表現の参考 | [arXiv](https://arxiv.org/abs/2308.16891) |
| [DNAct24] | DNAct: Diffusion Guided Multi-Task 3D Policy Learning | Yan et al. | arXiv 2024 | Diffusion, 3D | ★ | Diffusionガイドによるマルチタスク3Dポリシー学習 | ManiFlow論文で引用 | [arXiv](https://arxiv.org/abs/2403.04115) |

### 7. 汎化・ドメイン適応

| ID | タイトル | 著者 | 会議/年 | カテゴリ | 関連度 | 内容 | 本研究との関係 | リンク |
|----|---------|------|---------|---------|--------|------|---------------|--------|
| [DR17] | Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World | Tobin et al. | IROS 2017 | Domain Randomization | ★★ | シミュレータでのレンダリングをランダム化し、実世界への転移を実現。物体検出で1.5cm精度達成 | Domain Randomizationの先駆的研究。RoboTwin 2.0のDR設計の参考 | [arXiv](https://arxiv.org/abs/1703.06907) |
| [R3M22] | R3M: A Universal Visual Representation for Robot Manipulation | Nair et al. | CoRL 2022 | Visual Pretraining | ★★ | Ego4D人間動画で視覚表現を事前学習。時間対照学習+動画言語アライメント。12タスクで20%以上改善 | 視覚表現学習の参考。ManiFlowのエンコーダ選択に関連 | [arXiv](https://arxiv.org/abs/2203.12601) |
| [MVP22] | Masked Visual Pre-training for Motor Control | Xiao et al. | arXiv 2022 | Visual Pretraining | ★ | 実世界画像のマスク予測で視覚表現を事前学習。教師ありエンコーダを最大80%上回る | MAEベースの視覚事前学習の参考 | [arXiv](https://arxiv.org/abs/2203.06173) |
| [RealMVP22] | Real-World Robot Learning with Masked Visual Pre-training | Radosavovic et al. | CoRL 2022 | Visual Pretraining | ★ | 4.5M画像でMAE事前学習した307M ViT。CLIP/ImageNet事前学習を最大81%上回る | 大規模視覚事前学習の参考 | [arXiv](https://arxiv.org/abs/2210.03109) |
| [CLIP21] | Learning Transferable Visual Models From Natural Language Supervision | Radford et al. | ICML 2021 | Visual Pretraining | ★ | 4億画像-テキストペアで対照学習。ゼロショットでResNet50相当の性能 | 視覚-言語事前学習の基盤。OpenVLA等のエンコーダ | [arXiv](https://arxiv.org/abs/2103.00020) |
| [DINOv2_23] | DINOv2: Learning Robust Visual Features without Supervision | Oquab et al. | arXiv 2023 | Visual Pretraining | ★ | 142M画像で自己教師あり学習。ファインチューニングなしで汎用的な視覚特徴を提供 | 自己教師あり視覚表現の参考。OpenVLAのエンコーダ | [arXiv](https://arxiv.org/abs/2304.07193) |
| [MimicGen23] | MimicGen: A Data Generation System for Scalable Robot Learning Using Human Demonstrations | Mandlekar et al. | CoRL 2023 | Data Augmentation | ★ | 少数のデモから大量のデータを自動生成。スケーラブルなロボット学習を実現 | HAMSTER論文で引用。データ拡張手法の参考 | [arXiv](https://arxiv.org/abs/2310.17596) |

### 8. 点追跡・光流推定

| ID | タイトル | 著者 | 会議/年 | カテゴリ | 関連度 | 内容 | 本研究との関係 | リンク |
|----|---------|------|---------|---------|--------|------|---------------|--------|
| [TAPIR23] | TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement | Doersch et al. | ICCV 2023 | Point Tracking | ★★ | 任意の点を動画中で追跡。2段階アプローチ（マッチング+リファインメント）でTAP-Vidベンチマーク20%改善 | HAMSTERの2Dパス抽出に使用可能。動画からの軌道抽出の参考 | [arXiv](https://arxiv.org/abs/2306.08637) |
| [CoTracker24] | CoTracker: It is Better to Track Together | Karaev et al. | ECCV 2024 | Point Tracking | ★★ | 複数点を同時に追跡することで精度向上。70k点を同時追跡可能。オクルージョン/視野外でも追跡継続 | HAMSTERの2Dパス抽出に使用可能。HAMSTER論文で引用 | [arXiv](https://arxiv.org/abs/2307.07635) |
| [Track2Act24] | Track2Act: Predicting Point Tracks from Internet Videos enables Generalizable Robot Manipulation | Bharadhwaj et al. | ECCV 2024 | Point Tracking, Manipulation | ★★ | インターネット動画から点追跡を予測し、ゼロショット操作を実現。30タスクで汎化 | 動画からの軌道予測によるゼロショット操作。HAMSTERの関連研究 | [arXiv](https://arxiv.org/abs/2405.01527) |
| [FlowAsInterface24] | Flow as the Cross-Domain Manipulation Interface | Xu et al. | CoRL 2024 | Optical Flow, Manipulation | ★ | 光流を異ドメイン間の操作インターフェースとして使用 | HAMSTER論文で引用。異ドメイン転移の参考 | N/A |
| [GeneralFlow24] | General Flow as Foundation Affordance for Scalable Robot Learning | Yuan et al. | arXiv 2024 | Flow, Affordance | ★ | 一般的なフローをアフォーダンスの基盤として使用 | HAMSTER論文で引用 | [arXiv](https://arxiv.org/abs/2401.11439) |
| [AnyPointTraj23] | Any-Point Trajectory Modeling for Policy Learning | Wen et al. | arXiv 2023 | Point Tracking, Policy | ★ | 任意点軌道モデリングによるポリシー学習 | HAMSTER論文で引用 | [arXiv](https://arxiv.org/abs/2401.00025) |

### 9. テレオペレーション・データ収集

| ID | タイトル | 著者 | 会議/年 | カテゴリ | 関連度 | 内容 | 本研究との関係 | リンク |
|----|---------|------|---------|---------|--------|------|---------------|--------|
| [OpenTeleVision24] | Open-TeleVision: Teleoperation with Immersive Active Visual Feedback | Cheng et al. | CoRL 2024 | Teleoperation | ★ | Apple Vision Proを用いた没入型テレオペレーション | ManiFlowのヒューマノイドデータ収集に使用 | N/A |
| [BunnyVisionPro24] | Bunny-VisionPro: Real-Time Bimanual Dexterous Teleoperation for Imitation Learning | Ding et al. | arXiv 2024 | Teleoperation | ★ | Apple Vision Proによる双腕Dexterousテレオペレーション | ManiFlowの双腕データ収集に使用 | [arXiv](https://arxiv.org/abs/2407.03162) |

---

## 凡例

### 関連度
- ★★★: 直接関連（本研究のベース手法、直接比較対象）
- ★★: 手法参考（アーキテクチャや手法の参考）
- ★: 背景知識（分野の基礎知識、関連研究）

### カテゴリ例
- VLA: Vision-Language-Action
- VLM: Vision-Language Model
- Diffusion Policy: 拡散モデルベースのポリシー
- Flow Matching: フローマッチングベースの手法
- Hierarchical: 階層的アーキテクチャ
- Imitation Learning: 模倣学習
- Benchmark: ベンチマーク・評価環境
- Simulation: シミュレーション環境
- Domain Randomization: ドメインランダム化
- 3D Manipulation: 3D空間でのロボット操作
- Visual Pretraining: 視覚表現の事前学習
- LLM: 大規模言語モデル
- Generative Models: 生成モデル
- Point Tracking: 点追跡
- Data Collection: データ収集
- Teleoperation: テレオペレーション
- Affordance: アフォーダンス
- Dexterous: 器用な操作
