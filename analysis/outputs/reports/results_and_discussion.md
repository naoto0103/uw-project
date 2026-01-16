# 実験結果と考察

**作成日**: 2025-12-29

---

## 1. 実験結果

### 1.1 タスク成功率

**参照**: `tables/main_results.md`, `figures/main_results_grouped.png`

Table 1に、3タスク×6条件における成功率を示す。

| Task | C1 | C2 | C3 | C4 | C5 | C6 |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Click Bell | 4.0% | 2.0% | 4.0% | 3.0% | **20.0%** | 1.0% |
| Move Can to Pot | 21.0% | 20.0% | **26.0%** | 27.0% | 27.0% | 27.0% |
| Beat Block with Hammer | 4.0% | 6.0% | 1.0% | 31.0% | **37.0%** | 11.0% |
| **Average** | 9.7% | 9.3% | 10.3% | 20.3% | **28.0%** | 13.0% |

*C1-C3: cluttered環境で学習、C4-C6: clean環境で学習。全条件cluttered環境で評価。*

*C1,C4: オリジナルManiFlow。C2,C5: VILA current path。C3,C6: VILA initial+current path。*

**主要な発見**:

1. **条件5（clean学習 + current overlay）が最高性能**を達成（平均28.0%）
2. **clean環境で学習した条件（C4-C6）がcluttered環境で学習した条件（C1-C3）より全体的に高い成功率**を示した（平均20.3-28.0% vs 9.3-10.3%）
3. **initial+current（C3, C6）は期待に反して低い成功率**を示した（特にC6: 13.0%）

### 1.2 仮説検証結果

**参照**: `tables/hypothesis_results.md`, `figures/hypothesis_h1_h2.png`, `figures/hypothesis_h3.png`, `figures/hypothesis_h4.png`

#### H1: VILAのパスガイダンスは単一タスク精度を向上させる（in-domain）

| 比較 | 平均改善 | 支持タスク |
|:-----|:--------:|:----------:|
| C1→C2 | -0.3 | 1/3 |
| C1→C3 | +0.7 | 1/3 |

**結果**: H1は**支持されなかった**。in-domain条件（cluttered→cluttered）では、VILAパスガイダンスによる一貫した改善は見られなかった。

#### H2: VILAのパスガイダンスは汎化性能を向上させる（cross-domain）

| 比較 | 平均改善 | 支持タスク |
|:-----|:--------:|:----------:|
| C4→C5 | **+7.7** | 2/3 |
| C4→C6 | -7.3 | 0/3 |

**結果**: H2は**current pathモード（C5）で部分的に支持された**。C4→C5では平均+7.7%の改善が見られ、3タスク中2タスクで改善が確認された。一方、initial+current（C6）では逆に性能が低下した。

#### H3: Initial pathの追加はcurrent pathのみより性能を向上させる

**参照**: `figures/hypothesis_h3.png`

| 条件 | 比較 | 平均改善 | 支持タスク |
|:-----|:-----|:--------:|:----------:|
| In-domain | C2→C3 | +1.0 | 2/3 |
| Cross-domain | C5→C6 | **-15.0** | 0/3 |

**結果**: H3は**支持されなかった**。特にcross-domain条件（C5→C6）では、initial pathの追加により平均-15.0%という大幅な性能低下が観察された。

#### H4: パスガイダンスの効果はドメインギャップ条件でより顕著である

**参照**: `figures/hypothesis_h4.png`

| 比較 | In-domain Δ | Cross-domain Δ | 支持タスク |
|:-----|:-----------:|:--------------:|:----------:|
| C1→C2 vs C4→C5 | -0.3 | **+7.7** | 3/3 |
| C1→C3 vs C4→C6 | +0.7 | -7.3 | 0/3 |

**結果**: H4は**current pathモードで支持された**。C1→C2 vs C4→C5の比較では、3タスク全てでcross-domain条件の改善幅がin-domain条件より大きかった。

### 1.3 パス生成統計

**参照**: `tables/path_stats.md`, `figures/path_stats.png`

| タスク | 条件 | パス成功率 | Frame0成功率 | 平均フォールバック数 |
|:-------|:----:|:----------:|:------------:|:------------------:|
| Click Bell | C2 | 94.5% | 89.0% | 1.46 |
| | C5 | 91.1% | 89.0% | 2.11 |
| Move Can to Pot | C2 | 98.8% | 98.0% | 0.29 |
| | C5 | 98.2% | 98.0% | 0.40 |
| Beat Block with Hammer | C2 | 99.5% | 100.0% | 0.13 |
| | C5 | 99.1% | 100.0% | 0.19 |

**主要な発見**:

1. **パス生成成功率は全体的に高い**（91.1%〜99.5%）
2. **Click Bellタスクでパス生成成功率が相対的に低い**（91.1%〜94.5%）。これはベルのクリック動作が微細であり、VILAがパスを生成しにくいためと考えられる
3. **フォールバック使用頻度はClick Bellで最も高い**（1.46〜3.43回/エピソード）

### 1.4 推論時間

**参照**: `tables/timing_stats.md`

| 条件 | VILA (ms) | ManiFlow (ms) | オーバーヘッド |
|:----:|:---------:|:-------------:|:--------------:|
| C1 | - | 108.0 | - |
| C2 | 4312.5 | 117.8 | +4312ms |
| C3 | 4363.2 | 117.3 | +4363ms |
| C4 | - | 135.8 | - |
| C5 | 4703.4 | 136.2 | +4703ms |
| C6 | 4360.8 | 125.7 | +4361ms |

**主要な発見**:

1. **VILA推論に約4.3〜4.7秒のオーバーヘッド**が発生
2. **ManiFlow単体の推論時間は約100〜140ms**と高速
3. VILAパスガイダンス使用時は、16ステップごとにVILA推論が必要なため、**エピソード全体で約2倍の時間**がかかる

### 1.5 学習曲線分析

**参照**: `figures/training_curves_by_task.png`, `figures/validation_loss_comparison.png`, `figures/final_loss_comparison.png`

全18条件（3タスク×6条件）について、501エポックの学習曲線を分析した。

#### 1.5.1 学習の収束状況

| タスク | 条件 | 初期Loss | 最終Loss | 最小Loss | Loss削減率 |
|:-------|:----:|:--------:|:--------:|:--------:|:----------:|
| Click Bell | C1 | 2.761 | 0.0034 | 0.0028 | 99.9% |
| | C2 | 2.762 | 0.0034 | 0.0029 | 99.9% |
| | C3 | 2.765 | 0.0031 | 0.0028 | 99.9% |
| | C4 | 2.762 | 0.0029 | 0.0026 | 99.9% |
| | C5 | 2.762 | 0.0030 | 0.0027 | 99.9% |
| | C6 | 2.769 | 0.0025 | 0.0023 | 99.9% |
| Move Can to Pot | C1 | 2.330 | 0.0022 | 0.0019 | 99.9% |
| | C2 | 2.330 | 0.0021 | 0.0018 | 99.9% |
| | C3 | 2.343 | 0.0022 | 0.0019 | 99.9% |
| | C4 | 2.325 | 0.0023 | 0.0023 | 99.9% |
| | C5 | 2.325 | 0.0024 | 0.0023 | 99.9% |
| | C6 | 2.341 | 0.0026 | 0.0021 | 99.9% |
| Beat Block with Hammer | C1 | 2.510 | 0.0035 | 0.0003 | 99.9% |
| | C2 | 2.510 | 0.0013 | 0.0004 | 99.9% |
| | C3 | 2.530 | 0.0029 | 0.0006 | 99.9% |
| | C4 | 2.250 | 0.0023 | 0.0009 | 99.9% |
| | C5 | 2.280 | 0.0020 | 0.0010 | 99.9% |
| | C6 | 2.280 | 0.0016 | 0.0009 | 99.9% |

**主要な発見**:

1. **全条件で十分な収束**を達成（Loss削減率99.9%）
2. **Clean環境（C4-C6）の初期Lossがcluttered環境（C1-C3）より低い傾向**：特にBeat Block with Hammerで顕著（2.25-2.28 vs 2.51-2.53）
3. **タスク間で初期Loss値に差異**：Click Bell（約2.76）> Beat Block with Hammer（約2.25-2.53）> Move Can to Pot（約2.33）

#### 1.5.2 学習曲線の特徴

**参照**: `figures/convergence_analysis.png`

- **Click Bell**: 全条件で類似した学習曲線を示し、約100エポックで急速に収束
- **Move Can to Pot**: 最も安定した学習曲線を示し、条件間の差異が最小
- **Beat Block with Hammer**: Clean環境（C4-C6）とCluttered環境（C1-C3）で明確に異なる学習パターン

---

## 2. 考察

### 2.1 Clean環境学習の優位性

**参照**: `figures/main_results_grouped.png`, `figures/average_results.png`, `figures/training_curves_by_task.png`

最も顕著な発見は、**clean環境で学習したモデル（C4-C6）がcluttered環境で学習したモデル（C1-C3）より高い成功率を示した**ことである。これは当初の予想と逆の結果である。

学習曲線の分析から、以下の追加的な知見が得られた：

- **Clean環境での学習は初期Lossが低い**：特にBeat Block with Hammerでは、clean環境（C4-C6）の初期Loss（2.25-2.28）がcluttered環境（C1-C3）の初期Loss（2.51-2.53）より約10%低かった
- **収束後のLossは条件間で大差なし**：全条件で最終Loss 0.002-0.004程度まで収束しており、Lossの観点からは同等に学習が進んでいる

考えられる要因:

1. **学習データの質**: clean環境では視覚的なノイズが少なく、タスクに関連する特徴をより効率的に学習できた可能性がある。初期Lossの差はこれを裏付ける
2. **オーバーフィッティング**: cluttered環境で学習したモデルは、学習データ特有の散乱パターンにオーバーフィットし、評価時の異なる散乱パターンに対応できなかった可能性がある。**Lossが同等でも汎化性能が異なる**ことがこれを示唆している
3. **タスク本質の学習**: clean環境では、タスク遂行に本質的な動作パターンに集中して学習できた可能性がある

### 2.2 Current Path vs Initial+Current Path

**参照**: `figures/hypothesis_h3.png`, `tables/hypothesis_results.md`

当初、initial pathを追加することで「Memory Function」として機能し、オクルージョン時のロバスト性が向上すると期待していた。しかし、実験結果は**initial+current（C3, C6）がcurrent only（C2, C5）より低い成功率**を示した。

考えられる要因:

1. **入力の複雑化**: 2つのオーバーレイ画像（initial + current）を入力することで、モデルが処理すべき情報量が増加し、学習が困難になった可能性がある
2. **時間的不整合**: initial pathとcurrent pathの間に時間的なギャップがあり、モデルが2つのパスの関係性を適切に学習できなかった可能性がある
3. **initial pathの陳腐化**: エピソードが進むにつれて、initial path（frame 0時点のパス）の情報が現在の状況と乖離し、誤った誘導を与えた可能性がある

### 2.3 VILAパスガイダンスの効果

**参照**: `figures/hypothesis_h1_h2.png`, `figures/hypothesis_h4.png`

VILAパスガイダンスの効果は**条件によって大きく異なる**結果となった:

- **In-domain（cluttered→cluttered）**: ほぼ効果なし（H1不支持）
- **Cross-domain（clean→cluttered）+ current path**: 効果あり（H2部分的支持、H4支持）
- **Cross-domain + initial+current**: 逆効果

この結果は、**VILAのパスガイダンスがドメインギャップを埋めるブリッジとして機能する可能性**を示唆している。ただし、その効果はパスの入力方法（current only vs initial+current）に強く依存する。

### 2.4 タスク依存性

**参照**: `figures/main_results_all.png`

3タスク間で結果のパターンが異なることが観察された:

- **Click Bell**: C5で大幅改善（3%→20%）、他の条件では低迷
- **Move Can to Pot**: 条件間の差が小さい（20〜27%）
- **Beat Block with Hammer**: C4, C5で高い成功率（31%, 37%）、C3, C6で低迷

これは、**VILAパスガイダンスの効果がタスクの特性に依存する**ことを示唆している。特に、明確な軌道を必要とするタスク（Click Bell、Beat Block with Hammer）でパスガイダンスの効果が顕著に現れる傾向がある。

### 2.5 パス生成の信頼性

**参照**: `figures/path_stats.png`, `tables/path_stats.md`

パス生成成功率は全体的に高い（91〜99%）ものの、**Click Bellタスクで相対的に低い成功率とフォールバック使用頻度の増加**が観察された。これは、微細な動作を必要とするタスクではVILAのパス生成精度が低下する可能性を示唆している。

ただし、パス生成成功率とタスク成功率の間に明確な相関は見られなかった。例えば、Beat Block with Hammerはパス生成成功率が最も高い（99%以上）にもかかわらず、C3では1%という非常に低いタスク成功率を示した。これは、**パス生成の成功と適切なパスの生成は別の問題**であることを示唆している。

### 2.6 計算コストのトレードオフ

**参照**: `tables/timing_stats.md`

VILAパスガイダンスは約4〜5秒のオーバーヘッドを伴う。現在の実装では16ステップごとにVILA推論を行うため、エピソード全体では**約2倍の時間**がかかる。

Cross-domain条件でのC5の改善（+7.7%）を考慮すると、**汎化性能が重要な応用シナリオではこのオーバーヘッドは許容可能**と考えられる。一方、リアルタイム性が求められる応用では、より効率的なパス生成手法の開発が必要である。

---

## 3. 結論

本実験から得られた主要な知見:

1. **VILAパスガイダンスはcross-domain汎化に効果的**（current pathモード使用時）
2. **Memory Function（initial+current）は期待に反して逆効果**
3. **Clean環境での学習がcluttered環境での学習より優れた汎化性能**を示す
4. **パスガイダンスの効果はタスク特性に依存**

今後の研究方向:

1. initial+current方式の改良（時間的整合性の向上、適応的なパス更新）
2. より効率的なパス生成手法の開発
3. 多様なタスク・環境での検証
4. clean環境学習の優位性のメカニズム解明
5. 実ロボットでの検証

---

## 参照ファイル一覧

### 表

| ファイル | 内容 |
|:---------|:-----|
| `tables/main_results.md` | タスク成功率の主要結果 |
| `tables/hypothesis_results.md` | 仮説検証の数値結果 |
| `tables/path_stats.md` | パス生成統計 |
| `tables/timing_stats.md` | 推論時間統計 |

### 図

| ファイル | 内容 |
|:---------|:-----|
| `figures/main_results_grouped.png` | 条件別成功率（グループ化） |
| `figures/main_results_all.png` | 全条件の成功率比較 |
| `figures/average_results.png` | 平均成功率 |
| `figures/hypothesis_h1_h2.png` | H1・H2検証結果 |
| `figures/hypothesis_h3.png` | H3検証結果 |
| `figures/hypothesis_h4.png` | H4検証結果 |
| `figures/path_stats.png` | パス生成統計 |
| `figures/summary_figure.png` | 総合サマリー |
| `figures/training_curves_by_task.png` | タスク別学習曲線（6条件比較） |
| `figures/validation_loss_comparison.png` | 検証Loss比較 |
| `figures/final_loss_comparison.png` | 最終Loss比較（バーチャート） |
| `figures/convergence_analysis.png` | 収束速度分析 |
