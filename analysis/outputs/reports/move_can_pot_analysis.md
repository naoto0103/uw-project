# Analysis Report: Move Can to Pot

**Generated**: 2025-12-29 11:30:19

---


## Success Rates

| Condition | Train Env | Mode | Success Rate |
|-----------|-----------|------|--------------|
| 1 | cluttered | original | 21.0% |
| 2 | cluttered | overlay_current | 20.0% |
| 3 | cluttered | overlay_initial_current | 26.0% |
| 4 | clean | original | 27.0% |
| 5 | clean | overlay_current | 27.0% |
| 6 | clean | overlay_initial_current | 27.0% |


## Hypothesis Evaluation

### H1: VILA path guidance improves single-task accuracy (in-domain)

| Comparison | Baseline | Treatment | Improvement | Improved? |
|------------|----------|-----------|-------------|-----------|
| cond1 vs cond2 | 21.0% | 20.0% | -1.0 | ✗ |
| cond1 vs cond3 | 21.0% | 26.0% | +5.0 | ✓ |

### H2: VILA path guidance improves generalization (cross-domain)

| Comparison | Baseline | Treatment | Improvement | Improved? |
|------------|----------|-----------|-------------|-----------|
| cond4 vs cond5 | 27.0% | 27.0% | +0.0 | ✗ |
| cond4 vs cond6 | 27.0% | 27.0% | +0.0 | ✗ |

### H3: Initial path addition improves over current-only

| Comparison | Baseline | Treatment | Improvement | Improved? |
|------------|----------|-----------|-------------|-----------|
| cond2 vs cond3 (in-domain) | 20.0% | 26.0% | +6.0 | ✓ |
| cond5 vs cond6 (cross-domain) | 27.0% | 27.0% | +0.0 | ✗ |

### H4: Path guidance effect is larger in cross-domain condition

| Comparison | In-Domain Δ | Cross-Domain Δ | Difference | Cross > In? |
|------------|-------------|----------------|------------|-------------|
| Δ(cond2-cond1) vs Δ(cond5-cond4) | -1.0 | +0.0 | +1.0 | ✓ |
| Δ(cond3-cond1) vs Δ(cond6-cond4) | +5.0 | +0.0 | -5.0 | ✗ |

## Path Generation Statistics

*Statistics for VILA-based conditions (2, 3, 5, 6)*

| Condition | Path Success | Frame0 Success | Avg Fallbacks |
|-----------|--------------|----------------|---------------|
| 2 | 98.8% | 98.0% | 0.29 |
| 3 | 97.4% | 98.0% | 0.58 |
| 5 | 98.2% | 98.0% | 0.40 |
| 6 | 97.9% | 98.0% | 0.49 |


## Timing Statistics

| Condition | VILA Inference | ManiFlow Inference | Total Episode |
|-----------|----------------|---------------------|---------------|
| 1 | - | 116.4ms | 160.7s |
| 2 | 5001.8ms | 113.6ms | 280.4s |
| 3 | 5092.6ms | 118.4ms | 271.6s |
| 4 | - | 116.0ms | 159.9s |
| 5 | 4974.1ms | 114.0ms | 273.1s |
| 6 | 5024.9ms | 122.9ms | 279.1s |

