# Analysis Report: Click Bell

**Generated**: 2025-12-29 11:30:19

---


## Success Rates

| Condition | Train Env | Mode | Success Rate |
|-----------|-----------|------|--------------|
| 1 | cluttered | original | 4.0% |
| 2 | cluttered | overlay_current | 2.0% |
| 3 | cluttered | overlay_initial_current | 4.0% |
| 4 | clean | original | 3.0% |
| 5 | clean | overlay_current | 20.0% |
| 6 | clean | overlay_initial_current | 1.0% |


## Hypothesis Evaluation

### H1: VILA path guidance improves single-task accuracy (in-domain)

| Comparison | Baseline | Treatment | Improvement | Improved? |
|------------|----------|-----------|-------------|-----------|
| cond1 vs cond2 | 4.0% | 2.0% | -2.0 | ✗ |
| cond1 vs cond3 | 4.0% | 4.0% | +0.0 | ✗ |

### H2: VILA path guidance improves generalization (cross-domain)

| Comparison | Baseline | Treatment | Improvement | Improved? |
|------------|----------|-----------|-------------|-----------|
| cond4 vs cond5 | 3.0% | 20.0% | +17.0 | ✓ |
| cond4 vs cond6 | 3.0% | 1.0% | -2.0 | ✗ |

### H3: Initial path addition improves over current-only

| Comparison | Baseline | Treatment | Improvement | Improved? |
|------------|----------|-----------|-------------|-----------|
| cond2 vs cond3 (in-domain) | 2.0% | 4.0% | +2.0 | ✓ |
| cond5 vs cond6 (cross-domain) | 20.0% | 1.0% | -19.0 | ✗ |

### H4: Path guidance effect is larger in cross-domain condition

| Comparison | In-Domain Δ | Cross-Domain Δ | Difference | Cross > In? |
|------------|-------------|----------------|------------|-------------|
| Δ(cond2-cond1) vs Δ(cond5-cond4) | -2.0 | +17.0 | +19.0 | ✓ |
| Δ(cond3-cond1) vs Δ(cond6-cond4) | +0.0 | -2.0 | -2.0 | ✗ |

## Path Generation Statistics

*Statistics for VILA-based conditions (2, 3, 5, 6)*

| Condition | Path Success | Frame0 Success | Avg Fallbacks |
|-----------|--------------|----------------|---------------|
| 2 | 94.5% | 89.0% | 1.46 |
| 3 | 93.5% | 89.0% | 1.71 |
| 5 | 91.1% | 89.0% | 2.11 |
| 6 | 87.2% | 89.0% | 3.43 |


## Timing Statistics

| Condition | VILA Inference | ManiFlow Inference | Total Episode |
|-----------|----------------|---------------------|---------------|
| 1 | - | 113.6ms | 130.0s |
| 2 | 3413.0ms | 93.6ms | 198.9s |
| 3 | 3886.6ms | 121.3ms | 223.4s |
| 4 | - | 106.2ms | 137.7s |
| 5 | 3769.4ms | 113.1ms | 220.2s |
| 6 | 3866.0ms | 118.1ms | 255.6s |

