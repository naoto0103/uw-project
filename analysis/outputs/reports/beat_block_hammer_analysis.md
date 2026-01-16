# Analysis Report: Beat Block with Hammer

**Generated**: 2025-12-29 11:30:19

---


## Success Rates

| Condition | Train Env | Mode | Success Rate |
|-----------|-----------|------|--------------|
| 1 | cluttered | original | 4.0% |
| 2 | cluttered | overlay_current | 6.0% |
| 3 | cluttered | overlay_initial_current | 1.0% |
| 4 | clean | original | 31.0% |
| 5 | clean | overlay_current | 37.0% |
| 6 | clean | overlay_initial_current | 11.0% |


## Hypothesis Evaluation

### H1: VILA path guidance improves single-task accuracy (in-domain)

| Comparison | Baseline | Treatment | Improvement | Improved? |
|------------|----------|-----------|-------------|-----------|
| cond1 vs cond2 | 4.0% | 6.0% | +2.0 | ✓ |
| cond1 vs cond3 | 4.0% | 1.0% | -3.0 | ✗ |

### H2: VILA path guidance improves generalization (cross-domain)

| Comparison | Baseline | Treatment | Improvement | Improved? |
|------------|----------|-----------|-------------|-----------|
| cond4 vs cond5 | 31.0% | 37.0% | +6.0 | ✓ |
| cond4 vs cond6 | 31.0% | 11.0% | -20.0 | ✗ |

### H3: Initial path addition improves over current-only

| Comparison | Baseline | Treatment | Improvement | Improved? |
|------------|----------|-----------|-------------|-----------|
| cond2 vs cond3 (in-domain) | 6.0% | 1.0% | -5.0 | ✗ |
| cond5 vs cond6 (cross-domain) | 37.0% | 11.0% | -26.0 | ✗ |

### H4: Path guidance effect is larger in cross-domain condition

| Comparison | In-Domain Δ | Cross-Domain Δ | Difference | Cross > In? |
|------------|-------------|----------------|------------|-------------|
| Δ(cond2-cond1) vs Δ(cond5-cond4) | +2.0 | +6.0 | +4.0 | ✓ |
| Δ(cond3-cond1) vs Δ(cond6-cond4) | -3.0 | -20.0 | -17.0 | ✗ |

## Path Generation Statistics

*Statistics for VILA-based conditions (2, 3, 5, 6)*

| Condition | Path Success | Frame0 Success | Avg Fallbacks |
|-----------|--------------|----------------|---------------|
| 2 | 99.5% | 100.0% | 0.13 |
| 3 | 99.1% | 100.0% | 0.23 |
| 5 | 99.1% | 100.0% | 0.19 |
| 6 | 99.5% | 100.0% | 0.13 |


## Timing Statistics

| Condition | VILA Inference | ManiFlow Inference | Total Episode |
|-----------|----------------|---------------------|---------------|
| 1 | - | 94.0ms | 115.4s |
| 2 | 4522.6ms | 146.2ms | 262.6s |
| 3 | 4110.6ms | 112.1ms | 232.1s |
| 4 | - | 185.2ms | 146.0s |
| 5 | 5366.8ms | 181.5ms | 264.9s |
| 6 | 4191.4ms | 136.0ms | 245.2s |

