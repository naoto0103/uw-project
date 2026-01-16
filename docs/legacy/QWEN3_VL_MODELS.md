# Qwen3-VL Model Variants

**Release Date**: 2025-09-23
**Developer**: Qwen Team, Alibaba Cloud
**HuggingFace Collection**: [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)
**Requirements**: transformers >= 4.57.0

---

## Overview

Qwen3-VL is the most advanced vision-language model in the Qwen series, offering comprehensive upgrades in text understanding, visual perception, spatial reasoning, and agent interaction capabilities. The series includes 24 model variants spanning from 2B to 235B parameters, with both dense and Mixture-of-Experts (MoE) architectures.

---

## Model Variants Comparison

### Dense Models

| Model | Parameters | Type | Model ID | Size | Use Case |
|-------|-----------|------|----------|------|----------|
| **Qwen3-VL-2B-Instruct** | 2B | Dense | `Qwen/Qwen3-VL-2B-Instruct` | ~4GB | Edge devices, mobile agents |
| **Qwen3-VL-2B-Thinking** | 2B | Dense | `Qwen/Qwen3-VL-2B-Thinking` | ~4GB | Edge reasoning tasks |
| **Qwen3-VL-4B-Instruct** | 4-5B | Dense | `Qwen/Qwen3-VL-4B-Instruct` | ~9GB | Compact deployment |
| **Qwen3-VL-4B-Thinking** | 4-5B | Dense | `Qwen/Qwen3-VL-4B-Thinking` | ~9GB | Compact reasoning |
| **Qwen3-VL-8B-Instruct** | 9B | Dense | `Qwen/Qwen3-VL-8B-Instruct` | ~17GB | **Current default** |
| **Qwen3-VL-8B-Thinking** | 9B | Dense | `Qwen/Qwen3-VL-8B-Thinking` | ~17GB | Enhanced reasoning |
| **Qwen3-VL-32B-Instruct** | 33B | Dense | `Qwen/Qwen3-VL-32B-Instruct` | ~66GB | Balanced performance |
| **Qwen3-VL-32B-Thinking** | 33B | Dense | `Qwen/Qwen3-VL-32B-Thinking` | ~66GB | Advanced reasoning |

### MoE (Mixture-of-Experts) Models

| Model | Total Params | Active Params | Type | Model ID | Size | Use Case |
|-------|-------------|---------------|------|----------|------|----------|
| **Qwen3-VL-30B-A3B-Instruct** | 31B | 3B | MoE | `Qwen/Qwen3-VL-30B-A3B-Instruct` | ~62GB | Efficient inference |
| **Qwen3-VL-30B-A3B-Thinking** | 31B | 3B | MoE | `Qwen/Qwen3-VL-30B-A3B-Thinking` | ~62GB | Reasoning-optimized |
| **Qwen3-VL-235B-A22B-Instruct** | 236B | 22B | MoE | `Qwen/Qwen3-VL-235B-A22B-Instruct` | ~472GB | Flagship model |
| **Qwen3-VL-235B-A22B-Thinking** | 236B | 22B | MoE | `Qwen/Qwen3-VL-235B-A22B-Thinking` | ~472GB | Maximum capability |

### Quantized Variants

| Model | Quantization | Size Reduction | Model ID Pattern |
|-------|-------------|----------------|------------------|
| FP8 Variants | 8-bit floating point | ~50% | `*-FP8` suffix |
| GGUF Variants | Flexible quantization | Variable | `*-GGUF` suffix |

---

## Edition Types

### Instruct Edition
- **Purpose**: Direct instruction following and standard tasks
- **Optimization**: Fast response, efficient inference
- **Best for**: General-purpose applications, API services

### Thinking Edition
- **Purpose**: Enhanced reasoning and complex problem-solving
- **Optimization**: Step-by-step reasoning, deeper analysis
- **Best for**: Scientific reasoning, mathematical problems, complex visual analysis

---

## Key Capabilities

| Capability | Description | Performance |
|-----------|-------------|-------------|
| **Visual Recognition** | Celebrities, products, landmarks, flora, fauna | State-of-the-art |
| **OCR** | 32 languages, blur/lighting robust | High accuracy |
| **2D Grounding** | Object localization, bounding boxes | RefCOCO: 82-87% |
| **3D Grounding** | Spatial reasoning, depth perception | Advanced |
| **Video Understanding** | Temporal grounding, dynamics | Extended context |
| **Visual Agent** | GUI operation (PC/mobile) | Function recognition |
| **Visual Coding** | Draw.io, HTML, CSS, JS generation | Code synthesis |
| **Text Understanding** | On par with pure LLMs | High quality |

---

## Architecture Comparison

### Dense Models
- **Load**: All parameters active during inference
- **Pros**: Consistent performance, simpler deployment
- **Cons**: Higher memory requirements
- **Recommended**: When memory is available

### MoE (Mixture-of-Experts)
- **Load**: Only active experts used per input
- **Pros**: Better efficiency, larger total capacity
- **Cons**: More complex deployment
- **Recommended**: When optimizing compute/memory trade-offs

---

## Recommended Model Selection

### For Robot Trajectory Generation (This Project)

| Priority | Model | Rationale |
|---------|-------|-----------|
| 1st | **Qwen3-VL-8B-Instruct** | Current default, balanced performance/size |
| 2nd | **Qwen3-VL-4B-Instruct** | Faster inference, smaller memory footprint |
| 3rd | **Qwen3-VL-2B-Instruct** | Minimal resource usage, edge deployment |
| Alternative | **Qwen3-VL-8B-Thinking** | If reasoning-focused trajectories improve results |

### General Guidelines

| Constraint | Recommended Model |
|-----------|------------------|
| **VRAM < 8GB** | Qwen3-VL-2B-Instruct |
| **VRAM 8-16GB** | Qwen3-VL-4B-Instruct |
| **VRAM 16-32GB** | Qwen3-VL-8B-Instruct (current) |
| **VRAM 32-64GB** | Qwen3-VL-32B-Instruct |
| **VRAM 64GB+** | Qwen3-VL-30B-A3B-Instruct (MoE) |
| **Multi-GPU** | Qwen3-VL-235B-A22B-Instruct (flagship) |

---

## Performance Benchmarks

### Spatial Reasoning (Relevant for Robot Tasks)

| Model | RefCOCO | 3D Grounding | Spatial Relations |
|-------|---------|--------------|-------------------|
| Qwen3-VL-2B | ~70% | Basic | Good |
| Qwen3-VL-4B | ~75% | Moderate | Good |
| Qwen3-VL-8B | **82-87%** | **Strong** | **Excellent** |
| Qwen3-VL-32B | ~90% | Very Strong | Excellent |

---

## Usage Notes

### Switching Models

To use a different model in this project:

1. Update `server_qwen3.py` or use command-line argument:
   ```bash
   python server_qwen3.py --model-path Qwen/Qwen3-VL-4B-Instruct
   ```

2. Or set environment variable:
   ```bash
   export MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
   ./setup_qwen3_server.sh
   ```

### Model Download

Models are automatically downloaded from HuggingFace Hub to:
```
~/.cache/huggingface/hub/models--Qwen--{MODEL_NAME}/
```

### Version Requirements

- **transformers**: >= 4.57.0
- **torch**: >= 2.0.0 (nightly recommended for RTX 5090)
- **Python**: >= 3.8

---

## Links

- **Official Repository**: https://github.com/QwenLM/Qwen3-VL
- **HuggingFace Collection**: https://huggingface.co/collections/Qwen/qwen3-vl
- **Documentation**: https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl
- **Demo Space**: https://huggingface.co/spaces/Qwen/Qwen3-VL-8B-Instruct

---

**Last Updated**: 2025-11-20
