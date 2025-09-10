# Hybrid Self-Instruct for Conversational NPCs

An experimental approach combining GPT-4 bootstrap with Gemma self-growth to enhance Small Language Model (SLM) capabilities for game NPC conversational tasks.

## Motivation

This project explores whether we can significantly improve the conversational abilities of small language models (like Gemma 1B) for game NPCs by:

1. **Bootstrap Phase**: Using GPT-4's superior capabilities to generate high-quality conversational examples
2. **Self-Growth Phase**: Having the target SLM (Gemma) learn from and expand upon its own generated outputs
3. **Conversational Focus**: Specializing the entire pipeline for natural, game-appropriate NPC interactions

The goal is to create a cost-effective solution for game developers who need conversational NPCs but cannot afford to run large language models in real-time.

## Architecture Overview

### Phase 1: GPT-4 Bootstrap
- Generate high-quality conversational examples using GPT-4
- Create diverse NPC interaction patterns
- Establish quality baseline for self-growth

### Phase 2: Gemma Self-Growth 
- Use Gemma to iteratively expand the dataset
- Self-Instruct style bootstrapping with Gemma
- Grow from 1K → 5K → 10K+ examples

### Phase 3: Training & Deployment
- Quality control and filtering
- Convert to training format
- Fine-tune Gemma on expanded dataset

## Quick Start

```bash
# 1. Bootstrap with GPT-4
./scripts/bootstrap_gpt4.sh

# 2. Self-grow with Gemma
./scripts/self_grow_gemma.sh

# 3. Prepare for training
./scripts/prepare_training.sh
```

## Repository Structure

```
src/
├── bootstrap/          # GPT-4 bootstrap phase
├── self_growth/        # Gemma self-growth phase
└── utils/              # Shared utilities

data/
├── bootstrap/          # GPT-4 generated data
├── self_growth/        # Gemma expanded data
└── final/              # Final training data

scripts/                # Workflow scripts
```

## Key Innovation

Instead of using the same model for both generation and self-growth, we use:
- **GPT-4** for high-quality bootstrap (expensive but good)
- **Gemma** for iterative self-growth (cheap and fast)

This gives us the best of both worlds: quality + scalability.

## Experimental Nature

This is an experimental approach that investigates several research questions:

- **Can SLMs be effectively trained for conversational tasks using hybrid self-instruct?**
- **Does GPT-4 bootstrap + SLM self-growth produce better results than traditional fine-tuning?**
- **How well do small models perform on game-specific conversational tasks after this training?**

### Target Use Cases

- **Game NPCs**: Characters that can engage in natural conversation with players
- **Quest Givers**: NPCs that can explain objectives and provide hints
- **Shopkeepers**: NPCs that can discuss items and make recommendations
- **Lore Keepers**: NPCs that can explain game world history and background
- **Tutorial Guides**: NPCs that can help new players learn game mechanics

### Benefits for Game Development

- **Cost-Effective**: Small models can run on consumer hardware
- **Low Latency**: Fast response times for real-time gameplay
- **Offline Capable**: No need for internet connection during gameplay
- **Customizable**: Can be fine-tuned for specific game worlds and characters

## Citation

This work is based on the original Self-Instruct paper and implementation:

```bibtex
@misc{selfinstruct,
  title={Self-Instruct: Aligning Language Model with Self Generated Instructions},
  author={Wang, Yizhong and Kordi, Yeganeh and Mishra, Swaroop and Liu, Alisa and Smith, Noah A. and Khashabi, Daniel and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2212.10560},
  year={2022}
}
```

**Original Repository**: [https://github.com/yizhongw/self-instruct](https://github.com/yizhongw/self-instruct)

**Paper**: [https://arxiv.org/abs/2212.10560](https://arxiv.org/abs/2212.10560)
