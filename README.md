# Hybrid Self-Instruct for Conversational NPCs

A hybrid approach combining GPT-4 bootstrap with Gemma self-growth to create conversational NPC training data.

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
