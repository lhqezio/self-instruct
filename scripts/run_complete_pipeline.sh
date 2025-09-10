#!/bin/bash

# Complete Hybrid Self-Instruct Pipeline
# Run all phases in sequence

echo "Hybrid Self-Instruct for Conversational NPCs"
echo "============================================="
echo ""

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Phase 1: GPT-5 Mini Bootstrap
echo "Phase 1: GPT-5 Mini Bootstrap"
echo "------------------------------"
./scripts/bootstrap_gpt5_mini.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Bootstrap phase failed!"
    exit 1
fi

echo ""

# Phase 2: Gemma 3 1B Self-Growth
echo "Phase 2: Gemma 3 1B Self-Growth"
echo "-------------------------------"
./scripts/self_grow_gemma3.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Self-growth phase failed!"
    exit 1
fi

echo ""

# Phase 3: Prepare Training Data
echo "Phase 3: Prepare Training Data"
echo "------------------------------"
./scripts/prepare_training.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Training preparation phase failed!"
    exit 1
fi

echo ""
echo "Complete Pipeline Finished Successfully!"
echo "======================================="
echo ""
echo "Final Results:"
echo "  - Bootstrap data: data/bootstrap/gpt5_mini_bootstrap.jsonl"
echo "  - Expanded data: data/self_growth/gemma3_expanded.jsonl"
echo "  - Training data: data/final/gemma_training_data.jsonl"
echo ""
echo "Your conversational NPC training data is ready!"
echo "Use data/final/gemma_training_data.jsonl for Gemma fine-tuning."
