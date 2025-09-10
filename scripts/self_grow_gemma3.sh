#!/bin/bash

# Gemma 3 1B Self-Growth Phase
# Use Gemma 3 1B to iteratively expand the dataset

echo "Starting Gemma 3 1B Self-Growth Phase"
echo "======================================"

# Check if input data exists
if [ ! -f "data/bootstrap/gpt5_mini_bootstrap.jsonl" ]; then
    echo "ERROR: Bootstrap data not found. Please run bootstrap_gpt5_mini.sh first."
    exit 1
fi

# Create output directory
mkdir -p data/self_growth

# Run Gemma 3 1B self-growth
echo "Starting iterative self-growth with Gemma 3 1B..."
python src/self_growth/gemma3_self_growth.py \
    --input_file data/bootstrap/gpt5_mini_bootstrap.jsonl \
    --output_file data/self_growth/gemma3_expanded.jsonl \
    --growth_rounds 3 \
    --model_name "google/gemma-2-1b"

echo "Gemma 3 1B Self-Growth Phase Complete!"
echo "Expanded data: data/self_growth/gemma3_expanded.jsonl"
