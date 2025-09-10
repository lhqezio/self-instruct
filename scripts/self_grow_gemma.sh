#!/bin/bash

# Gemma Self-Growth Phase
# Use Gemma to iteratively expand the dataset

echo "Starting Gemma Self-Growth Phase"
echo "================================="

# Check if input data exists
if [ ! -f "data/bootstrap/gpt4_bootstrap.jsonl" ]; then
    echo "ERROR: Bootstrap data not found. Please run bootstrap_gpt4.sh first."
    exit 1
fi

# Create output directory
mkdir -p data/self_growth

# Run Gemma self-growth
echo "Starting iterative self-growth with Gemma..."
python src/self_growth/gemma_self_growth.py \
    --input_file data/bootstrap/gpt4_bootstrap.jsonl \
    --output_file data/self_growth/gemma_expanded.jsonl \
    --growth_rounds 3 \
    --model_name "google/gemma-1b"

echo "Gemma Self-Growth Phase Complete!"
echo "Expanded data: data/self_growth/gemma_expanded.jsonl"
