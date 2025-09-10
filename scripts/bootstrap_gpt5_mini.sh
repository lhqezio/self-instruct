#!/bin/bash

# GPT-5 Mini Bootstrap Phase
# Generate high-quality conversational examples using GPT-5 Mini

echo "Starting GPT-5 Mini Bootstrap Phase"
echo "===================================="

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Create output directory
mkdir -p data/bootstrap

# Run GPT-5 Mini bootstrap
echo "Generating conversational scenarios with GPT-5 Mini..."
python src/bootstrap/gpt5_mini_bootstrap.py \
    --num_scenarios 5 \
    --output_file data/bootstrap/gpt5_mini_bootstrap.jsonl \
    --api_key "$OPENAI_API_KEY"

echo "GPT-5 Mini Bootstrap Phase Complete!"
echo "Generated data: data/bootstrap/gpt5_mini_bootstrap.jsonl"
