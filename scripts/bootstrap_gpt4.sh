#!/bin/bash

# GPT-4 Bootstrap Phase
# Generate high-quality conversational examples using GPT-4

echo "Starting GPT-4 Bootstrap Phase"
echo "==============================="

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Create output directory
mkdir -p data/bootstrap

# Run GPT-4 bootstrap
echo "Generating conversational scenarios with GPT-4..."
python src/bootstrap/gpt4_bootstrap.py \
    --num_scenarios 500 \
    --output_file data/bootstrap/gpt4_bootstrap.jsonl \
    --api_key "$OPENAI_API_KEY"

echo "GPT-4 Bootstrap Phase Complete!"
echo "Generated data: data/bootstrap/gpt4_bootstrap.jsonl"
