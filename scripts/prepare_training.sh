#!/bin/bash

# Prepare Training Data Phase
# Process and prepare data for Gemma fine-tuning

echo "Starting Training Data Preparation Phase"
echo "========================================"

# Check if self-growth data exists
if [ ! -f "data/self_growth/gemma3_expanded.jsonl" ]; then
    echo "ERROR: Self-growth data not found. Please run self_grow_gemma3.sh first."
    exit 1
fi

# Create output directory
mkdir -p data/final

# Process data
echo "Processing and cleaning data..."

# Filter quality and remove duplicates
python src/utils/data_processor.py \
    --input_file data/self_growth/gemma3_expanded.jsonl \
    --output_file data/final/processed_data.jsonl \
    --filter_quality \
    --deduplicate \
    --stats

# Convert to Gemma format
echo "Converting to Gemma training format..."
python src/utils/data_processor.py \
    --input_file data/final/processed_data.jsonl \
    --output_file data/final/gemma_training_data.jsonl \
    --convert_gemma \
    --stats

echo "Training Data Preparation Complete!"
echo "Final training data: data/final/gemma_training_data.jsonl"
echo ""
echo "Your data is ready for Gemma fine-tuning!"
echo "Use data/final/gemma_training_data.jsonl as your training dataset."
