#!/usr/bin/env python3
"""
Data processing utilities for hybrid Self-Instruct
"""

import json
import argparse
from typing import List, Dict
from pathlib import Path
import random

class DataProcessor:
    def __init__(self):
        pass
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file"""
        
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue
        
        return data
    
    def save_data(self, data: List[Dict], file_path: str):
        """Save data to JSONL file"""
        
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    def convert_to_gemma_format(self, data: List[Dict]) -> List[Dict]:
        """Convert data to Gemma training format"""
        
        gemma_data = []
        
        for item in data:
            # Create conversational format
            text = f"### Human:\n{item['input']}\n\n### Assistant:\n{item['output']}"
            
            gemma_data.append({
                "text": text,
                "scenario_type": item.get('scenario_type', 'unknown'),
                "source": item.get('source', 'unknown')
            })
        
        return gemma_data
    
    def filter_quality(self, data: List[Dict], min_length: int = 10, max_length: int = 500) -> List[Dict]:
        """Filter data based on quality criteria"""
        
        filtered_data = []
        
        for item in data:
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            
            # Check length requirements
            if len(input_text) < min_length or len(output_text) < min_length:
                continue
            
            if len(input_text) > max_length or len(output_text) > max_length:
                continue
            
            # Check for basic quality
            if not input_text.strip() or not output_text.strip():
                continue
            
            # Check for repetitive content
            if input_text.lower() == output_text.lower():
                continue
            
            filtered_data.append(item)
        
        return filtered_data
    
    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicate conversations"""
        
        seen = set()
        unique_data = []
        
        for item in data:
            # Create a key based on input and output
            key = (item['input'].lower().strip(), item['output'].lower().strip())
            
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        return unique_data
    
    def balance_dataset(self, data: List[Dict], target_size: int = None) -> List[Dict]:
        """Balance dataset by scenario type"""
        
        if target_size is None:
            target_size = len(data)
        
        # Group by scenario type
        by_type = {}
        for item in data:
            scenario_type = item.get('scenario_type', 'unknown')
            if scenario_type not in by_type:
                by_type[scenario_type] = []
            by_type[scenario_type].append(item)
        
        # Calculate target per type
        types = list(by_type.keys())
        per_type = target_size // len(types)
        
        balanced_data = []
        for scenario_type in types:
            type_data = by_type[scenario_type]
            if len(type_data) > per_type:
                # Sample if too many
                balanced_data.extend(random.sample(type_data, per_type))
            else:
                # Use all if not enough
                balanced_data.extend(type_data)
        
        # Fill remaining slots randomly
        remaining = target_size - len(balanced_data)
        if remaining > 0:
            all_data = [item for items in by_type.values() for item in items]
            balanced_data.extend(random.sample(all_data, min(remaining, len(all_data))))
        
        return balanced_data
    
    def get_statistics(self, data: List[Dict]) -> Dict:
        """Get dataset statistics"""
        
        stats = {
            'total_examples': len(data),
            'scenario_types': {},
            'sources': {},
            'avg_input_length': 0,
            'avg_output_length': 0
        }
        
        total_input_length = 0
        total_output_length = 0
        
        for item in data:
            # Count scenario types
            scenario_type = item.get('scenario_type', 'unknown')
            stats['scenario_types'][scenario_type] = stats['scenario_types'].get(scenario_type, 0) + 1
            
            # Count sources
            source = item.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
            
            # Calculate lengths
            input_length = len(item.get('input', ''))
            output_length = len(item.get('output', ''))
            total_input_length += input_length
            total_output_length += output_length
        
        stats['avg_input_length'] = total_input_length / len(data) if data else 0
        stats['avg_output_length'] = total_output_length / len(data) if data else 0
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Data processing utilities")
    parser.add_argument("--input_file", required=True, help="Input data file")
    parser.add_argument("--output_file", required=True, help="Output data file")
    parser.add_argument("--convert_gemma", action="store_true", help="Convert to Gemma format")
    parser.add_argument("--filter_quality", action="store_true", help="Filter for quality")
    parser.add_argument("--deduplicate", action="store_true", help="Remove duplicates")
    parser.add_argument("--balance", type=int, help="Balance dataset to target size")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DataProcessor()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = processor.load_data(args.input_file)
    print(f"Loaded {len(data)} examples")
    
    # Process data
    if args.filter_quality:
        print("Filtering for quality...")
        data = processor.filter_quality(data)
        print(f"After quality filtering: {len(data)} examples")
    
    if args.deduplicate:
        print("Removing duplicates...")
        data = processor.deduplicate(data)
        print(f"After deduplication: {len(data)} examples")
    
    if args.balance:
        print(f"Balancing dataset to {args.balance} examples...")
        data = processor.balance_dataset(data, args.balance)
        print(f"After balancing: {len(data)} examples")
    
    if args.convert_gemma:
        print("Converting to Gemma format...")
        data = processor.convert_to_gemma_format(data)
        print("Converted to Gemma format")
    
    # Save processed data
    processor.save_data(data, args.output_file)
    print(f"Saved processed data to {args.output_file}")
    
    # Show statistics
    if args.stats:
        stats = processor.get_statistics(data)
        print("\nDataset Statistics:")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Average input length: {stats['avg_input_length']:.1f}")
        print(f"Average output length: {stats['avg_output_length']:.1f}")
        
        print("\nScenario types:")
        for scenario_type, count in stats['scenario_types'].items():
            print(f"  {scenario_type}: {count}")
        
        print("\nSources:")
        for source, count in stats['sources'].items():
            print(f"  {source}: {count}")

if __name__ == "__main__":
    main()
