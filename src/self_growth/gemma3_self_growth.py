#!/usr/bin/env python3
"""
Gemma 3 1B Self-Growth Phase
Use Gemma 3 1B to iteratively expand the dataset using Self-Instruct principles
"""

import json
import argparse
import random
import os
from typing import List, Dict
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gemma3SelfGrowth:
    def __init__(self, model_name: str = "google/gemma-2-1b-it", device: str = "auto", hf_token: str = None):
        """Initialize Gemma 3 1B model for self-growth"""
        
        self.model_name = model_name
        self.device = device
        
        # Get HF token from parameter or environment
        if hf_token:
            self.hf_token = hf_token
        else:
            self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.hf_token:
            print("Warning: No Hugging Face token provided. Some models may require authentication.")
            print("Set HUGGINGFACE_TOKEN environment variable or pass --hf_token")
        
        print(f"Loading {model_name}...")
        
        # Load tokenizer with token
        tokenizer_kwargs = {}
        if self.hf_token:
            tokenizer_kwargs["token"] = self.hf_token
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Load model with token
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device
        }
        if self.hf_token:
            model_kwargs["token"] = self.hf_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_new_conversations(self, existing_data: List[Dict], num_new: int = 1000) -> List[Dict]:
        """Generate new conversations based on existing data"""
        
        new_conversations = []
        
        # Sample existing data for prompting
        sample_size = min(50, len(existing_data))
        sample_data = random.sample(existing_data, sample_size)
        
        for i in range(num_new):
            # Create prompt from existing data
            prompt = self._create_generation_prompt(sample_data)
            
            # Generate new conversation
            new_conversation = self._generate_conversation(prompt)
            
            if new_conversation:
                new_conversation['source'] = 'gemma3_self_growth'
                new_conversations.append(new_conversation)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_new} conversations...")
        
        return new_conversations
    
    def _create_generation_prompt(self, sample_data: List[Dict]) -> str:
        """Create prompt for generating new conversations"""
        
        # Select random examples
        examples = random.sample(sample_data, min(5, len(sample_data)))
        
        prompt = "Generate a new conversational exchange between a player and an NPC. Here are some examples:\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Player: {example['input']}\n"
            prompt += f"NPC: {example['output']}\n\n"
        
        prompt += "Now generate a new, different conversation:\n"
        prompt += "Player: "
        
        return prompt
    
    def _generate_conversation(self, prompt: str) -> Dict:
        """Generate a single conversation using Gemma"""
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract conversation
            conversation = self._extract_conversation(response, prompt)
            
            return conversation
            
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return None
    
    def _extract_conversation(self, response: str, prompt: str) -> Dict:
        """Extract conversation from model response"""
        
        # Remove the prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Split into lines and find player/NPC exchanges
        lines = response.split('\n')
        
        player_input = ""
        npc_output = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("Player:"):
                player_input = line.replace("Player:", "").strip()
            elif line.startswith("NPC:"):
                npc_output = line.replace("NPC:", "").strip()
            elif line and not player_input:
                # If no "Player:" prefix, assume first line is player input
                player_input = line
            elif line and player_input and not npc_output:
                # If no "NPC:" prefix, assume second line is NPC output
                npc_output = line
        
        if player_input and npc_output:
            return {
                'instruction': f"Respond to the player's message in a friendly, conversational way. You are a helpful NPC.",
                'input': player_input,
                'output': npc_output,
                'scenario_type': 'gemma_generated'
            }
        
        return None
    
    def iterative_growth(self, initial_data: List[Dict], growth_rounds: int = 3) -> List[Dict]:
        """Perform iterative self-growth"""
        
        all_data = initial_data.copy()
        
        for round_num in range(growth_rounds):
            print(f"\nSelf-growth round {round_num + 1}/{growth_rounds}")
            print(f"Current dataset size: {len(all_data)}")
            
            # Generate new data based on current dataset
            new_data = self.generate_new_conversations(all_data, num_new=1000)
            
            # Add new data to dataset
            all_data.extend(new_data)
            
            print(f"Generated {len(new_data)} new conversations")
            print(f"Total dataset size: {len(all_data)}")
        
        return all_data
    
    def save_data(self, data: List[Dict], output_file: str):
        """Save data to JSONL file"""
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(data)} examples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Gemma 3 1B Self-Growth Phase")
    parser.add_argument("--input_file", required=True, help="Input data file from bootstrap phase")
    parser.add_argument("--output_file", default="data/self_growth/gemma3_expanded.jsonl", help="Output file")
    parser.add_argument("--growth_rounds", type=int, default=3, help="Number of growth rounds")
    parser.add_argument("--model_name", default="google/gemma-3-1b-it", help="Gemma 3 1B model name")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--hf_token", help="Hugging Face token for model access")
    
    args = parser.parse_args()
    
    # Load initial data
    print(f"Loading initial data from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        initial_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(initial_data)} initial examples")
    
    # Initialize self-growth
    self_growth = Gemma3SelfGrowth(args.model_name, args.device, args.hf_token)
    
    print(f"Starting Gemma 3 1B self-growth phase...")
    print(f"Growth rounds: {args.growth_rounds}")
    
    # Perform iterative growth
    expanded_data = self_growth.iterative_growth(initial_data, args.growth_rounds)
    
    # Save expanded data
    self_growth.save_data(expanded_data, args.output_file)
    
    print(f"\nSelf-growth phase complete!")
    print(f"Final dataset size: {len(expanded_data)} examples")

if __name__ == "__main__":
    main()
