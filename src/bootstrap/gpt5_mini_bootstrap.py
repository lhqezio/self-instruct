#!/usr/bin/env python3
"""
GPT-5 Mini Bootstrap Phase
Generate high-quality conversational examples using GPT-5 Mini
"""

import json
import openai
import argparse
import time
import random
from typing import List, Dict
from pathlib import Path

class GPT5MiniBootstrap:
    def __init__(self, api_key: str = None):
        if api_key:
            openai.api_key = api_key
        
        self.conversation_types = [
            "casual_chat", "quest_help", "game_advice", "lore_explanation",
            "item_recommendation", "strategy_tips", "social_interaction", "problem_solving"
        ]
    
    def generate_conversation_scenarios(self, num_scenarios: int = 500) -> List[Dict]:
        """Generate diverse conversation scenarios using GPT-4"""
        
        scenarios = []
        
        for i in range(num_scenarios):
            scenario_type = random.choice(self.conversation_types)
            
            prompt = self._create_scenario_prompt(scenario_type)
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.8
                )
                
                content = response.choices[0].message.content.strip()
                scenario = self._parse_scenario_response(content, scenario_type)
                
                if scenario:
                    scenarios.append(scenario)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating scenario {i}: {e}")
                continue
        
        return scenarios
    
    def _create_scenario_prompt(self, scenario_type: str) -> str:
        """Create prompt for specific scenario type"""
        
        prompts = {
            "casual_chat": """Generate a natural, friendly conversation between a player and an NPC. 
            The NPC should be warm, engaging, and conversational. Create 3-5 exchanges that feel like real chat.
            
            Examples:
            - "Hey there! How's it going?"
            - "What's new around here?"
            - "I'm new to this area, any tips?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",
            
            "quest_help": """Generate a conversation where a player asks an NPC for help with a quest or task.
            The NPC should be knowledgeable, helpful, and conversational (not robotic).
            
            Examples:
            - "I'm stuck on this quest, can you help?"
            - "Where do I find the ancient artifact?"
            - "What should I do next?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",
            
            "game_advice": """Generate a conversation where a player asks for game advice or tips.
            The NPC should give helpful advice in a conversational, friendly way.
            
            Examples:
            - "What's the best way to level up?"
            - "Should I buy this item?"
            - "How do I beat this boss?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",
            
            "lore_explanation": """Generate a conversation where a player asks about game lore, history, or world details.
            The NPC should explain things in an engaging, conversational way.
            
            Examples:
            - "What's the history of this place?"
            - "Tell me about the ancient war"
            - "Who built this castle?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",
            
            "item_recommendation": """Generate a conversation where a player asks for item or equipment recommendations.
            The NPC should give personalized advice in a friendly, conversational tone.
            
            Examples:
            - "What weapon should I use?"
            - "Is this armor good for my class?"
            - "Should I upgrade this item?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",
            
            "strategy_tips": """Generate a conversation where a player asks for strategy or combat tips.
            The NPC should give practical advice in a conversational way.
            
            Examples:
            - "How do I fight this enemy?"
            - "What's the best strategy for this dungeon?"
            - "Any tips for PvP?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",
            
            "social_interaction": """Generate a conversation that's purely social - the NPC and player just chatting.
            The NPC should be friendly, interesting, and have personality.
            
            Examples:
            - "Nice weather today, isn't it?"
            - "I love this game's music"
            - "What's your favorite part of the game?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",
            
            "problem_solving": """Generate a conversation where a player has a problem and the NPC helps solve it.
            The NPC should be patient, helpful, and conversational.
            
            Examples:
            - "I can't figure out this puzzle"
            - "My game keeps crashing"
            - "I'm lost and can't find the exit"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields."""
        }
        
        return prompts.get(scenario_type, prompts["casual_chat"])
    
    def _parse_scenario_response(self, content: str, scenario_type: str) -> Dict:
        """Parse GPT-4 response into scenario format"""
        
        # Clean up response
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
        
        try:
            scenario = json.loads(content)
            if 'exchanges' in scenario and isinstance(scenario['exchanges'], list):
                return {
                    'type': scenario_type,
                    'exchanges': scenario['exchanges']
                }
        except:
            # Fallback: try to extract conversation manually
            lines = content.split('\n')
            exchanges = []
            current_player = ""
            current_npc = ""
            
            for line in lines:
                if 'player' in line.lower() and ':' in line:
                    current_player = line.split(':', 1)[1].strip().strip('"')
                elif 'npc' in line.lower() and ':' in line:
                    current_npc = line.split(':', 1)[1].strip().strip('"')
                    if current_player and current_npc:
                        exchanges.append({
                            'player': current_player,
                            'npc': current_npc
                        })
                        current_player = ""
                        current_npc = ""
            
            if exchanges:
                return {
                    'type': scenario_type,
                    'exchanges': exchanges
                }
        
        return None
    
    def convert_to_training_format(self, scenarios: List[Dict]) -> List[Dict]:
        """Convert scenarios to training format"""
        
        training_data = []
        
        for scenario in scenarios:
            for exchange in scenario['exchanges']:
                training_data.append({
                    'instruction': f"Respond to the player's message in a friendly, conversational way. You are a helpful NPC.",
                    'input': exchange['player'],
                    'output': exchange['npc'],
                    'scenario_type': scenario['type'],
                    'source': 'gpt5_mini_bootstrap'
                })
        
        return training_data
    
    def save_data(self, data: List[Dict], output_file: str):
        """Save data to JSONL file"""
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(data)} examples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="GPT-5 Mini Bootstrap Phase")
    parser.add_argument("--num_scenarios", type=int, default=500, help="Number of scenarios to generate")
    parser.add_argument("--output_file", default="data/bootstrap/gpt5_mini_bootstrap.jsonl", help="Output file")
    parser.add_argument("--api_key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Initialize bootstrap
    bootstrap = GPT5MiniBootstrap(args.api_key)
    
    print(f"Starting GPT-5 Mini bootstrap phase...")
    print(f"Generating {args.num_scenarios} conversation scenarios...")
    
    # Generate scenarios
    scenarios = bootstrap.generate_conversation_scenarios(args.num_scenarios)
    print(f"Generated {len(scenarios)} scenarios")
    
    # Convert to training format
    training_data = bootstrap.convert_to_training_format(scenarios)
    print(f"Converted to {len(training_data)} training examples")
    
    # Save data
    bootstrap.save_data(training_data, args.output_file)
    
    # Show statistics
    scenario_counts = {}
    for item in training_data:
        scenario_type = item.get('scenario_type', 'unknown')
        scenario_counts[scenario_type] = scenario_counts.get(scenario_type, 0) + 1
    
    print("\nScenario type distribution:")
    for scenario_type, count in scenario_counts.items():
        print(f"  {scenario_type}: {count} examples")
    
    print("\nGPT-5 Mini bootstrap phase complete!")

if __name__ == "__main__":
    main()
