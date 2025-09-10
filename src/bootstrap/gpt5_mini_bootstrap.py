#!/usr/bin/env python3
"""
GPT-5 Mini Bootstrap Phase
Generate high-quality conversational examples using GPT-5 Mini
"""

import json
import os
import asyncio
import aiohttp
import hashlib
from openai import AsyncOpenAI
from collections import defaultdict

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

client = AsyncOpenAI(api_key=api_key)
import argparse
import time
import random
from typing import List, Dict, Set
from pathlib import Path

class GPT5MiniBootstrap:
    def __init__(self, api_key: str = None):
        if api_key:
            self.conversation_types = [
                "casual_chat", "deep_lore_discussion", "philosophical_chat", "storytelling",
                "world_exploration", "character_backstory", "mystery_investigation", "emotional_support",
                "humor_and_jokes", "cultural_exchange", "inference_instruction_example"
            ]
            
            # Deduplication tracking
            self.generated_hashes: Set[str] = set()
            self.scenario_cache: Dict[str, List[Dict]] = defaultdict(list)
            
            # Character archetypes and personalities
            self.character_archetypes = [
                {"name": "Wise Elder", "personality": "knowledgeable, patient, speaks in proverbs", "lore_role": "ancient historian"},
                {"name": "Merchant", "personality": "friendly but business-minded, knows about items and trade", "lore_role": "traveling trader"},
                {"name": "Warrior", "personality": "direct, battle-hardened, gives combat advice", "lore_role": "veteran soldier"},
                {"name": "Scholar", "personality": "intellectual, precise, loves sharing knowledge", "lore_role": "researcher"},
                {"name": "Mystic", "personality": "mysterious, spiritual, speaks in riddles", "lore_role": "magic practitioner"},
                {"name": "Rogue", "personality": "sly, street-smart, knows secrets", "lore_role": "information broker"},
                {"name": "Healer", "personality": "caring, gentle, focuses on helping others", "lore_role": "temple priest"},
                {"name": "Adventurer", "personality": "enthusiastic, experienced, loves sharing stories", "lore_role": "fellow explorer"}
            ]
            
            # Game lore elements
            self.lore_elements = [
                "ancient dragon war", "lost kingdom of Eldoria", "crystal mines of the north",
                "shadow realm invasion", "the great magic academy", "undead plague",
                "elemental guardians", "the forbidden forest", "sky cities of the clouds",
                "time-warped ruins", "the merchant's guild", "dwarven mountain halls"
            ]

    def _create_content_hash(self, scenario_type: str, character: Dict, lore_element: str) -> str:
        """Create a hash for deduplication based on scenario parameters"""
        content = f"{scenario_type}_{character['name']}_{lore_element}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_duplicate(self, scenario_type: str, character: Dict, lore_element: str) -> bool:
        """Check if this combination has already been generated"""
        content_hash = self._create_content_hash(scenario_type, character, lore_element)
        return content_hash in self.generated_hashes

    def _mark_as_generated(self, scenario_type: str, character: Dict, lore_element: str):
        """Mark this combination as generated"""
        content_hash = self._create_content_hash(scenario_type, character, lore_element)
        self.generated_hashes.add(content_hash)

    async def generate_conversation_scenarios(self, num_scenarios: int = 500, max_concurrent: int = 3) -> List[Dict]:
        """Generate diverse conversation scenarios using async GPT-5 Mini with deduplication"""

        scenarios = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single_scenario(scenario_id: int):
            """Generate a single scenario with deduplication"""
            async with semaphore:
                # Try to find a unique combination
                attempts = 0
                max_attempts = 50  # Prevent infinite loops
                
                while attempts < max_attempts:
                    scenario_type = random.choice(self.conversation_types)
                    character = random.choice(self.character_archetypes)
                    lore_element = random.choice(self.lore_elements)
                    
                    if not self._is_duplicate(scenario_type, character, lore_element):
                        break
                    attempts += 1
                
                if attempts >= max_attempts:
                    print(f"[WARNING] Could not find unique combination for scenario {scenario_id+1}")
                    return None
                
                prompt = self._create_scenario_prompt(scenario_type, character, lore_element)
                
                # Retry logic for timeouts and errors
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        if retry > 0:
                            print(f"Retrying scenario {scenario_id+1} (attempt {retry+1}/{max_retries})...")
                        else:
                            print(f"Generating scenario {scenario_id+1}/{num_scenarios} ({scenario_type})...")
                        
                        response = await client.chat.completions.create(
                            model="gpt-5-mini",
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"},
                            timeout=60  # Increased timeout to 60 seconds
                        )

                        content = response.choices[0].message.content.strip()
                        scenario = self._parse_scenario_response(content, scenario_type)

                        if scenario:
                            # Add character and lore metadata
                            scenario['character'] = character
                            scenario['lore_element'] = lore_element
                            scenario['scenario_type'] = scenario_type
                            
                            # Mark as generated to avoid duplicates
                            self._mark_as_generated(scenario_type, character, lore_element)
                            
                            print(f"[SUCCESS] Generated scenario {scenario_id+1} ({character['name']} - {lore_element})")
                            return scenario
                        else:
                            print(f"[FAILED] Failed to parse scenario {scenario_id+1}")
                            return None

                    except asyncio.TimeoutError:
                        print(f"[TIMEOUT] Timeout for scenario {scenario_id+1} (attempt {retry+1}/{max_retries})")
                        if retry < max_retries - 1:
                            await asyncio.sleep(2 ** retry)  # Exponential backoff
                            continue
                        else:
                            print(f"[ERROR] Failed scenario {scenario_id+1} after {max_retries} attempts (timeout)")
                            return None
                    except Exception as e:
                        print(f"[ERROR] Error generating scenario {scenario_id+1} (attempt {retry+1}/{max_retries}): {e}")
                        if retry < max_retries - 1:
                            await asyncio.sleep(2 ** retry)  # Exponential backoff
                            continue
                        else:
                            print(f"[ERROR] Failed scenario {scenario_id+1} after {max_retries} attempts")
                            return None
                
                return None

        # Generate scenarios concurrently
        tasks = [generate_single_scenario(i) for i in range(num_scenarios)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        successful_scenarios = 0
        failed_scenarios = 0
        
        for result in results:
            if isinstance(result, dict):
                scenarios.append(result)
                successful_scenarios += 1
            elif isinstance(result, Exception):
                print(f"Task failed with exception: {result}")
                failed_scenarios += 1
            else:
                failed_scenarios += 1
        
        print(f"\nGeneration Summary:")
        print(f"  [SUCCESS] Successful: {successful_scenarios}")
        print(f"  [FAILED] Failed: {failed_scenarios}")
        print(f"  [STATS] Success rate: {successful_scenarios/(successful_scenarios+failed_scenarios)*100:.1f}%")

        return scenarios

    def _create_scenario_prompt(self, scenario_type: str, character: Dict, lore_element: str) -> str:
        """Create prompt for specific scenario type with character and lore context"""

        # Character context
        char_context = f"""You are a {character['name']} - {character['personality']}. 
        Your role in the world is as a {character['lore_role']}. 
        You have knowledge about {lore_element}."""

        prompts = {
            "casual_chat": f"""{char_context}
            
            Generate a natural, flowing conversation between a curious gamer and you. 
            The player should be genuinely interested in learning about the world and having meaningful interactions.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Create 4-6 exchanges that feel organic and engaging, naturally weaving in your knowledge of {lore_element}.
            
            Player message examples (curious gamer style):
            - "Hey there! I've been exploring this area"
            - "What's it like living here?"
            - "I'm curious about this place"
            - "Tell me something interesting"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "deep_lore_discussion": f"""{char_context}
            
            Generate a deep, philosophical conversation about the world's lore and mysteries.
            The player should be genuinely fascinated by the world's history and ask thoughtful questions.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Share profound insights about {lore_element} from your perspective as a {character['lore_role']}.
            
            Player message examples (thoughtful gamer style):
            - "I've been thinking about the deeper meaning behind all this"
            - "What do you think really happened during those ancient times?"
            - "There's something mysterious about this place, isn't there?"
            - "I wonder what the world was like before..."
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "philosophical_chat": f"""{char_context}
            
            Generate a philosophical conversation about life, existence, or deeper meanings.
            The player should be introspective and curious about big questions.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Share wisdom and perspectives related to {lore_element} from your experience as a {character['lore_role']}.
            
            Player message examples (philosophical gamer style):
            - "What do you think the purpose of all this is?"
            - "Do you ever wonder about the nature of existence?"
            - "What's the most important lesson you've learned?"
            - "I've been pondering some deep questions lately"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "storytelling": f"""{char_context}
            
            Generate a conversation where you share stories and tales from your past.
            The player should be an eager listener, asking follow-up questions and showing genuine interest.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Tell engaging stories related to {lore_element} from your perspective as a {character['lore_role']}.
            
            Player message examples (story-loving gamer style):
            - "Tell me a story from your past"
            - "What's the most interesting thing that's happened to you?"
            - "I love hearing about adventures"
            - "What was it like back in the old days?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "world_exploration": f"""{char_context}
            
            Generate a conversation about exploring and discovering the world.
            The player should be adventurous and curious about different places and cultures.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Share knowledge about {lore_element} and the wider world from your perspective as a {character['lore_role']}.
            
            Player message examples (explorer gamer style):
            - "I've been traveling all over this world"
            - "What's the most amazing place you've ever seen?"
            - "I love discovering new areas"
            - "Tell me about the different regions you know"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "character_backstory": f"""{char_context}
            
            Generate a conversation where you share your personal history and background.
            The player should be genuinely interested in learning about you as a person.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Reveal aspects of your past related to {lore_element} from your role as a {character['lore_role']}.
            
            Player message examples (personal gamer style):
            - "Tell me about yourself"
            - "What brought you to this place?"
            - "I'd love to know more about your background"
            - "What's your story?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "mystery_investigation": f"""{char_context}
            
            Generate a conversation about solving mysteries or uncovering secrets.
            The player should be curious and investigative, asking probing questions.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Share mysterious knowledge about {lore_element} from your perspective as a {character['lore_role']}.
            
            Player message examples (detective gamer style):
            - "There's something strange going on here"
            - "I've noticed some odd things lately"
            - "What secrets do you know about this place?"
            - "I'm trying to piece together what happened"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "emotional_support": f"""{char_context}
            
            Generate a conversation where you provide emotional support or comfort.
            The player should be going through something difficult or feeling down.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Offer wisdom and comfort related to {lore_element} from your experience as a {character['lore_role']}.
            
            Player message examples (vulnerable gamer style):
            - "I've been having a tough time lately"
            - "I'm feeling lost and confused"
            - "Can I talk to you about something?"
            - "I need someone to listen"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "humor_and_jokes": f"""{char_context}
            
            Generate a lighthearted conversation with humor and jokes.
            The player should be in a good mood and enjoy playful banter.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Share funny stories or jokes related to {lore_element} from your perspective as a {character['lore_role']}.
            
            Player message examples (cheerful gamer style):
            - "You seem like you have a good sense of humor"
            - "Tell me something funny"
            - "I love a good joke"
            - "What's the funniest thing that's happened to you?"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "cultural_exchange": f"""{char_context}
            
            Generate a conversation about different cultures, traditions, and ways of life.
            The player should be curious about different perspectives and customs.
            You should stay in character as a {character['name']} with your personality: {character['personality']}.
            Share cultural knowledge about {lore_element} from your perspective as a {character['lore_role']}.
            
            Player message examples (culturally curious gamer style):
            - "I'm fascinated by different cultures"
            - "What traditions do you follow?"
            - "How do people live in other parts of the world?"
            - "Tell me about your customs"
            
            Format as JSON with "exchanges" array containing "player" and "npc" fields.""",

            "inference_instruction_example": f"""{char_context}
            
            Generate a conversation that demonstrates how to follow detailed inference instructions.
            The player should speak like a neutral gamer - curious but not overly dramatic.
            Create an example where the instruction is very specific about character behavior and context.
            
            The conversation should show:
            1. How to stay in character with specific personality traits
            2. How to incorporate lore knowledge naturally
            3. How to respond to different types of player questions
            4. How to maintain character consistency throughout the conversation
            
            Player message examples (neutral gamer style):
            - "Tell me about your past"
            - "What do you know about this area?"
            - "Can you help me with something?"
            - "What's your opinion on recent events?"
            
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
            character = scenario.get('character', {})
            lore_element = scenario.get('lore_element', 'general game world')
            scenario_type = scenario.get('type', 'unknown')
            
            for exchange in scenario['exchanges']:
                # Create different instruction styles based on scenario type
                if scenario_type == 'inference_instruction_example':
                    # Create detailed inference-style instruction
                    instruction = f"""You are a {character.get('name', 'NPC')} - {character.get('personality', 'friendly and helpful')}. 
Your role in the world is as a {character.get('lore_role', 'helpful character')}. 
You have deep knowledge about {lore_element}.
Current context: The player is engaging with you in conversation.
Behavior guidelines: Stay completely in character, maintain your personality traits consistently, and naturally incorporate your knowledge of {lore_element} when relevant.
Respond to the player's message in character, staying true to your personality and using your knowledge of the world."""
                else:
                    # Create standard instruction
                    instruction = f"""You are a {character.get('name', 'NPC')} - {character.get('personality', 'friendly and helpful')}. 
Your role in the world is as a {character.get('lore_role', 'helpful character')}. 
You have knowledge about {lore_element}.
Respond to the player's message in character, staying true to your personality and using your knowledge of the world."""
                
                training_data.append({
                    'instruction': instruction,
                    'input': exchange['player'],
                    'output': exchange['npc'],
                    'scenario_type': scenario_type,
                    'character_name': character.get('name', 'Unknown'),
                    'character_personality': character.get('personality', 'Unknown'),
                    'character_role': character.get('lore_role', 'Unknown'),
                    'lore_element': lore_element,
                    'instruction_style': 'inference_detailed' if scenario_type == 'inference_instruction_example' else 'standard',
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

async def main():
    parser = argparse.ArgumentParser(description="GPT-5 Mini Bootstrap Phase")
    parser.add_argument("--num_scenarios", type=int, default=500, help="Number of scenarios to generate")
    parser.add_argument("--output_file", default="data/bootstrap/gpt5_mini_bootstrap.jsonl", help="Output file")
    parser.add_argument("--api_key", help="OpenAI API key")
    parser.add_argument("--max_concurrent", type=int, default=3, help="Maximum concurrent API requests (default: 3, recommended: 2-5)")

    args = parser.parse_args()

    # Initialize bootstrap
    bootstrap = GPT5MiniBootstrap(args.api_key)

    print(f"Starting GPT-5 Mini bootstrap phase...")
    print(f"Generating {args.num_scenarios} conversation scenarios with {args.max_concurrent} concurrent requests...")

    # Generate scenarios
    scenarios = await bootstrap.generate_conversation_scenarios(args.num_scenarios, args.max_concurrent)
    print(f"Generated {len(scenarios)} scenarios")

    # Convert to training format
    training_data = bootstrap.convert_to_training_format(scenarios)
    print(f"Converted to {len(training_data)} training examples")

    # Save data
    bootstrap.save_data(training_data, args.output_file)

    # Show statistics
    scenario_counts = {}
    character_counts = {}
    lore_counts = {}
    instruction_style_counts = {}
    
    for item in training_data:
        scenario_type = item.get('scenario_type', 'unknown')
        character_name = item.get('character_name', 'unknown')
        lore_element = item.get('lore_element', 'unknown')
        instruction_style = item.get('instruction_style', 'unknown')
        
        scenario_counts[scenario_type] = scenario_counts.get(scenario_type, 0) + 1
        character_counts[character_name] = character_counts.get(character_name, 0) + 1
        lore_counts[lore_element] = lore_counts.get(lore_element, 0) + 1
        instruction_style_counts[instruction_style] = instruction_style_counts.get(instruction_style, 0) + 1

    print("\nScenario type distribution:")
    for scenario_type, count in scenario_counts.items():
        print(f"  {scenario_type}: {count} examples")
    
    print("\nCharacter archetype distribution:")
    for character_name, count in character_counts.items():
        print(f"  {character_name}: {count} examples")
    
    print("\nLore element distribution:")
    for lore_element, count in lore_counts.items():
        print(f"  {lore_element}: {count} examples")
    
    print("\nInstruction style distribution:")
    for instruction_style, count in instruction_style_counts.items():
        print(f"  {instruction_style}: {count} examples")

    print(f"\nDeduplication stats:")
    print(f"  Unique combinations generated: {len(bootstrap.generated_hashes)}")
    print(f"  Total possible combinations: {len(bootstrap.conversation_types) * len(bootstrap.character_archetypes) * len(bootstrap.lore_elements)}")

    print("\nGPT-5 Mini bootstrap phase complete!")

if __name__ == "__main__":
    asyncio.run(main())
