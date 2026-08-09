import os
import json
import logging
from dotenv import load_dotenv

# 1. Load the secret API key from the .env file
load_dotenv() 

from brain.llm_wrapper import LLMWrapper
from brain.structure_planner import StructurePlanner

# Set up pretty logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 2. Initialize the Brain with the REAL API
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY not found in .env file!")
        return

    # use_mock=False forces it to talk to the real Groq/Llama-3 API
    llm = LLMWrapper(api_key=api_key, use_mock=False) 
    planner = StructurePlanner()

    # 3. The User Prompt (Try changing this to whatever you want!)
    user_prompt = "Generate a 2-minute epic cinematic track. Starts very sad and sparse in A minor, then explodes into a massive, heroic orchestral chorus."

    print(f"\n🎤 USER PROMPT: '{user_prompt}'\n")
    print("⏳ Sending to Llama-3 via Groq...")

    # 4. Phase 1: LLM translates text to high-level intent
    llm_intent = llm.generate_intent(user_prompt)
    print("\n--- 🧠 LLM HIGH-LEVEL INTENT (Raw from Llama-3) ---")
    print(json.dumps(llm_intent, indent=2))

    # 5. Phase 2: Structure Planner translates intent to dense math
    print("\n⏳ Passing to Algorithmic Structure Planner...")
    dense_timeline = planner.plan(llm_intent)
    
    print("\n--- 🎼 DENSE ALGORITHMIC TIMELINE (Ready for HarmonyRouter) ---")
    # Just print the first section to avoid flooding the terminal
    print(json.dumps(dense_timeline["timeline"][0], indent=2))
    print(f"... ({len(dense_timeline['timeline'])} total sections generated)")

if __name__ == "__main__":
    main()