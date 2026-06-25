import os
import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# --- UPGRADED PYDANTIC SCHEMAS (The Guardrails) ---
class ModulationIntent(BaseModel):
    target_section: str = Field(description="The section to modulate into, e.g., 'Bridge', 'Final Chorus'")
    target_key: Optional[str] = Field(None, description="The new key root, e.g., 'C', 'G'. Leave null to use relative.")
    target_mode: Optional[str] = Field(None, description="The new mode, e.g., 'major', 'minor'. Leave null to use relative.")

class SectionIntent(BaseModel):
    name: str = Field(description="Section name: Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro")
    mood: str = Field(description="Emotional mood: sad, heroic, happy, melancholy, tension, calm")
    length: str = Field(description="Relative length: 'short' (4 bars), 'medium' (8 bars), or 'long' (16 bars)")

class SongIntent(BaseModel):
    global_key: str = Field(description="ONLY the root note letter, e.g., A, C, F#. Do NOT include 'major'/'minor'")
    global_mode: str = Field(description="strictly 'major' or 'minor'")
    global_bpm: int = Field(description="Tempo in Beats Per Minute, e.g., 120")
    style: str = Field("cinematic", description="Musical style: cinematic, pop, jazz, electronic, classical")
    energy_curve: str = Field("build", description="Overall energy arc: 'build' (starts quiet, ends huge), 'flat' (steady), 'fade' (starts huge, ends quiet)")
    modulations: List[ModulationIntent] = Field(default_factory=list, description="Optional key changes for specific sections")
    sections: List[SectionIntent] = Field(description="Ordered list of song sections")

class LLMWrapper:
    """
    The 'Creative Director'.
    Translates raw human text into strict, mathematically valid JSON blueprints.
    """
    def __init__(self, api_key: str = None, use_mock: bool = False):
        self.use_mock = use_mock
        if not use_mock:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
                self.model = "llama-3.1-8b-instant" # Fast and cheap
            except ImportError:
                logger.warning("Groq not installed. Falling back to Mock LLM.")
                self.use_mock = True

    def _get_mock_response(self, prompt: str) -> dict:
        """Fallback for testing or when the API fails."""
        logger.info("🧠 [Brain] Using Mock LLM fallback.")
        return {
            "global_key": "A", 
            "global_mode": "minor", 
            "global_bpm": 110,
            "style": "cinematic",
            "energy_curve": "build",
            "modulations": [
                {"target_section": "Bridge", "target_key": None, "target_mode": None} # Will resolve to C Major (Relative)
            ],
            "sections": [
                {"name": "Intro", "mood": "sad", "length": "short"},
                {"name": "Verse", "mood": "melancholy", "length": "medium"},
                {"name": "Chorus", "mood": "heroic", "length": "long"},
                {"name": "Bridge", "mood": "tension", "length": "medium"},
                {"name": "Outro", "mood": "calm", "length": "short"}
            ]
        }

    def generate_intent(self, user_prompt: str) -> dict:
        logger.info(f"🧠 [Brain] Analyzing prompt: '{user_prompt}'")

        if self.use_mock:
            raw_json = self._get_mock_response(user_prompt)
        else:
            # --- BULLETPROOF SYSTEM PROMPT ---
            system_prompt = """You are an expert music producer and composer. 
Your task is to convert a user's song description into a STRICT JSON schema.

The JSON MUST exactly match this structure:
{
  "global_key": "string (ONLY the root note letter, e.g., A, C, F#. Do NOT include 'major'/'minor')",
  "global_mode": "string (strictly 'major' or 'minor')",
  "global_bpm": "integer (e.g., 120)",
  "style": "string (cinematic, pop, jazz, electronic, classical)",
  "energy_curve": "string ('build', 'flat', or 'fade')",
  "modulations": [
    {
      "target_section": "string (e.g., 'Bridge')",
      "target_key": "string or null (leave null for relative key)",
      "target_mode": "string or null (leave null for relative key)"
    }
  ],
  "sections": [
    {
      "name": "string (Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro)",
      "mood": "string (sad, heroic, happy, melancholy, tension, calm)",
      "length": "string (short, medium, long)"
    }
  ]
}

CRITICAL RULES:
1. Output ONLY valid JSON. No markdown, no explanations.
2. All strings must be enclosed in double quotes.
3. global_bpm must be an integer.
4. global_key MUST be just the letter (e.g., "A", not "A minor").
"""

            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )
                raw_json = json.loads(completion.choices[0].message.content)
            except Exception as e:
                logger.error(f"❌ [Brain] Groq API failed or returned invalid JSON: {e}")
                logger.info("⚠️ Falling back to Mock LLM to prevent pipeline crash.")
                raw_json = self._get_mock_response(user_prompt)

        # Validate and enforce schema using Pydantic
        try:
            validated_intent = SongIntent(**raw_json)
            logger.info("✅ [Brain] Intent validated successfully.")
            return validated_intent.model_dump()
        except ValidationError as e:
            logger.error(f"❌ [Brain] LLM output failed Pydantic validation: {e}")
            return self._get_mock_response(user_prompt)