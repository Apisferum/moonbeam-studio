import os
import re
import json
import logging
import difflib
import requests
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator
from shared.music_theory_constants import NOTE_MAP

logger = logging.getLogger(__name__)

# =====================================================================
# THE STUDIO PALETTE (The ONLY instruments the LLM is allowed to pick)
# =====================================================================
STUDIO_INSTRUMENTS = [
    "Violin", "Viola", "Cello", "Double Bass",
    "Clarinet", "Flute", "Oboe", "Bassoon",
    "Trumpet", "Trombone", "French Horn", "Tuba",
    "Percussion"
]

# fallback bucket for instruments the fuzzy matcher can't place —
# prevents an LLM-named instrument from silently vanishing from the section
INSTRUMENT_FALLBACK_KEYWORDS = {
    "piano": "Violin", "keyboard": "Violin", "organ": "French Horn",
    "guitar": "Cello", "bass": "Double Bass", "synth": "Violin",
    "marimba": "Percussion", "xylophone": "Percussion", "timpani": "Percussion",
    "vibraphone": "Percussion", "bongo": "Percussion", "conga": "Percussion",
    "tuba": "Tuba", "euphonium": "Tuba",
    "sax": "Clarinet", "oboe": "Oboe", "flute": "Flute",
    "violin": "Violin", "cello": "Cello", "bassoon": "Bassoon",
    "trumpet": "Trumpet", "trombone": "Trombone", "horn": "French Horn",
    "vocal": "Oboe", "voice": "Oboe", "choir": "French Horn",
    "drums": "Percussion", "drum": "Percussion"
}


def _match_studio_instrument(inst_str: str) -> Optional[str]:
    """Exact -> fuzzy -> keyword-bucket match. Returns None only if truly unplaceable."""
    matched = next((i for i in STUDIO_INSTRUMENTS if i.lower() == inst_str.lower()), None)
    if matched:
        return matched
    close = difflib.get_close_matches(inst_str, STUDIO_INSTRUMENTS, n=1, cutoff=0.6)
    if close:
        return close[0]
    low = inst_str.lower()
    for kw, bucket in INSTRUMENT_FALLBACK_KEYWORDS.items():
        if kw in low:
            return bucket
    return None


# =====================================================================
# EXPLICIT MELODY TOKEN FORMAT
# =====================================================================
# 'NOTE OCTAVE:DURATION' using scientific pitch notation (middle C = C4).
# Duration codes below; append '.' for a dotted value, or give a raw beat
# count directly (e.g. "C4:1.5"). "R:<duration>" is a rest. Validation here
# is deliberately lightweight (parseable or not) — the real numeric
# conversion into (octave, pitch_class, duration_beats) happens once, in
# StructurePlanner, which is the single source of truth for it.
_MELODY_DURATION_CODES = {
    "w": 4.0, "h": 2.0, "q": 1.0, "e": 0.5, "s": 0.25,
    "w.": 6.0, "h.": 3.0, "q.": 1.5, "e.": 0.75, "s.": 0.375,
}
_MELODY_TOKEN_RE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+):(.+)$')
_MELODY_REST_RE = re.compile(r'^[Rr]:(.+)$')


def _melody_duration_is_valid(token: str) -> bool:
    token = token.strip()
    if token in _MELODY_DURATION_CODES:
        return True
    try:
        return float(token) > 0
    except ValueError:
        return False


def _melody_token_is_valid(token: str) -> bool:
    s = str(token).strip()
    rest_match = _MELODY_REST_RE.match(s)
    if rest_match:
        return _melody_duration_is_valid(rest_match.group(1))
    m = _MELODY_TOKEN_RE.match(s)
    if not m:
        return False
    letter, accidental, _octave_str, dur_token = m.groups()
    if (letter.upper() + accidental) not in NOTE_MAP:
        return False
    return _melody_duration_is_valid(dur_token)


# --- PYDANTIC SCHEMAS ---
class ModulationIntent(BaseModel):
    target_section: str = Field(description="The section to modulate into, e.g., 'Bridge', 'Final Chorus'")
    target_key: Optional[str] = Field(None, description="The new key root, e.g., 'C', 'G'. Leave null to use relative.")
    target_mode: Optional[str] = Field(None, description="The new mode, e.g., 'major', 'minor'. Leave null to use relative.")


class SectionIntent(BaseModel):
    name: str = Field(description="Section name: Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro")
    mood: str = Field(description="Emotional mood: sad, heroic, happy, melancholy, tension, calm")
    length: str = Field(description="Relative length: 'short' (4 bars), 'medium' (8 bars), or 'long' (16 bars)")
    instruments: List[str] = Field(default_factory=list, description=f"List of instruments for this section. MUST ONLY use: {', '.join(STUDIO_INSTRUMENTS)}")
    explicit_chords: Optional[List[str]] = Field(
        None,
        description="Optional user-specified chord progression for THIS section, e.g. ['Cmaj7','Am7','Dm7','G7']. "
                    "When present, this OVERRIDES automatic harmonic generation for the section. Only fill this in "
                    "when the user's request names actual chords, a roman-numeral progression, or a specific "
                    "cadence/loop they want played — resolve roman numerals or key-relative language (e.g. "
                    "'ii-V-i in D minor') into concrete chord symbols yourself (['Em7b5','A7','Dm7']). Leave this "
                    "empty/omitted for any section where the user only described a mood or style."
    )
    explicit_chord_durations: Optional[List[float]] = Field(
        None,
        description="Optional per-chord duration in BEATS, same length and order as explicit_chords. Omit to "
                    "default to one chord per bar. Only set this if the user specified uneven chord lengths "
                    "(e.g. 'hold the first chord for two bars')."
    )
    explicit_melody: Optional[List[str]] = Field(
        None,
        description="Optional user-specified melody/riff for this section's lead instrument, as a list of "
                    "'NOTE OCTAVE:DURATION' tokens in scientific pitch notation (middle C = C4), e.g. "
                    "['C4:q','E4:q','G4:h']. Duration codes: w=whole(4 beats), h=half(2), q=quarter(1), "
                    "e=eighth(0.5), s=sixteenth(0.25); append '.' for a dotted value (e.g. 'q.'=1.5 beats), or "
                    "give a raw beat count directly (e.g. 'C4:1.5'). Use 'R:<duration>' for a rest. When "
                    "present, this OVERRIDES automatic lead-voice generation for the section and loops to fill "
                    "it if the melody is shorter than the section. Only fill this in when the user actually "
                    "spells out, hums in words, or names real notes for a melody/hook — leave it empty/omitted "
                    "for any section where the user only described a mood or style."
    )
    riff: Optional[bool] = Field(
        False,
        description="Set true when the user wants a short repeating ostinato/riff/groove for the lead line "
                    "instead of a lyrical, evolving melodic phrase (e.g. 'a driving synth riff', 'looping "
                    "arpeggio hook'). Only relevant when explicit_melody is NOT set — it changes how the "
                    "automatic lead-voice generator shapes its phrase (short repeating cell vs. an evolving "
                    "arc). Leave false for anything melodic/song-like."
    )
    solo_instrument: Optional[str] = Field(
        None,
        description="Optional: name one instrument from this section's `instruments` list to feature as a "
                    "foreground SOLO for this section (e.g. a violin solo, a sax solo, a trumpet solo). MUST "
                    "be exactly one of the names already listed in this section's own `instruments`. When set, "
                    "that instrument becomes the lead voice and the other instruments thin out to accompany it, "
                    "instead of everyone playing at equal prominence. Leave null unless the user actually asks "
                    "for a solo or featured instrumental line."
    )

    @field_validator('instruments', mode='before')
    @classmethod
    def enforce_section_instruments(cls, v):
        if not v:
            return []
        valid = []
        for inst in v:
            matched = _match_studio_instrument(str(inst).strip())
            if matched:
                valid.append(matched)
            else:
                logger.warning(f"⚠️ [Brain] Dropping unrecognized instrument: '{inst}'")
        return valid

    @field_validator('explicit_chords', mode='before')
    @classmethod
    def validate_explicit_chords(cls, v):
        if not v:
            return None
        valid = []
        for symbol in v:
            s = str(symbol).strip()
            root_str = s[:2] if len(s) >= 2 and s[1] in ('#', 'b') else s[:1]
            if root_str not in NOTE_MAP:
                logger.warning(f"⚠️ [Brain] Dropping unparseable chord symbol: '{symbol}' (unrecognized root note).")
                continue
            valid.append(s)
        # Falls back to None (not []) when everything gets dropped, so
        # StructurePlanner's `section.get("explicit_chords") or []` check
        # correctly treats "user gave garbage" the same as "user gave
        # nothing" and falls through to automatic generation instead of
        # trying to build a progression out of zero valid chords.
        return valid or None

    @field_validator('explicit_chord_durations', mode='before')
    @classmethod
    def validate_explicit_chord_durations(cls, v):
        if not v:
            return None
        try:
            durs = [float(x) for x in v]
        except (TypeError, ValueError):
            logger.warning("⚠️ [Brain] Dropping unparseable explicit_chord_durations (non-numeric entries).")
            return None
        if any(d <= 0 for d in durs):
            logger.warning("⚠️ [Brain] Dropping explicit_chord_durations containing non-positive values.")
            return None
        return durs

    @field_validator('explicit_melody', mode='before')
    @classmethod
    def validate_explicit_melody(cls, v):
        if not v:
            return None
        valid = []
        for tok in v:
            if _melody_token_is_valid(tok):
                valid.append(str(tok).strip())
            else:
                logger.warning(f"⚠️ [Brain] Dropping unparseable melody token: '{tok}'")
        return valid or None

    @field_validator('solo_instrument', mode='before')
    @classmethod
    def validate_solo_instrument(cls, v):
        if not v:
            return None
        matched = _match_studio_instrument(str(v).strip())
        if not matched:
            logger.warning(f"⚠️ [Brain] Dropping unrecognized solo_instrument: '{v}'")
            return None
        return matched


SUPPORTED_TIME_SIGNATURES = ["4/4", "3/4", "2/4", "6/8", "9/8", "12/8"]


class SongIntent(BaseModel):
    global_key: str = Field(description="ONLY the root note letter, e.g., A, C, F#. Do NOT include 'major'/'minor'")
    global_mode: str = Field(description="strictly 'major' or 'minor'")
    global_bpm: int = Field(description="Tempo in Beats Per Minute, e.g., 120")
    global_time_signature: str = Field(
        "4/4",
        description=f"Time signature as 'numerator/denominator'. MUST be exactly one of: {', '.join(SUPPORTED_TIME_SIGNATURES)}. "
                    f"Use 3/4 for waltzes and minuets, 6/8 or 9/8 for compound/lilting meters, 4/4 as the default for pop/cinematic."
    )
    style: str = Field("cinematic", description="Musical style: cinematic, pop, jazz, electronic, classical, orchestral, symphonic")
    lead_instrument: str = Field("Violin", description=f"The primary instrument. MUST be exactly one of: {', '.join(STUDIO_INSTRUMENTS)}")
    energy_curve: str = Field("build", description="Overall energy arc: 'build', 'flat', or 'fade'")
    modulations: List[ModulationIntent] = Field(default_factory=list)
    sections: List[SectionIntent] = Field(description="Ordered list of song sections")

    @field_validator('lead_instrument', mode='before')
    @classmethod
    def enforce_lead_instrument(cls, v: str) -> str:
        if not v:
            return "Violin"
        matched = _match_studio_instrument(v)
        return matched or "Violin"

    @field_validator('global_time_signature', mode='before')
    @classmethod
    def enforce_time_signature(cls, v: str) -> str:
        if not v:
            return "4/4"
        v = str(v).strip()
        return v if v in SUPPORTED_TIME_SIGNATURES else "4/4"


# =====================================================================
# GEMINI JSON SCHEMA (Grammar-Constrained Decoding)
# =====================================================================
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "global_key": {"type": "STRING"},
        "global_mode": {"type": "STRING", "enum": ["major", "minor"]},
        "global_bpm": {"type": "INTEGER"},
        "global_time_signature": {"type": "STRING", "enum": ["4/4", "3/4", "2/4", "6/8", "9/8", "12/8"]},
        "style": {"type": "STRING"},
        "lead_instrument": {"type": "STRING", "enum": STUDIO_INSTRUMENTS},
        "energy_curve": {"type": "STRING", "enum": ["build", "flat", "fade"]},
        "modulations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "target_section": {"type": "STRING"},
                    "target_key": {"type": "STRING", "nullable": True},
                    "target_mode": {"type": "STRING", "nullable": True},
                },
                "required": ["target_section"],
            },
        },
        "sections": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "mood": {"type": "STRING"},
                    "length": {"type": "STRING", "enum": ["short", "medium", "long"]},
                    "instruments": {
                        "type": "ARRAY",
                        "items": {"type": "STRING", "enum": STUDIO_INSTRUMENTS}
                    },
                    "explicit_chords": {
                        "type": "ARRAY", "items": {"type": "STRING"}, "nullable": True
                    },
                    "explicit_chord_durations": {
                        "type": "ARRAY", "items": {"type": "NUMBER"}, "nullable": True
                    },
                    "explicit_melody": {
                        "type": "ARRAY", "items": {"type": "STRING"}, "nullable": True
                    },
                    "riff": {"type": "BOOLEAN", "nullable": True},
                    "solo_instrument": {"type": "STRING", "nullable": True},
                },
                "required": ["name", "mood", "length"],
            },
        },
    },
    "required": ["global_key", "global_mode", "global_bpm", "lead_instrument", "sections"],
}

SYSTEM_PROMPT = f"""You are an expert music producer and composer.
Convert the user's song description into a STRICT JSON object matching the given schema.

CRITICAL RULES:
1. Output ONLY valid JSON. No markdown, no explanations, no code fences.
2. global_key MUST be just the root letter (e.g., "A", not "A minor").
3. global_bpm must be an integer.
4. INSTRUMENT CONSTRAINT: You MUST ONLY choose instruments from this exact list:
   {', '.join(STUDIO_INSTRUMENTS)}.
   If the user asks for an instrument NOT on this list (e.g., 'Banjo', 'Theremin'), map it to the closest valid equivalent (e.g., 'Acoustic Guitar', 'Synth Lead').
5. Always produce at least 3 sections (e.g. Intro, Verse, Chorus).
6. For each section, provide a list of 'instruments' that play in that section to create a dynamic arrangement.
7. TIME SIGNATURE: global_time_signature must be exactly one of 4/4, 3/4, 2/4, 6/8, 9/8, 12/8.
   Default to 4/4 unless the request implies otherwise (e.g. "waltz" -> 3/4, "minuet" -> 3/4,
   "lullaby"/"jig" -> 6/8).
8. SYMPHONIC / ORCHESTRAL / CHORAL REQUESTS: if the user asks for a symphony, orchestral piece,
   choral work, or classical form, set style to "orchestral", "symphonic", or "classical", and you
   may use classical section names instead of Intro/Verse/Chorus: Exposition, Development,
   Recapitulation, Theme, Variation I/II/III/IV, Rondo A/B/C, Movement, Coda. Give each such
   section at least 2 instruments so the arrangement can spread parts across registers rather
   than doubling in unison.
9. EXPLICIT CHORDS: if the user names actual chords, a specific progression, or a roman-numeral
   cadence for a section (e.g. "verse should go Cmaj7 - Am7 - Dm7 - G7", "use a ii-V-i in Bb minor",
   "loop a I-V-vi-IV"), resolve any roman-numeral or key-relative language into concrete chord
   symbols yourself and put them in that section's explicit_chords list, one string per chord, in
   order (e.g. "ii-V-i in D minor" -> ["Em7b5","A7","Dm7"]). Only set explicit_chord_durations if
   the user specified uneven chord lengths (e.g. "hold that first chord twice as long") — otherwise
   leave it null and one chord will default to one bar. PRESERVE inversions and extensions exactly as
   given — do not simplify "F/A" to "F" or "Dm(add9)" to "Dm"; write slash chords as "F/A" (root
   chord, slash, bass note) and keep 9/11/13/add9/sus qualities attached to the root (e.g. "Gm9",
   "A7sus4", "Dm(add9)" -> "Dmadd9"). Do NOT populate explicit_chords for a
   section where the user only described a mood, genre, or style rather than actual chords — leave
   it null/omitted so that section keeps automatic harmonic generation.
10. EXPLICIT MELODY: if the user spells out, hums-in-words, or names an actual note sequence for a
    section's lead line (e.g. "the hook should go C4 E4 G4 C5", "melody: quarter note C4, quarter E4,
    half G4"), put it in that section's explicit_melody as a list of 'NOTE OCTAVE:DURATION' tokens
    (e.g. ["C4:q","E4:q","G4:h"]), using scientific pitch notation (middle C = C4) and duration codes
    w/h/q/e/s (append '.' for dotted). Use 'R:<duration>' for a rest. Leave this empty/omitted unless
    the user actually specified real notes — do not invent a melody into this field.
11. RIFF / SOLO: set riff=true when the user wants a short repeating groove/ostinato instead of an
    evolving melodic phrase for a section's lead line. Set solo_instrument to feature one instrument
    as a foreground solo (that name MUST already appear in this section's own instruments list) — the
    rest of the arrangement will automatically thin to accompany it. Leave both at their defaults
    unless the user actually asks for a riff or a featured solo.
"""


def _log_http_error_body(resp: requests.Response, tier_name: str) -> None:
    """
    requests.Response.raise_for_status() only produces a generic
    '<code> Client Error: <reason> for url: <url>' string — it never
    includes the response body. That body is where the provider's actual
    'invalid_request_error' / unsupported-parameter / bad-schema message
    lives, so we log it explicitly before raising.
    """
    if resp.ok:
        return
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    logger.error(f"❌ [Brain] {tier_name} HTTP {resp.status_code} error body: {body}")


class LLMWrapper:
    """
    Routing (in order, first success wins):
      1. Grok 4.5           - xAI flagship (api.x.ai)
      2. Gemini 3.5 Flash   - native schema-constrained decoding (Google AI Studio)
      3. Qwen 3.7 Plus      - via Fireworks (OpenAI-compatible)
      4. Kimi K2.7 Code     - via Fireworks (OpenAI-compatible)
      5. GLM-5.2            - via Fireworks (OpenAI-compatible)
      6. Rule-based extractor - deterministic, always succeeds, zero network.

    NOTE ON KAGGLE: os.environ.get(...) does NOT pick up Kaggle secrets automatically.
    Kaggle's "Add-ons > Secrets" panel requires an explicit pull, e.g. in your notebook
    cell (not this module):

        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        os.environ["GROK_API_KEY"]   = secrets.get_secret("GROK_API_KEY")
        os.environ["GEMINI_API_KEY"] = secrets.get_secret("GEMINI_API_KEY")
        os.environ["FIREWORKS_API_KEY"] = secrets.get_secret("FIREWORKS_API_KEY")

    If you skip this, every key stays None, every tier is silently skipped, and you'll
    always land on the rule-based fallback with no error raised.

    NOTE ON DEBUGGING 400s: every _call_* method below logs the provider's raw
    error body via _log_http_error_body() before raise_for_status() fires. Check
    the logs for a line like "❌ [Brain] Grok 4.5 HTTP 400 error body: {...}" —
    that JSON tells you exactly which param/field the provider rejected. Without
    it you only ever see the generic "400 Client Error: Bad Request for url: ..."
    which carries zero diagnostic information.
    """

    def __init__(
        self,
        grok_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        fireworks_api_key: Optional[str] = None,
        fireworks_api_base: Optional[str] = None,
        qwen_model: str = "accounts/fireworks/models/qwen3p7-plus",
        kimi_model: str = "accounts/fireworks/models/kimi-k2p7-code",
        glm_model: str = "accounts/fireworks/models/glm-5p2",
        use_mock: bool = False,
    ):
        self.use_mock = use_mock
        self.grok_api_key = grok_api_key or os.environ.get("GROK_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        # Fireworks hosts Qwen, Kimi, and GLM behind one OpenAI-compatible endpoint/key
        self.fireworks_api_key = fireworks_api_key or os.environ.get("FIREWORKS_API_KEY")
        self.fireworks_api_base = fireworks_api_base or os.environ.get(
            "FIREWORKS_API_BASE", "https://api.fireworks.ai/inference/v1/chat/completions"
        )
        self.qwen_model = qwen_model
        self.kimi_model = kimi_model
        self.glm_model = glm_model

    def generate_intent(self, user_prompt: str) -> dict:
        logger.info(f"🧠 [Brain] Analyzing prompt: '{user_prompt}'")

        if self.use_mock:
            raw = self._rule_based_fallback(user_prompt)
            return self._validate(raw, user_prompt)

        tiers = [
            ("Grok 4.5", self.grok_api_key, self._call_grok),
            ("Gemini 3.5 Flash", self.gemini_api_key, self._call_gemini),
            ("Qwen 3.7 Plus", self.fireworks_api_key, lambda p: self._call_fireworks(p, self.qwen_model, "Qwen 3.7 Plus")),
            ("Kimi K2.7 Code", self.fireworks_api_key, lambda p: self._call_fireworks(p, self.kimi_model, "Kimi K2.7 Code")),
            ("GLM-5.2", self.fireworks_api_key, lambda p: self._call_fireworks(p, self.glm_model, "GLM-5.2")),
        ]

        for name, key, call_fn in tiers:
            if not key:
                logger.info(f"⏭️ [Brain] Skipping {name} — no API key configured.")
                continue
            try:
                raw = call_fn(user_prompt)
                validated = self._validate(raw, user_prompt, raise_on_fail=True)
                logger.info(f"✅ [Brain] {name} produced a valid intent.")
                return validated
            except Exception as e:
                logger.warning(f"⚠️ [Brain] {name} tier failed: {e}")

        logger.info("🔧 [Brain] All LLM tiers failed or missing keys - using rule-based extractor.")
        raw = self._rule_based_fallback(user_prompt)
        return self._validate(raw, user_prompt)

    def _call_grok(self, user_prompt: str) -> dict:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.grok_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "grok-4.5",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        _log_http_error_body(resp, "Grok 4.5")
        resp.raise_for_status()
        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(f"Grok API error: {data}")
        return json.loads(data["choices"][0]["message"]["content"])

    def _call_gemini(self, user_prompt: str) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={self.gemini_api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_RESPONSE_SCHEMA,
                "temperature": 0.7,
            },
        }
        resp = requests.post(url, json=payload, timeout=300)
        _log_http_error_body(resp, "Gemini 3.5 Flash")
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)

    def _call_fireworks(self, user_prompt: str, model_id: str, tier_name: str = "Fireworks") -> dict:
        """Shared caller for Qwen / Kimi / GLM — all served OpenAI-compatible via Fireworks."""
        headers = {"Authorization": f"Bearer {self.fireworks_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
        }
        resp = requests.post(self.fireworks_api_base, headers=headers, json=payload, timeout=300)
        _log_http_error_body(resp, tier_name)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return json.loads(text)

    def _rule_based_fallback(self, user_prompt: str) -> dict:
        p = user_prompt.lower()

        instrument_keywords = {
            "violin": "Violin", "cello": "Cello", "viola": "Viola", "double bass": "Double Bass",
            "bass": "Double Bass", "clarinet": "Clarinet", "flute": "Flute", "oboe": "Oboe",
            "bassoon": "Bassoon", "trumpet": "Trumpet", "trombone": "Trombone", "horn": "French Horn",
            "tuba": "Tuba", "guitar": "Cello", "piano": "Violin", "synth": "Violin", "drums": "Percussion"
        }
        lead_instrument = "Violin"
        for kw, name in instrument_keywords.items():
            if kw in p:
                lead_instrument = name
                break

        mood_map = {
            "sad": ["sad", "heartbreak", "grief", "loss", "mourn"],
            "melancholy": ["melancholy", "wistful", "bittersweet", "nostalgic"],
            "heroic": ["heroic", "epic", "triumphant", "victory", "battle"],
            "happy": ["happy", "joy", "upbeat", "cheerful", "fun"],
            "tension": ["tense", "tension", "dark", "suspense", "dread", "anxious"],
            "calm": ["calm", "peaceful", "relax", "ambient", "gentle", "soft"],
        }
        primary_mood = "cinematic"
        for mood, kws in mood_map.items():
            if any(k in p for k in kws):
                primary_mood = mood
                break

        global_mode = "minor" if primary_mood in ("sad", "melancholy", "tension") else "major"
        if "minor" in p:
            global_mode = "minor"
        elif "major" in p:
            global_mode = "major"

        # Fixed: prefer "key of X" / "X major|minor" patterns before a bare capital
        # letter, so sentence-initial words like "A ballad..." don't hijack the match.
        key_match = (
            re.search(r"\bkey of\s+([A-G])(#|b)?\s*(major|minor)?", user_prompt, re.IGNORECASE)
            or re.search(r"\b([A-G])(#|b)?\s+(major|minor)\b", user_prompt)
        )
        global_key = (key_match.group(1).upper() + (key_match.group(2) or "")) if key_match else "C"

        bpm_match = re.search(r"(\d{2,3})\s*bpm", p)
        if bpm_match:
            global_bpm = int(bpm_match.group(1))
        elif any(w in p for w in ["slow", "ballad", "gentle"]):
            global_bpm = 75
        elif any(w in p for w in ["fast", "upbeat", "energetic", "driving"]):
            global_bpm = 140
        else:
            global_bpm = 110

        style_map = {"jazz": "jazz", "electronic": "electronic", "edm": "electronic",
                     "classical": "classical", "pop": "pop", "cinematic": "cinematic", "film": "cinematic"}
        style = "cinematic"
        for kw, name in style_map.items():
            if kw in p:
                style = name
                break

        if any(w in p for w in ["build", "rise", "grows", "climax"]):
            energy_curve = "build"
        elif any(w in p for w in ["fade", "wind down", "quiet ending", "decrescendo"]):
            energy_curve = "fade"
        else:
            energy_curve = "flat" if primary_mood == "calm" else "build"

        return {
            "global_key": global_key, "global_mode": global_mode, "global_bpm": global_bpm,
            "style": style, "lead_instrument": lead_instrument, "energy_curve": energy_curve,
            "modulations": [],
            "sections": [
                {"name": "Intro", "mood": primary_mood, "length": "short", "instruments": [lead_instrument]},
                {"name": "Verse", "mood": primary_mood, "length": "medium", "instruments": [lead_instrument, "Double Bass"]},
                {"name": "Chorus", "mood": "heroic" if primary_mood != "calm" else "calm", "length": "long", "instruments": [lead_instrument, "Double Bass", "Cello", "Percussion"]},
                {"name": "Outro", "mood": "calm", "length": "short", "instruments": [lead_instrument]},
            ],
        }

    def _validate(self, raw_json: dict, user_prompt: str, raise_on_fail: bool = False) -> dict:
        try:
            validated = SongIntent(**raw_json)
            return validated.model_dump()
        except ValidationError as e:
            logger.error(f"❌ [Brain] Schema validation failed: {e}")
            if raise_on_fail:
                raise
            return SongIntent(**self._rule_based_fallback(user_prompt)).model_dump()