import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =====================================================================
# TOKENIZER RESOLUTION CONSTANTS (SINGLE SOURCE OF TRUTH)
# =====================================================================
# Moonbeam's MusicTokenizer operates at a 10ms resolution (inherited from anticipation).
# 1 second = 100 ticks. This MUST match the TIME_RESOLUTION default in MusicTokenizer.
# If this changes, SoftRefiner's boundary clipper and AgenticComposer's tick math will break.
TICKS_PER_SECOND = 100
TIME_RESOLUTION_MS = 10 

DRUM_INSTRUMENT_IDS = {128, 129, 130}

NOTE_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

PC_TO_NAME = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F', 
    6: 'F#', 7: 'G', 8: 'Ab', 9: 'A', 10: 'Bb', 11: 'B'
}

SCALE_MODES = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
    "lydian":         [0, 2, 4, 6, 7, 9, 11],
    "phrygian":       [0, 1, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "locrian":        [0, 1, 3, 5, 6, 8, 10]
}

CHORD_INTERVALS = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "dim": [0, 3, 6],
    "aug": [0, 4, 8], "sus2": [0, 2, 7], "sus4": [0, 5, 7],
    "min9": [0, 3, 7, 10, 2], "maj9": [0, 4, 7, 11, 2], "9": [0, 4, 7, 10, 2]
}

# EXHAUSTIVE GM FAMILIES (Covers 0-127 + Drums)
INSTRUMENT_FAMILIES = {
    "Piano": list(range(0, 8)), 
    "Chromatic_Percussion": list(range(8, 16)),
    "Organ": list(range(16, 24)), 
    "Guitar": list(range(24, 32)),
    "Bass": list(range(32, 40)), 
    "Strings": list(range(40, 48)),
    "Ensemble": list(range(48, 56)), 
    "Brass": list(range(56, 64)),
    "Reed": list(range(64, 72)), 
    "Pipe": list(range(72, 80)),
    "Synth_Lead": list(range(80, 88)), 
    "Synth_Pad": list(range(88, 96)),
    "Sound_Effects": list(range(96, 104)),
    "Ethnic": list(range(104, 112)),
    "Percussive": list(range(112, 120)),
    "Sound_Effects_2": list(range(120, 128)),
    "Percussion": list(range(128, 130))
}
# NOTE ON "Strings" vs get_program("Strings"): this family bucket (40-47)
# is real-GM-accurate — Violin/Viola/Cello/Contrabass/Tremolo/Pizzicato/
# Harp/Timpani. Real GM puts "String Ensemble 1" (program 48) in the
# ENSEMBLE family (48-55) instead, which is where it sits here too. The
# wrinkle is that gm_instruments.get_program("Strings") resolves the
# generic word "Strings" to 48 (String Ensemble) rather than anything in
# this 40-47 bucket, since "a string section" more naturally means
# "ensemble" than "solo violin." That's not a bug in either dict — it's
# just why HardScorer's instrument-match check deliberately unions this
# family bucket with the exact get_program() result rather than trusting
# either alone. Don't "fix" this range to include 48; that would misclassify
# it away from Ensemble, which is where real GM puts it.

HARMONIC_FUNCTIONS_MAJOR = {
    "T": ["I", "vi", "iii"], "S": ["IV", "ii"], "D": ["V", "vii°"]
}
HARMONIC_FUNCTIONS_MINOR = {
    "T": ["i", "III", "VI"], "S": ["iv", "ii°"], "D": ["v", "V", "vii°"]
}

MACRO_SKELETONS = {
    "Intro":      ["T", "T", "T", "T"],
    "Verse":      ["T", "T", "S", "S", "T", "T", "D", "D"],
    "Pre-Chorus": ["S", "S", "D", "D"],
    "Chorus":     ["S", "S", "T", "T", "S", "S", "D", "T"],
    "Bridge":     ["S", "D", "S", "D", "T", "T", "D", "D"],
    "Outro":      ["T", "S", "D", "T"]
}

MAJOR_QUALITIES = {"I": "maj7", "ii": "min7", "iii": "min7", "IV": "maj7", "V": "7", "vi": "min7", "vii°": "dim"}
MINOR_QUALITIES = {"i": "min7", "ii°": "dim", "III": "maj7", "iv": "min7", "v": "min7", "V": "7", "VI": "maj7", "vii°": "dim"}

ARRANGEMENT_PROFILES = {
    "build": {
        "Intro": ["Violin"], "Verse": ["Violin", "Cello"], "Pre-Chorus": ["Violin", "Cello", "Double Bass"],
        "Chorus": ["Violin", "Cello", "French Horn", "Percussion"], "Bridge": ["Cello", "Clarinet"], "Outro": ["Violin", "Cello"]
    },
    "sparse_to_dense": {
        "Intro": ["Violin"], "Verse": ["Violin", "Double Bass"], "Pre-Chorus": ["Violin", "Double Bass", "Cello"],
        "Chorus": ["Violin", "Double Bass", "Cello", "Percussion"], "Bridge": ["Violin", "Cello"], "Outro": ["Violin"]
    },
}

TIES_PRESETS = {
    "structure_heavy": {"commu_lora": 0.7, "emopia_lora": 0.15, "slakh_lora": 0.15},
    "emotion_heavy":   {"commu_lora": 0.2, "emopia_lora": 0.6, "slakh_lora": 0.2},
    "orchestra_heavy": {"commu_lora": 0.2, "emopia_lora": 0.15, "slakh_lora": 0.65},
    "balanced":        {"commu_lora": 0.4, "emopia_lora": 0.3, "slakh_lora": 0.3},
}

ADAPTER_PROFILES = {
    "commu":  {"pitch_strength": 0.95, "quant_strength": 0.90, "allow_accidentals": False},
    "emopia": {"pitch_strength": 0.40, "quant_strength": 0.50, "allow_accidentals": True},
    "slakh":  {"pitch_strength": 0.60, "quant_strength": 0.70, "allow_accidentals": True},
    "hybrid": {"pitch_strength": 0.70, "quant_strength": 0.65, "allow_accidentals": True},
}

# =====================================================================
# SYMPHONIC MODE ADDITIONS
# =====================================================================

# ---------------------------------------------------------------------
# TIME SIGNATURES
# ---------------------------------------------------------------------
# beats_per_bar drives every place that used to hardcode "* 4" for beats
# per bar throughout structure_planner / chord_realizer / hard_scorer /
# agentic_composer. beat_unit is tracked for future use but isn't consumed
# by the beat math yet — bpm is currently always interpreted as
# "quarter-note beats per minute" regardless of beat_unit, which is a known
# simplification: 6/8 currently runs as 6 quarter-note pulses per bar rather
# than a true 2-dotted-quarter compound-meter feel. Fine for structure and
# scoring; not yet idiomatically "compound."
TIME_SIGNATURES = {
    "4/4": {"beats_per_bar": 4, "beat_unit": 4},
    "3/4": {"beats_per_bar": 3, "beat_unit": 4},
    "2/4": {"beats_per_bar": 2, "beat_unit": 4},
    "6/8": {"beats_per_bar": 6, "beat_unit": 8},
    "9/8": {"beats_per_bar": 9, "beat_unit": 8},
    "12/8": {"beats_per_bar": 12, "beat_unit": 8},
}
DEFAULT_TIME_SIGNATURE = "4/4"

# ---------------------------------------------------------------------
# CLASSICAL FORM SKELETONS
# ---------------------------------------------------------------------
# Keyed by section name, same convention MACRO_SKELETONS already uses.
# StructurePlanner checks MACRO_SKELETONS first, then falls back to this
# dict, so if "Development"/"Theme"/etc. is ever added to MACRO_SKELETONS
# directly, that takes priority. These are functional (T/S/D) skeletons —
# same vocabulary HARMONIC_FUNCTIONS_MAJOR/MINOR already resolve — giving
# the harmonic *contour* a section needs (how much tonic vs. dominant
# tension) rather than a literal bar-by-bar transcription of sonata form.
CLASSICAL_MACRO_SKELETONS = {
    "Exposition": ["T", "T", "S", "D", "T", "S", "D", "D"],
    "Development": ["S", "D", "S", "T", "D", "S", "D"],
    "Recapitulation": ["T", "S", "D", "T", "S", "D", "T", "T"],
    "Theme": ["T", "S", "D", "T"],
    "Variation I": ["T", "S", "D", "T"],
    "Variation II": ["T", "D", "S", "D", "T"],
    "Variation III": ["T", "S", "T", "D", "T"],
    "Variation IV": ["S", "T", "D", "S", "T"],
    "Rondo A": ["T", "D", "T"],
    "Rondo B": ["S", "D", "S", "T"],
    "Rondo C": ["T", "S", "D", "D", "T"],
    "Coda": ["S", "D", "T", "T"],
    "Movement": ["T", "S", "D", "T", "S", "D", "T"],
}

# ---------------------------------------------------------------------
# ORCHESTRAL ARRANGEMENT PROFILE
# ---------------------------------------------------------------------
# Kept separate from ARRANGEMENT_PROFILES rather than merged in, so nothing
# about "build"/"sparse_to_dense" behavior changes for pop/song-form
# requests. StructurePlanner only reaches for this dict when
# llm_output["style"] is "orchestral"/"symphonic"/"classical". Instrument
# names are restricted to what's actually in STUDIO_INSTRUMENTS
# (llm_wrapper.py) — there's no separate Viola/Double Bass/Harp entry yet,
# so "Strings" and "Cello" do double duty for the string section. Real
# divisi (violins vs. violas vs. cellos as distinct target instruments)
# would mean expanding STUDIO_INSTRUMENTS too — that's still an open gap,
# not something fixed by this dict alone.
ORCHESTRAL_ARRANGEMENT_PROFILES = {
    "orchestral_tutti": {
        "Exposition": ["Violin", "French Horn", "Cello", "Flute"],
        "Development": ["Violin", "Cello", "Clarinet", "French Horn"],
        "Recapitulation": ["Violin", "French Horn", "Trumpet", "Cello", "Percussion"],
        "Theme": ["Violin", "Cello"],
        "Movement": ["Violin", "French Horn", "Cello", "Oboe"],
    }
}
ORCHESTRAL_DEFAULT_VOICES = ["Violin", "French Horn", "Cello"]