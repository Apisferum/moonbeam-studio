import logging

# Set up logger for the whole studio
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- DRUM MAPPINGS ---
DRUM_INSTRUMENT_IDS = {128, 129, 130}

# --- CHROMATIC TO PITCH CLASS MAP ---
NOTE_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

# --- REVERSE PITCH CLASS MAP (Needed by Planner to output "C", "F#", etc.) ---
PC_TO_NAME = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F', 
    6: 'F#', 7: 'G', 8: 'Ab', 9: 'A', 10: 'Bb', 11: 'B'
}

# --- SCALE / MODE DEFINITIONS ---
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

# --- CHORD-AWARE INTERVAL DICTIONARY ---
CHORD_INTERVALS = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "dim": [0, 3, 6],
    "aug": [0, 4, 8], "sus2": [0, 2, 7], "sus4": [0, 5, 7],
    "min9": [0, 3, 7, 10, 2], "maj9": [0, 4, 7, 11, 2], "9": [0, 4, 7, 10, 2]
}

# --- INSTRUMENT FAMILY MAPPINGS (For the Hard Scorer) ---
INSTRUMENT_FAMILIES = {
    "Piano": list(range(0, 8)), "Chromatic_Percussion": list(range(8, 16)),
    "Organ": list(range(16, 24)), "Guitar": list(range(24, 32)),
    "Bass": list(range(32, 40)), "Strings": list(range(40, 48)),
    "Ensemble": list(range(48, 56)), "Brass": list(range(56, 64)),
    "Reed": list(range(64, 72)), "Pipe": list(range(72, 80)),
    "Synth_Lead": list(range(80, 88)), "Synth_Pad": list(range(88, 96)),
    "Percussion": list(range(128, 130)) # Standard MIDI Drums
}

# =====================================================================
# PLANNER-SPECIFIC CONSTANTS (The Algorithmic Composer Engine)
# =====================================================================

# --- HARMONIC FUNCTIONS (The Macro Skeleton) ---
HARMONIC_FUNCTIONS_MAJOR = {
    "T": ["I", "vi", "iii"],      # Tonic (Rest)
    "S": ["IV", "ii"],            # Subdominant (Movement)
    "D": ["V", "vii°"]            # Dominant (Tension)
}
HARMONIC_FUNCTIONS_MINOR = {
    "T": ["i", "III", "VI"],
    "S": ["iv", "ii°"],
    "D": ["v", "V", "vii°"]       # Includes harmonic minor V
}

# --- SECTION MACRO SKELETONS ---
MACRO_SKELETONS = {
    "Intro":      ["T", "T", "T", "T"],
    "Verse":      ["T", "T", "S", "S", "T", "T", "D", "D"],
    "Pre-Chorus": ["S", "S", "D", "D"],
    "Chorus":     ["S", "S", "T", "T", "S", "S", "D", "T"],
    "Bridge":     ["S", "D", "S", "D", "T", "T", "D", "D"],
    "Outro":      ["T", "S", "D", "T"]
}

# --- DEGREE TO QUALITY MAPPINGS ---
MAJOR_QUALITIES = {"I": "maj7", "ii": "min7", "iii": "min7", "IV": "maj7", "V": "7", "vi": "min7", "vii°": "dim"}
MINOR_QUALITIES = {"i": "min7", "ii°": "dim", "III": "maj7", "iv": "min7", "v": "min7", "V": "7", "VI": "maj7", "vii°": "dim"}

# --- ARRANGEMENT PROFILES (Instrument Entry/Exit) ---
ARRANGEMENT_PROFILES = {
    "build": {
        "Intro": ["Piano"], "Verse": ["Piano", "Strings"], "Pre-Chorus": ["Piano", "Strings", "Bass"],
        "Chorus": ["Piano", "Strings", "Brass", "Percussion"], "Bridge": ["Strings", "Reed"], "Outro": ["Piano", "Strings"]
    },
    "sparse_to_dense": {
        "Intro": ["Piano"], "Verse": ["Piano", "Bass"], "Pre-Chorus": ["Piano", "Bass", "Strings"],
        "Chorus": ["Piano", "Bass", "Strings", "Percussion"], "Bridge": ["Piano", "Strings"], "Outro": ["Piano"]
    },
}

# --- TIES WEIGHT PRESETS ---
TIES_PRESETS = {
    "structure_heavy": {"commu_lora": 0.7, "emopia_lora": 0.15, "slakh_lora": 0.15},
    "emotion_heavy":   {"commu_lora": 0.2, "emopia_lora": 0.6, "slakh_lora": 0.2},
    "orchestra_heavy": {"commu_lora": 0.2, "emopia_lora": 0.15, "slakh_lora": 0.65},
    "balanced":        {"commu_lora": 0.4, "emopia_lora": 0.3, "slakh_lora": 0.3},
}

# --- ADAPTER PROFILES (Used by the Critic Engine) ---
ADAPTER_PROFILES = {
    "commu":  {"pitch_strength": 0.95, "quant_strength": 0.90, "allow_accidentals": False},
    "emopia": {"pitch_strength": 0.40, "quant_strength": 0.50, "allow_accidentals": True},
    "slakh":  {"pitch_strength": 0.60, "quant_strength": 0.70, "allow_accidentals": True},
    "hybrid": {"pitch_strength": 0.70, "quant_strength": 0.65, "allow_accidentals": True},
}