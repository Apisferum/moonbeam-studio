import difflib
import logging

logger = logging.getLogger(__name__)

GM_PROGRAM_MAP = {
    "acoustic grand piano": 0, "piano": 0, "grand piano": 0,
    "bright acoustic piano": 1, "electric grand piano": 2, "honky-tonk piano": 3,
    "electric piano": 4, "rhodes": 4, "epiano": 4, "electric piano 2": 5,
    "harpsichord": 6, "clavinet": 7, "celesta": 8, "glockenspiel": 9,
    "music box": 10, "vibraphone": 11, "vibes": 11, "marimba": 12,
    "xylophone": 13, "tubular bells": 14, "bells": 14, "dulcimer": 15,
    "organ": 19, "church organ": 19, "hammond organ": 16, "accordion": 21,
    "harmonica": 22, "acoustic guitar": 24, "guitar": 24, "nylon guitar": 24,
    "steel guitar": 25, "acoustic guitar steel": 25, "jazz guitar": 26,
    "clean guitar": 27, "electric guitar": 27, "muted guitar": 28,
    "overdriven guitar": 29, "distortion guitar": 30, "distorted guitar": 30,
    "acoustic bass": 32, "upright bass": 32, "bass": 33, "electric bass": 33, 
    "finger bass": 33, "pick bass": 34, "fretless bass": 35, "slap bass": 36,
    "synth bass": 38, "violin": 40, "viola": 41, "cello": 42,
    "contrabass": 43, "double bass": 43, "tremolo strings": 44,
    "pizzicato strings": 45, "harp": 46, "timpani": 47,
    "strings": 48, "string ensemble": 48, "orchestral strings": 48,
    "synth strings": 50, "choir": 52, "choir aahs": 52, "voice oohs": 53,
    "orchestra hit": 55, "trumpet": 56, "trombone": 57, "tuba": 58,
    "muted trumpet": 59, "french horn": 60, "horn": 60, "brass": 61, 
    "brass section": 61, "synth brass": 62, "sax": 64, "soprano sax": 64,
    "alto sax": 65, "tenor sax": 66, "baritone sax": 67, "oboe": 68,
    "english horn": 69, "bassoon": 70, "clarinet": 71, "flute": 73,
    "piccolo": 72, "recorder": 74, "pan flute": 75, "synth lead": 80, 
    "lead synth": 80, "synth pad": 88, "pad": 88, "warm pad": 89,
    "atmosphere": 99, "ambient pad": 99, "sitar": 104, "banjo": 105,
    "shamisen": 106, "koto": 107, "kalimba": 108, "bagpipe": 109, "fiddle": 110,
    "synth": 80, "keyboard": 0,
    "percussion": 128, "drums": 128, "drum": 128,
}

DRUM_PROGRAM = 128

# Functional Bass Programs (Protects Tuba & Contrabass from Mud Transposition)
BASS_PROGRAMS = {32, 33, 34, 35, 36, 38, 43, 58}
LEAD_ISH_PROGRAMS = {0, 4, 5, 6, 24, 25, 26, 40, 56, 64, 65, 66, 71, 73, 80}

# Single source of truth for "this instrument legitimately plays low and
# shouldn't be flagged as register mud or transposed up." Everything in
# BASS_PROGRAMS, plus Cello (42) and Bassoon (70) — both routinely sit well
# below middle C in ordinary use despite not being bass-family programs.
# MIDIPolisher already protects Tuba/Contrabass via BASS_PROGRAMS directly;
# HardScorer's mud-penalty check imports THIS set instead of redefining its
# own, so the two modules can't quietly drift out of sync with each other.
LOW_REGISTER_SAFE_PROGRAMS = BASS_PROGRAMS | {42, 70}

DEFAULT_OCTAVE_BY_ROLE = {
    "bass": 2,
    "harmony": 4,
    "lead": 5,
}

def get_program(instrument_name: str) -> int:
    if not instrument_name: return 0
    key = instrument_name.strip().lower()
    if key in GM_PROGRAM_MAP: return GM_PROGRAM_MAP[key]
    for name, prog in GM_PROGRAM_MAP.items():
        if name in key or key in name: return prog
    close = difflib.get_close_matches(key, GM_PROGRAM_MAP.keys(), n=1, cutoff=0.6)
    if close: return GM_PROGRAM_MAP[close[0]]
    logger.warning(f"⚠️ Unknown instrument '{instrument_name}', defaulting to Piano (0)")
    return 0

def infer_role(program: int) -> str:
    if program in BASS_PROGRAMS: return "bass"
    if program in LEAD_ISH_PROGRAMS: return "lead"
    return "harmony"

def default_octave_for_role(role: str) -> int:
    return DEFAULT_OCTAVE_BY_ROLE.get(role, 4)