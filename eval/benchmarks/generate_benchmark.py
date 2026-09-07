import json
import os

styles = ["pop", "classical", "cinematic", "jazz"]
keys = ["C", "G", "D", "A", "E", "F", "Bb", "Eb", "Ab", "Am", "Em", "Dm"]
modes = {"Am": "minor", "Em": "minor", "Dm": "minor"}

prompts = []
prompt_id = 1

# Style-specific archetypes
archetypes = {
    "pop": [
        ("happy", "A vibrant energetic dance pop track with bright synths and driving bass", [115, 120, 128]),
        ("romantic", "A warm uplifting acoustic pop ballad with emotional chords", [85, 95, 105]),
        ("calm", "A gentle lo-fi chill pop groove with mellow electric piano", [75, 80, 88]),
        ("sad", "A heart-wrenching emotional breakup pop song with tender piano", [70, 78, 85])
    ],
    "classical": [
        ("sad", "A solemn melancholic classical piano nocturne with delicate phrasing", [65, 72, 80]),
        ("calm", "A serene contemplative pastoral string quartet in baroque counterpoint", [70, 78, 84]),
        ("dramatic", "A virtuosic dramatic sonata movement with bold harmonic tensions", [100, 112, 126]),
        ("happy", "A joyful classical rondo with playful staccato melody and bright harmony", [120, 130, 140])
    ],
    "cinematic": [
        ("tense", "A high-stakes suspenseful orchestral cue with brooding cellos and ostinato", [105, 115, 125]),
        ("epic", "A heroic sweeping orchestral fantasy theme with brass fanfares and soaring strings", [120, 132, 144]),
        ("dark", "A sinister gothic film score with ominous low brass and creeping dissonance", [60, 72, 80]),
        ("calm", "An ethereal celestial ambient soundtrack with shimmering pads and gentle harp", [68, 75, 82])
    ],
    "jazz": [
        ("happy", "A swinging upbeat bebop standard with lively syncopated walking bass", [130, 140, 155]),
        ("calm", "A smoky midnight modal jazz ballad with lush extended chords", [60, 68, 74]),
        ("romantic", "A sweet nostalgic bossa nova with acoustic guitar and silky piano voicings", [100, 110, 120]),
        ("tense", "An intricate hard bop progression with chromatic voice leading and high energy", [125, 135, 145])
    ]
}

section_templates = {
    "pop": [
        {"name": "Intro", "length": "short"},
        {"name": "Verse", "length": "medium"},
        {"name": "Chorus", "length": "medium"},
        {"name": "Outro", "length": "short"}
    ],
    "classical": [
        {"name": "Exposition", "length": "medium"},
        {"name": "Development", "length": "long"},
        {"name": "Recapitulation", "length": "medium"},
        {"name": "Coda", "length": "short"}
    ],
    "cinematic": [
        {"name": "Prologue", "length": "short"},
        {"name": "Buildup", "length": "medium"},
        {"name": "Climax", "length": "medium"},
        {"name": "Resolution", "length": "short"}
    ],
    "jazz": [
        {"name": "Head In", "length": "medium"},
        {"name": "Solo A", "length": "medium"},
        {"name": "Solo B", "length": "medium"},
        {"name": "Head Out", "length": "medium"}
    ]
}

for style in styles:
    style_count = 0
    while style_count < 25:
        arch_idx = style_count % len(archetypes[style])
        mood, desc, bpms = archetypes[style][arch_idx]
        key_raw = keys[(style_count * 3) % len(keys)]
        mode = modes.get(key_raw, "minor" if "m" in key_raw else "major")
        key = key_raw.replace("m", "")
        bpm = bpms[(style_count // len(archetypes[style])) % len(bpms)]
        
        full_text = f"{desc} in {key} {mode}, {bpm} BPM."
        
        sections = []
        for s in section_templates[style]:
            sec = s.copy()
            sec["mood"] = mood
            sections.append(sec)
            
        prompts.append({
            "id": prompt_id,
            "prompt": full_text,
            "global_key": key,
            "global_mode": mode,
            "global_bpm": bpm,
            "global_time_signature": "4/4",
            "style": style,
            "sections": sections
        })
        prompt_id += 1
        style_count += 1

out_path = "/home/aashishbishow/ProjectX/moonbeam-studio/eval/benchmarks/benchmark_100.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(prompts, f, indent=2)

print(f"Successfully generated {len(prompts)} balanced evaluation prompts in {out_path}")
