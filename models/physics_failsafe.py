import os
import random
import torch
import torch.nn as nn
try:
    import pretty_midi
except ImportError:
    pretty_midi = None
from shared.music_theory_constants import logger, TICKS_PER_SECOND

# ----------------------------
# Neural Network
# ----------------------------
class WavePINNModel(nn.Module):
    def __init__(self, input_dim=8, wave_dim=32, latent_dim=16):
        super(WavePINNModel, self).__init__()
        self.wave_layer = nn.Linear(input_dim, wave_dim)
        self.latent_layer = nn.Linear(wave_dim, latent_dim)
        self.pinn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        wave = torch.sin(self.wave_layer(x))
        latent = self.latent_layer(wave)
        next_latent = self.pinn(latent)
        output = self.decoder(next_latent)
        return output

# ----------------------------
# Helper Helpers
# ----------------------------
def get_seed_pitch(key: str, mode: str, primer_midi: "pretty_midi.PrettyMIDI" = None) -> int:
    if primer_midi and pretty_midi is not None and primer_midi.instruments:
        notes = []
        for inst in primer_midi.instruments:
            if not getattr(inst, "is_drum", False):
                notes.extend(inst.notes)
        if notes:
            last_note = max(notes, key=lambda n: n.end)
            return last_note.pitch
            
    pc_map = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, 
        "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
    }
    pc = pc_map.get(key, 0)
    return 60 + pc  # Default to middle octave

def make_seed_tensor(pitch: int, amp: float = 0.65, dur: float = 0.5, emotion=None, instrument: int = 25) -> torch.Tensor:
    if emotion is None:
        emotion = [0.8, 0.1, 0.2, 0.1]
    hz = float(440.0 * (2.0 ** ((pitch - 69.0) / 12.0)))
    data = [hz, float(amp), float(dur)] + list(map(float, emotion)) + [instrument / 127.0]
    return torch.tensor([data], dtype=torch.float32)

def parse_chord_pcs(chord_str: str) -> list[int]:
    root_map = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, 
        "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
    }
    if not chord_str:
        return [0, 4, 7]
    parts = chord_str.split(":")
    root_name = parts[0]
    quality = parts[1] if len(parts) > 1 else "maj"
    
    root_pc = 0
    for name, pc in root_map.items():
        if root_name.startswith(name):
            if len(name) > len(root_name) or (len(name) == 1 and len(root_name) > 1 and root_name[1] in ["#", "b"]):
                continue
            root_pc = pc
            break
            
    quality = quality.lower()
    intervals = [0, 4, 7]
    if "min" in quality or "m" in quality:
        if "dim" in quality:
            intervals = [0, 3, 6]
        else:
            intervals = [0, 3, 7]
    elif "dim" in quality:
        intervals = [0, 3, 6]
    elif "aug" in quality or "+" in quality:
        intervals = [0, 4, 8]
    elif "sus2" in quality:
        intervals = [0, 2, 7]
    elif "sus4" in quality or "sus" in quality:
        intervals = [0, 5, 7]
        
    return [(root_pc + i) % 12 for i in intervals]

def get_active_chord_pcs(t_beats: float, chord_timeline: list) -> list[int]:
    if not chord_timeline:
        return [0, 4, 7]
    for chord_event in chord_timeline:
        start = chord_event.get("start_beat", 0.0)
        end = chord_event.get("end_beat", 999.0)
        if start <= t_beats < end:
            return parse_chord_pcs(chord_event.get("chord", "C:maj"))
    return parse_chord_pcs(chord_timeline[-1].get("chord", "C:maj"))

# ----------------------------
# PINN Controller
# ----------------------------
class PhysicsFailsafe:
    def __init__(self, config: dict, device: str = "cpu"):
        self.device = device
        self.config = config.get("physics_failsafe", {}) if config else {}
        self.enabled = self.config.get("enabled", True)
        
        latent_dim = self.config.get("latent_dim", 16)
        wave_dim = self.config.get("wave_dim", 32)
        
        self.model = WavePINNModel(input_dim=8, wave_dim=wave_dim, latent_dim=latent_dim)
        
        model_path = self.config.get("model_path")
        if self.enabled:
            if not model_path:
                logger.warning("⚠️ Physics Failsafe is enabled but model_path is not specified in default.yaml. Disabling failsafe.")
                self.enabled = False
            elif not os.path.exists(model_path):
                logger.warning(f"⚠️ Physics Failsafe is enabled but checkpoint '{model_path}' was not found. Disabling failsafe.")
                self.enabled = False
            else:
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=device))
                    logger.info(f"Loaded Physics Failsafe model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load Physics Failsafe weights: {e}. Disabling failsafe.")
                    self.enabled = False

        self.model.to(device)

    def generate_section(self, section: dict, primer_midi: "pretty_midi.PrettyMIDI" = None) -> "pretty_midi.PrettyMIDI":
        if pretty_midi is None:
            raise ImportError("pretty_midi is not installed. generate_section requires pretty_midi.")
        pm = pretty_midi.PrettyMIDI()
        
        # Resolve instrument programs from note_events if available
        note_events = section.get("note_events", [])
        guitar_program = self.config.get("guitar_program", 25)
        melody_program = guitar_program
        pad_program = guitar_program
        bass_program = guitar_program
        arp_program = guitar_program
        
        if note_events:
            def get_pitch(e):
                if "pitch" in e:
                    return e["pitch"]
                return e.get("octave", 5) * 12 + e.get("pitch_class", 0)
                
            bass_events = [e for e in note_events if get_pitch(e) < 45]
            melody_events = [e for e in note_events if get_pitch(e) >= 60]
            pad_events = [e for e in note_events if 45 <= get_pitch(e) < 60]
            
            def get_most_common_program(events, fallback):
                progs = [e.get("instrument_program") for e in events if e.get("instrument_program") is not None]
                if progs:
                    return max(set(progs), key=progs.count)
                return fallback
                
            bass_program = get_most_common_program(bass_events, guitar_program)
            melody_program = get_most_common_program(melody_events, guitar_program)
            pad_program = get_most_common_program(pad_events, guitar_program)
            arp_program = melody_program
            
        instruments_dict = {}
        def get_instrument_track(program: int, is_drum: bool = False) -> "pretty_midi.Instrument":
            key = (program, is_drum)
            if key not in instruments_dict:
                inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
                instruments_dict[key] = inst
                pm.instruments.append(inst)
            return instruments_dict[key]
        
        bpm = section.get("bpm", 120)
        bars = section.get("bars", 8)
        beats_per_bar = section.get("beats_per_bar", 4) or 4
        chord_timeline = section.get("chord_timeline", [])
        
        key = section.get("key", "C")
        mode = section.get("mode", "major")
        
        root_map = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, 
            "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
        }
        key_pc = root_map.get(key, 0)
        
        def build_scale_pitch_classes(key_pc: int, mode: str) -> set[int]:
            major = [0, 2, 4, 5, 7, 9, 11]
            minor = [0, 2, 3, 5, 7, 8, 10]
            steps = major if mode == "major" else minor
            return {(key_pc + s) % 12 for s in steps}
            
        scale_pcs = build_scale_pitch_classes(key_pc, mode)
        
        beat_seconds = 60.0 / float(bpm)
        bar_seconds = beats_per_bar * beat_seconds
        seconds = bars * bar_seconds
        
        prev_chord = None
        prev_arp = None
        chord_root_base = 47
        arp_root_base = 67
        bass_near = 33
        
        arp_pattern_8 = [0, 1, 0, 1, 0, 1, 0, 1]
        arp_pattern_16 = [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1]
        
        style = section.get("style", "cinematic")
        
        # Helper snapping functions
        def clamp_int(v, lo, hi):
            return max(lo, min(hi, int(v)))
            
        def nearest_in_scale(midi_pitch: int, scale_pitch_classes: set[int]) -> int:
            if midi_pitch % 12 in scale_pitch_classes:
                return midi_pitch
            for delta in range(1, 12):
                up = midi_pitch + delta
                down = midi_pitch - delta
                if up % 12 in scale_pitch_classes: return up
                if down % 12 in scale_pitch_classes: return down
            return midi_pitch
            
        def pitch_for_pc_near(pc: int, near: int, lo: int, hi: int) -> int:
            best = None
            best_dist = None
            for delta in range(-36, 37):
                p = near + delta
                if p < lo or p > hi: continue
                if p % 12 != pc: continue
                dist = abs(delta)
                if best is None or dist < best_dist:
                    best = p
                    best_dist = dist
            return clamp_int(best if best is not None else near, lo, hi)

        def realize_chord_near(root_base: int, chord_pcs: list[int], prev: list[int] | None) -> list[int]:
            candidates = []
            for inversion in range(3):
                chord = []
                for k in range(3):
                    pc = chord_pcs[(k + inversion) % len(chord_pcs)]
                    target = root_base + 4 * k
                    p = target
                    p = nearest_in_scale(p, set(chord_pcs))
                    while p % 12 != pc:
                        p += 1
                        if p > 127: break
                    chord.append(clamp_int(p, 0, 127))
                chord_sorted = sorted(chord)
                if prev is None:
                    score = 0
                else:
                    score = sum(abs(a - b) for a, b in zip(chord_sorted, sorted(prev)))
                candidates.append((score, chord_sorted))
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        # Accompaniment generation
        for bar in range(bars):
            start_t = bar * bar_seconds
            end_t = min(seconds, (bar + 1) * bar_seconds)
            
            start_beat = bar * beats_per_bar
            chord_pcs = get_active_chord_pcs(start_beat, chord_timeline)
            
            swell = min(1.0, max(0.0, start_t / max(1e-6, seconds)))
            swell = swell * swell
            
            if style == "ambient":
                pad_vel = 42 + int(round(10 * swell))
                arp_vel = 0
                bass_vel = 48 + int(round(10 * swell))
            else:
                pad_vel = 40 + int(round(18 * swell))
                arp_vel = 44 + int(round(18 * swell))
                bass_vel = 54 + int(round(22 * swell))
                
            pad_vel = clamp_int(pad_vel, 30, 85)
            arp_vel = clamp_int(arp_vel, 35, 95)
            bass_vel = clamp_int(bass_vel, 40, 110)
            
            chord_notes = realize_chord_near(chord_root_base, chord_pcs[:3], prev_chord)
            prev_chord = chord_notes
            
            pad_notes = chord_notes[:3]
            for p in pad_notes:
                get_instrument_track(pad_program).notes.append(
                    pretty_midi.Note(
                        velocity=pad_vel,
                        pitch=clamp_int(p, 21, 108),
                        start=float(start_t),
                        end=float(end_t if style != "ambient" else start_t + bar_seconds * 0.9)
                    )
                )
                
            root_pc = chord_pcs[0]
            for beat in (0, 2):
                bt = start_t + beat * beat_seconds
                if bt >= seconds: continue
                bass_pitch = pitch_for_pc_near(root_pc, bass_near, 21, 60)
                get_instrument_track(bass_program).notes.append(
                    pretty_midi.Note(
                        velocity=bass_vel,
                        pitch=bass_pitch,
                        start=float(bt),
                        end=float(min(seconds, bt + beat_seconds * 0.95))
                    )
                )
                
            if style != "ambient":
                arp_chord = realize_chord_near(arp_root_base, chord_pcs[:3], prev_arp)
                prev_arp = arp_chord
                
                section_name = section.get("section_name", "verse").lower()
                if any(x in section_name for x in ["chorus", "climax", "bridge"]):
                    arp_step = beat_seconds / 4.0
                    arp_pattern = arp_pattern_16
                else:
                    arp_step = beat_seconds / 2.0
                    arp_pattern = arp_pattern_8
                    
                for s, idx in enumerate(arp_pattern):
                    t0 = start_t + s * arp_step
                    if t0 >= end_t or t0 >= seconds: break
                    p = clamp_int(arp_chord[idx % len(arp_chord)], 48, 96)
                    get_instrument_track(arp_program).notes.append(
                        pretty_midi.Note(
                            velocity=arp_vel,
                            pitch=p,
                            start=float(t0),
                            end=float(min(seconds, t0 + arp_step * 0.92))
                        )
                    )

        # PINN Melody generation
        step_seconds = self.config.get("step_seconds", 0.5)
        steps = int(max(1.0, float(seconds)) / max(1e-6, step_seconds))
        
        seed_pitch = get_seed_pitch(key, mode, primer_midi)
        
        # Try to parse emotional quadrants if present in timeline
        seed_tensor = make_seed_tensor(seed_pitch, amp=0.65, dur=step_seconds, instrument=melody_program)
        current_input = seed_tensor.to(self.device)
        
        melody_min, melody_max = (44, 82) if style == "cinematic" else (40, 88)
        prev_melody = clamp_int(seed_pitch, melody_min, melody_max)
        prev_melody = nearest_in_scale(prev_melody, scale_pcs)
        
        self.model.eval()
        current_time = 0.0
        
        def nearest_pitch_in_pcs(prev: int, pcs: list[int], lo: int, hi: int) -> int:
            best = None
            best_dist = None
            for pc in pcs:
                p = pitch_for_pc_near(pc, prev, lo, hi)
                dist = abs(p - prev)
                if best is None or dist < best_dist:
                    best = p
                    best_dist = dist
            return clamp_int(best if best is not None else prev, lo, hi)

        with torch.no_grad():
            for _ in range(steps):
                out = self.model(current_input)
                
                raw_pitch = out[0, 0]
                raw_amp = out[0, 1]
                raw_dur = out[0, 2]
                
                amp = float(torch.sigmoid(raw_amp).item())
                dur = float(step_seconds * (0.6 + 0.5 * torch.sigmoid(raw_dur).item()))
                
                target = melody_min + int(round(torch.sigmoid(raw_pitch).item() * (melody_max - melody_min)))
                
                current_beat = current_time / beat_seconds
                chord_pcs = get_active_chord_pcs(current_beat, chord_timeline)
                
                beat_pos = current_time / max(1e-6, beat_seconds)
                on_beat = abs(beat_pos - round(beat_pos)) < 1e-6
                
                if (not on_beat) and random.random() < (0.16 if style == "ambient" else 0.08):
                    pitch = None
                else:
                    if on_beat:
                        chord_pitch = nearest_pitch_in_pcs(prev_melody, chord_pcs, melody_min, melody_max)
                        if random.random() < 0.55:
                            chord_pitch = clamp_int(chord_pitch + random.choice([-2, 0, 2]), melody_min, melody_max)
                        candidate = int(round(0.6 * chord_pitch + 0.4 * target))
                        pitch = clamp_int(candidate, melody_min, melody_max)
                        pitch = nearest_pitch_in_pcs(pitch, chord_pcs if random.random() < 0.65 else list(scale_pcs), melody_min, melody_max)
                    else:
                        step = int(round(torch.clamp(torch.randn(1) * 2.0, -4, 4).item()))
                        candidate = int(round(0.75 * (prev_melody + step) + 0.25 * target))
                        pitch = clamp_int(candidate, melody_min, melody_max)
                        pitch = nearest_in_scale(pitch, scale_pcs)
                        
                    if pitch is not None and random.random() < 0.06:
                        pitch = clamp_int(pitch + random.choice([-1, 1]), melody_min, melody_max)
                        
                    if style == "cinematic" and pitch is not None:
                        p_int = int(pitch)
                        drop_prob = 0.90 if p_int >= 82 else (0.60 if p_int >= 78 else 0.0)
                        if drop_prob and random.random() < drop_prob:
                            pitch = clamp_int(p_int - 12, melody_min, melody_max)
                            pitch = nearest_in_scale(pitch, scale_pcs)
                            p_int = int(pitch)
                        if prev_melody is not None and p_int >= 78 and int(prev_melody) == p_int:
                            pitch = clamp_int(p_int - random.choice([2, 3, 5]), melody_min, melody_max)
                            pitch = nearest_in_scale(pitch, scale_pcs)
                            
                swell = min(1.0, max(0.0, current_time / max(1e-6, seconds)))
                swell = swell * swell
                
                section_name = section.get("section_name", "verse").lower()
                base_vel = (48 if "intro" in section_name or "verse" in section_name else 64) + int(round(18 * swell))
                velocity = clamp_int(round(base_vel + amp * 50.0), 35, 120)
                
                if style == "cinematic" and pitch is not None and int(pitch) >= 78:
                    velocity = clamp_int(int(velocity * 0.85), 30, 110)
                    
                start_t = float(current_time)
                end_t = float(min(seconds, current_time + dur))
                
                if pitch is not None:
                    get_instrument_track(melody_program).notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=int(pitch),
                            start=start_t,
                            end=end_t
                        )
                    )
                    prev_melody = int(pitch)
                    
                current_time += float(step_seconds)
                if current_time >= seconds: break
                
                next_input = out.clone()
                next_input[0, 0] = float(440.0 * (2.0 ** ((prev_melody - 69.0) / 12.0)))
                next_input[0, 1] = amp
                next_input[0, 2] = float(dur)
                next_input[0, 3:-1] = current_input[0, 3:-1]
                next_input[0, -1] = melody_program / 127.0
                current_input = next_input
                
        return pm
