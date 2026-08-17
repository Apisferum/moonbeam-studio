import streamlit as st
import os
import random
import pandas as pd

# Design page styling
st.set_page_config(page_title="Moonbeam Subjective Listening Survey", layout="wide")

# Theme Palette (sleek dark/indigo theme)
st.markdown("""
<style>
    .main { background-color: #0f111a; color: #e2e8f0; }
    h1 { color: #818cf8; font-family: 'Inter', sans-serif; font-weight: 800; }
    .stButton>button { background-color: #4f46e5; color: white; border-radius: 8px; font-weight: 600; }
    .stAudio { background-color: #1e1b4b; border-radius: 8px; padding: 5px; }
</style>
""", unsafe_style_html=True)

st.title("🎼 Moonbeam Evaluation — Subjective Listening Survey")
st.markdown("Help us evaluate the musicality, coherence, and quality of our symbolic music model configurations. All audio clips are rendered using a normalized acoustic synthesizer.")

# Local file configuration
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
CSV_PATH = os.path.join(RESULTS_DIR, "survey_responses.csv")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Select length-stratified condition
length_condition = st.sidebar.selectbox(
    "Clip Length Stratification Filter",
    ["30 seconds", "2 minutes", "Full-length"]
)

# Simulated audio clips lists (will play placeholders if no MIDI rendered files exist)
audio_options = [
    {"name": "Prompt 1 (Full System)", "path": "midi/full_system_prompt_1.wav", "config": "full_system"},
    {"name": "Prompt 1 (Vanilla Moonbeam)", "path": "midi/vanilla_moonbeam_prompt_1.wav", "config": "vanilla_moonbeam"},
    {"name": "Prompt 1 (Hi-ACG)", "path": "midi/hi_acg_prompt_1.wav", "config": "hi_acg"},
]

st.header(f"Blind Pairwise Evaluation (Stratified: {length_condition})")
st.write("Listen to the two options below and rate them blindly. The models are randomized.")

# Randomized options state
if "option_a_config" not in st.session_state:
    choices = random.sample(audio_options, 2)
    st.session_state.option_a_name = choices[0]["name"]
    st.session_state.option_a_config = choices[0]["config"]
    st.session_state.option_b_name = choices[1]["name"]
    st.session_state.option_b_config = choices[1]["config"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔊 Audio Option A")
    # In a live setup, we'd render the WAV path. For survey demo, we display an audio player placeholder.
    st.audio(data=None, format="audio/wav")
    st.markdown("*Plays generated sequence blindly.*")

with col2:
    st.subheader("🔊 Audio Option B")
    st.audio(data=None, format="audio/wav")
    st.markdown("*Plays alternative generated sequence blindly.*")

# Rating forms
st.markdown("---")
st.subheader("📊 Rate the Musical Selections (MOS 1 to 5 scale)")

with st.form("rating_form"):
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Rate Option A")
        mus_a = st.slider("Musicality (Option A)", 1, 5, 3, help="Harmonic richness, lack of sour notes")
        nat_a = st.slider("Naturalness (Option A)", 1, 5, 3, help="Human-like phrasing and rhythmic variation")
        bnd_a = st.slider("Boundary Transition Clarity (Option A)", 1, 5, 3, help="Seamlessness of verse-to-chorus boundaries")
        blu_a = st.slider("Blueprint Adherence (Option A)", 1, 5, 3, help="Alignment with chord layout and orchestration rules")
        coh_a = st.slider("Long-term Coherence (Option A)", 1, 5, 3, help="Melodic motifs repeating coherently across sections")

    with c2:
        st.markdown("### Rate Option B")
        mus_b = st.slider("Musicality (Option B)", 1, 5, 3)
        nat_b = st.slider("Naturalness (Option B)", 1, 5, 3)
        bnd_b = st.slider("Boundary Transition Clarity (Option B)", 1, 5, 3)
        blu_b = st.slider("Blueprint Adherence (Option B)", 1, 5, 3)
        coh_b = st.slider("Long-term Coherence (Option B)", 1, 5, 3)

    submitted = st.form_submit_button("Submit Blind Response")
    
    if submitted:
        # Create response log
        response = {
            "timestamp": pd.Timestamp.now(),
            "length_condition": length_condition,
            "config_a": st.session_state.option_a_config,
            "config_b": st.session_state.option_b_config,
            "musicality_a": mus_a, "musicality_b": mus_b,
            "naturalness_a": nat_a, "naturalness_b": nat_b,
            "boundary_a": bnd_a, "boundary_b": bnd_b,
            "blueprint_a": blu_a, "blueprint_b": blu_b,
            "coherence_a": coh_a, "coherence_b": coh_b
        }
        
        # Save to CSV
        df = pd.DataFrame([response])
        if os.path.exists(CSV_PATH):
            df.to_csv(CSV_PATH, mode='a', header=False, index=False)
        else:
            df.to_csv(CSV_PATH, index=False)
            
        st.success("🎉 Response logged successfully! Resetting options for next blind pair...")
        
        # Reset randomized choices for next round
        choices = random.sample(audio_options, 2)
        st.session_state.option_a_name = choices[0]["name"]
        st.session_state.option_a_config = choices[0]["config"]
        st.session_state.option_b_name = choices[1]["name"]
        st.session_state.option_b_config = choices[1]["config"]
        st.experimental_rerun()

st.markdown("---")
# Show current responses summary
if os.path.exists(CSV_PATH):
    st.subheader("📊 Collected Survey Stats")
    df_collected = pd.read_csv(CSV_PATH)
    st.write(f"Total responses collected so far: **{len(df_collected)}**")
    st.dataframe(df_collected.tail(5))
