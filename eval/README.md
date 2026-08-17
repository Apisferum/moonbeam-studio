# Moonbeam Evaluation Framework

This directory contains the driver and configs for evaluating the Moonbeam model and its various ablation configurations.

## Setup & Environment

Ensure you activate your virtual environment:
```bash
# E.g. activate your moon environment
source activate moon
```

If dependencies (such as custom packages) are not fully linked, run the following from the root workspace folder:

1. **Install custom `transformers_minimal`**:
   ```bash
   pip install -e ../moonbeam-codebase/src/llama_recipes/transformers_minimal
   ```
2. **Install `moonbeam-codebase`** (which contains the custom `recipes` package):
   ```bash
   pip install -e ../moonbeam-codebase
   ```
3. **Install additional evaluation dependencies**:
   ```bash
   pip install fairscale faiss-cpu pyyaml pydantic numpy pretty_midi python-dotenv
   ```

## Model Checkpoint Paths

The evaluation driver uses environment variables to resolve weights and configs. If not set, it will attempt autodiscovery within common workspace directories. To explicitly declare them:

```bash
export BASE_MODEL_PATH="/path/to/moonbeam_checkpoint/moonbeam_839M.pt"
export LORA_DIR="/path/to/moonbeam_checkpoint/multi_task_lora"
export CONFIG_PATH="/path/to/src/llama_recipes/configs/model_config_multi_task.json"
export MASTER_DICT_PATH="/path/to/processed/ComMU/indexed_tokens_dict.json"
export GEMINI_API_KEY="your-api-key"
```

## Running Evaluations

### 1. Dry-Run / Sanity Check (1 Prompt)
To test imports, checkpoint loading, intent generation, planning, composing, and metrics calculation on a single prompt:
```bash
python eval/run_eval.py --prompts_limit 1
```

### 2. Full Evaluation (All 3 Prompts)
To run the full evaluation across all configurations:
```bash
python eval/run_eval.py
```

### 3. Mock Simulation Mode (Dry-Run / Metric Testing)
To run a fast simulation using dummy MIDI files (no GPU required):
```bash
python eval/run_eval.py --mock
```

## Configuration Matrix

The framework evaluates the following six ablation configurations:
* **`full_system`**: Standard pipeline with structural planning, FAISS motif memory, TIES weight adaptation, soft refiner, and hard scorer.
* **`no_planner`**: Bypasses `StructurePlanner` (uses fallback timeline with empty `note_events` triggering pure model free-sampling).
* **`no_faiss`**: Bypasses continuation context and seed retrieval/saving from FAISS motif memory.
* **`single_adapter`**: Replaces the TIES mixture of LoRA adapter weights with a single adapter (`commu_lora = 1.0`) and disables feedback adaptation.
* **`no_soft`**: Disables token-level velocity and constraint boundary filtering.
* **`no_hard`**: Bypasses the `HardScorer` retry loop, accepting the first generation attempt immediately.

## Output Structure

The results are generated in the `eval/results/` directory:
* **`summary_table.md`**: Markdown report comparing metrics across configurations.
* **`aggregate_metrics.json`**: Detailed JSON record of all runs.
* **`midi/`**: Directory containing the generated MIDI files for each configuration.
* **`*_trace.json`**: Individual instrumentation traces saved for each piece.
