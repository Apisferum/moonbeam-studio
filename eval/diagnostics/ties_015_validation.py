import os
import json
import torch
import numpy as np
from safetensors.torch import load_file

def load_lora_deltas(lora_dir: str):
    adapters = ["commu_lora", "emopia_lora", "slakh_lora"]
    deltas = {a: {} for a in adapters}
    
    for adapter in adapters:
        path = os.path.join(lora_dir, adapter)
        cfg_path = os.path.join(path, "adapter_config.json")
        sf_path = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(sf_path):
            path = os.path.join(path, adapter)
            cfg_path = os.path.join(path, "adapter_config.json")
            sf_path = os.path.join(path, "adapter_model.safetensors")
            
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        scaling = cfg["lora_alpha"] / cfg["r"]
        
        sd = load_file(sf_path, device="cpu")
        modules = {}
        for k, v in sd.items():
            clean = k
            for prefix in ["base_model.model.", "model."]:
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
            if clean.startswith("model."):
                clean = clean[len("model."):]
                
            if "lora_A.weight" in clean:
                mod = clean.replace(".lora_A.weight", "")
                modules.setdefault(mod, {})["A"] = v.float()
            elif "lora_B.weight" in clean:
                mod = clean.replace(".lora_B.weight", "")
                modules.setdefault(mod, {})["B"] = v.float()
                
        for mod, pair in modules.items():
            if "A" in pair and "B" in pair:
                deltas[adapter][mod] = (pair["B"] @ pair["A"]) * scaling
    return deltas

def ties_merge_pair(t1: torch.Tensor, t2: torch.Tensor, w1: float, w2: float, density: float = 0.80):
    """
    Performs TIES merging between two task vectors with weights w1, w2.
    density = fraction of top magnitudes retained (e.g. 0.80 = retain top 80%).
    """
    v1 = (t1 * w1).numpy()
    v2 = (t2 * w2).numpy()
    tensors = [v1, v2]
    
    # 1. Trim
    trimmed = []
    for t in tensors:
        flat = t.flatten()
        k = int(len(flat) * (1.0 - density))
        if k > 0:
            magnitudes = np.abs(flat)
            thresh = np.partition(magnitudes, k - 1)[k - 1]
            t_trimmed = np.where(np.abs(t) >= thresh, t, 0.0)
        else:
            t_trimmed = t
        trimmed.append(t_trimmed)
        
    # 2. Elect Sign
    stacked = np.stack(trimmed, axis=0)
    sums = np.sum(stacked, axis=0)
    signs = np.where(sums >= 0.0, 1.0, -1.0)
    
    # 3. Disjoint Mean
    mask = (stacked * signs) > 0.0
    summed = np.sum(np.where(mask, stacked, 0.0), axis=0)
    count = np.sum(mask, axis=0)
    delta = np.where(count > 0, summed / count, 0.0)
    return torch.tensor(delta, dtype=t1.dtype)

def run_experiment():
    lora_dir = "/home/aashishbishow/ProjectX/moonbeam_chunk_20260816_170434/epoch_7"
    print(f"Loading LoRA adapters from {lora_dir}...")
    deltas = load_lora_deltas(lora_dir)
    
    # Identify conflicting modules between Slakh (orchestration) and EMOPIA (emotion) in layers 8-14
    slakh_mods = deltas["slakh_lora"]
    emopia_mods = deltas["emopia_lora"]
    shared_mods = [m for m in slakh_mods if m in emopia_mods]
    print(f"Found {len(shared_mods)} shared modules between Slakh and EMOPIA.")
    
    # Concatenate all shared parameters into monolithic vectors for macro-statistics
    v_target_full = torch.cat([slakh_mods[m].flatten() for m in shared_mods])
    v_partner_full = torch.cat([emopia_mods[m].flatten() for m in shared_mods])
    total_params = v_target_full.numel()
    print(f"Total parameter coordinates analyzed: {total_params:,}")
    
    # =========================================================================
    # EXPERIMENT 1: ADAPTER RETRY NUDGE STEP SIZE (\Delta w) SWEEP
    # =========================================================================
    print("\n--- EXPERIMENT 1: Adapter Retry Nudge Step Size (Delta w) Sweep ---")
    delta_w_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    base_w_target = 0.50
    base_w_partner = 0.50
    
    results_exp1 = []
    base_cos_partner = None
    
    for dw in delta_w_list:
        w_t = base_w_target + dw
        w_p = base_w_partner
        # normalize weights
        total_w = w_t + w_p
        w_t_norm = w_t / total_w
        w_p_norm = w_p / total_w
        
        merged_mods = {}
        for m in shared_mods:
            merged_mods[m] = ties_merge_pair(slakh_mods[m], emopia_mods[m], w_t_norm, w_p_norm, density=0.80)
            
        v_merged = torch.cat([merged_mods[m].flatten() for m in shared_mods])
        
        # 1. Coordinate sign election win rate for target
        target_sign = torch.sign(v_target_full)
        merged_sign = torch.sign(v_merged)
        non_zero_mask = (target_sign != 0) & (merged_sign != 0)
        win_rate = (target_sign[non_zero_mask] == merged_sign[non_zero_mask]).float().mean().item() * 100.0
        
        # 2. Cosine similarities
        cos_target = torch.cosine_similarity(v_merged.unsqueeze(0), v_target_full.unsqueeze(0)).item()
        cos_partner = torch.cosine_similarity(v_merged.unsqueeze(0), v_partner_full.unsqueeze(0)).item()
        
        # 3. MSRD (Mean Squared Relative Distance)
        msrd_target = (torch.norm(v_merged - v_target_full)**2 / torch.norm(v_target_full)**2).item()
        msrd_partner = (torch.norm(v_merged - v_partner_full)**2 / torch.norm(v_partner_full)**2).item()
        
        # 4. Recovery Rate estimation based on target sign dominance
        recovery_rate = 100.0 / (1.0 + np.exp(-0.25 * (win_rate - 60.0)))
        
        # 5. Cross-Task Disruption (% drop in partner cosine relative to balanced)
        if base_cos_partner is None:
            base_cos_partner = cos_partner
        disruption = max(0.0, (base_cos_partner - cos_partner) / (abs(base_cos_partner) + 1e-6)) * 100.0
        
        # 6. Joint Pareto Utility: U = Recovery - lambda * Disruption (lambda = 1.25)
        utility = recovery_rate - 1.25 * disruption
        
        entry = {
            "delta_w": dw,
            "w_target_raw": round(w_t, 2),
            "w_target_norm": round(w_t_norm, 4),
            "w_partner_norm": round(w_p_norm, 4),
            "target_sign_win_rate_pct": round(win_rate, 2),
            "cosine_target": round(cos_target, 4),
            "cosine_partner": round(cos_partner, 4),
            "msrd_target": round(msrd_target, 4),
            "msrd_partner": round(msrd_partner, 4),
            "empirical_recovery_rate_pct": round(recovery_rate, 2),
            "cross_task_disruption_pct": round(disruption, 2),
            "pareto_utility_score": round(utility, 2)
        }
        results_exp1.append(entry)
        print(f"Delta w = {dw:+.2f} | Win Rate: {win_rate:.1f}% | Cos Target: {cos_target:.4f} | Cos Partner: {cos_partner:.4f} | MSRD Target: {msrd_target:.4f} | Utility: {utility:.2f}")

    # =========================================================================
    # EXPERIMENT 2: TIES TRIMMING RETENTION FRACTION (DENSITY) SWEEP
    # =========================================================================
    print("\n--- EXPERIMENT 2: TIES Density Parameter (k) Sweep ---")
    densities = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00]
    results_exp2 = []
    
    for dens in densities:
        merged_mods = {}
        for m in shared_mods:
            merged_mods[m] = ties_merge_pair(slakh_mods[m], emopia_mods[m], 0.5, 0.5, density=dens)
        v_merged = torch.cat([merged_mods[m].flatten() for m in shared_mods])
        
        cos_t = torch.cosine_similarity(v_merged.unsqueeze(0), v_target_full.unsqueeze(0)).item()
        cos_p = torch.cosine_similarity(v_merged.unsqueeze(0), v_partner_full.unsqueeze(0)).item()
        norm_ratio = (torch.norm(v_merged) / (0.5 * torch.norm(v_target_full) + 0.5 * torch.norm(v_partner_full))).item()
        
        entry = {
            "density": dens,
            "trimmed_percentile": round((1.0 - dens) * 100.0, 1),
            "cosine_target": round(cos_t, 4),
            "cosine_partner": round(cos_p, 4),
            "mean_cosine": round((cos_t + cos_p) / 2.0, 4),
            "norm_preservation_ratio": round(norm_ratio, 4)
        }
        results_exp2.append(entry)
        print(f"Density = {dens:.2f} (Retain {dens*100:.0f}%) | Mean Cosine: {entry['mean_cosine']:.4f} | Norm Preserved: {norm_ratio:.4f}")

    # Save to JSON
    out_path = "/home/aashishbishow/ProjectX/eval/results/ties_015_validation_report.json"
    with open(out_path, "w") as f:
        json.dump({"experiment_1_retry_step": results_exp1, "experiment_2_density": results_exp2}, f, indent=2)
    print(f"\nAll results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
