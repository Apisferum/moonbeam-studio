import os
import time
from typing import List, Dict, Any

# Optional torch import
try:
    import torch
except ImportError:
    torch = None

def get_peak_gpu_memory_mb() -> float:
    """Returns the peak GPU memory allocated in megabytes."""
    if torch is not None and torch.cuda.is_available():
        try:
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        except Exception:
            return 0.0
    return 0.0

def get_system_memory_usage_mb() -> float:
    """Returns the current CPU RAM usage of the process in megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))
    except ImportError:
        return 0.0

def compute_efficiency_stats(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes efficiency stats across all sections from a PieceTrace dictionary.
    """
    latencies = []
    total_tokens = 0
    total_attempts = 0
    reused_kv = 0
    
    for s in sections:
        latencies.append(s.get("latency_ms", 0.0))
        if s.get("kv_cache_reused"):
            reused_kv += 1
            
        attempts = s.get("attempts", [])
        total_attempts += len(attempts)
        for attempt in attempts:
            total_tokens += attempt.get("token_count", 0)
            
    total_latency_sec = sum(latencies) / 1000.0
    token_gen_rate = total_tokens / total_latency_sec if total_latency_sec > 0 else 0.0
    
    return {
        "total_latency_ms": float(sum(latencies)),
        "mean_section_latency_ms": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "total_attempts": int(total_attempts),
        "mean_attempts_per_section": float(total_attempts / len(sections)) if sections else 0.0,
        "token_generation_rate_per_sec": float(token_gen_rate),
        "kv_cache_reuse_ratio": float(reused_kv / len(sections)) if sections else 0.0
    }
