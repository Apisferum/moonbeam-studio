import numpy as np
from scipy import stats
from collections import defaultdict
from typing import List, Dict, Any

def drift_slope(section_scores: List[float]) -> Dict[str, float]:
    """
    Fits a linear regression line over a sequence of section-wise metrics.
    section_scores[i] = quality metric for section i.
    """
    if len(section_scores) < 2:
        return {"slope": 0.0, "p_value": 1.0, "r2": 0.0}
        
    x = np.arange(len(section_scores))
    slope, intercept, r, p, se = stats.linregress(x, section_scores)
    return {
        "slope": float(slope),
        "p_value": float(p),
        "r2": float(r**2)
    }

def rejection_rate_vs_index(traces: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Calculates the fraction of attempts rejected by the hard refiner,
    binned by section index, across multiple piece traces.
    """
    bucket = defaultdict(list)
    for piece in traces:
        for s in piece.get("sections", []):
            attempts = s.get("attempts", [])
            if not attempts:
                continue
            # A section is accepted on some attempt index.
            # Total rejections is the number of attempts that were not accepted.
            rejected = sum(1 for a in attempts if not a.get("accepted", False))
            bucket[s["section_idx"]].append(rejected / len(attempts))
            
    return {int(i): float(np.mean(v)) for i, v in bucket.items()}
