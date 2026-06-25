use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

// ── Cache key ─────────────────────────────────────────────────────────────────
fn weights_key(weights: &[f32]) -> Vec<u64> {
    weights
        .iter()
        .map(|&w| (w * 1_000_000.0).round() as u64)
        .collect()
}

// ── TIES math ─────────────────────────────────────────────────────────────────
fn trim_vec(data: &[f32], density: f32) -> Vec<f32> {
    if density >= 1.0 {
        return data.to_vec();
    }

    let k = ((data.len() as f32) * (1.0 - density)) as usize;
    if k == 0 {
        return data.to_vec();
    }

    // O(n) partial select instead of O(n log n) sort
    let mut magnitudes: Vec<f32> = data.iter().map(|x| x.abs()).collect();
    let threshold = *magnitudes
        .select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap())
        .1;

    data.iter()
        .map(|&x| if x.abs() >= threshold { x } else { 0.0 })
        .collect()
}

fn elect_sign_vec(tensors: &[Vec<f32>]) -> Vec<f32> {
    let n = tensors[0].len();
    let mut sums = vec![0.0f32; n];
    for t in tensors {
        for (s, &v) in sums.iter_mut().zip(t.iter()) {
            *s += v;
        }
    }
    sums.iter()
        .map(|&s| if s >= 0.0 { 1.0 } else { -1.0 })
        .collect()
}

fn disjoint_merge_vec(tensors: &[Vec<f32>], elected_sign: &[f32]) -> Vec<f32> {
    let n = tensors[0].len();
    let mut merged = vec![0.0f32; n];
    let mut count  = vec![0.0f32; n];

    for t in tensors {
        for i in 0..n {
            if t[i] * elected_sign[i] > 0.0 {
                merged[i] += t[i];
                count[i]  += 1.0;
            }
        }
    }

    merged.iter().zip(count.iter())
        .map(|(&m, &c)| if c > 0.0 { m / c } else { 0.0 })
        .collect()
}

// ── Stateful struct ───────────────────────────────────────────────────────────
#[pyclass]
struct TIESMerger {
    density: f32,

    // [module_idx][adapter_idx][element]
    task_vectors: Vec<Vec<Vec<f32>>>,

    // cache: weights_key -> [module_idx][adapter_idx][element] (trimmed, sparse)
    trimmed_cache: HashMap<Vec<u64>, Vec<Vec<Vec<f32>>>>,
}

#[pymethods]
impl TIESMerger {
    /// task_vectors_per_module: list[list[np.ndarray]]
    ///   outer = modules, inner = one delta per adapter
    #[new]
    fn new(
        task_vectors_per_module: Vec<Vec<PyReadonlyArray1<f32>>>,
        density: f32,
    ) -> Self {
        let task_vectors = task_vectors_per_module
            .iter()
            .map(|module_vecs| {
                module_vecs
                    .iter()
                    .map(|a| a.as_slice().unwrap().to_vec())
                    .collect()
            })
            .collect();

        TIESMerger {
            density,
            task_vectors,
            trimmed_cache: HashMap::new(),
        }
    }

    /// Merge base weights with routing weights.
    ///
    /// - Trim:  parallel across modules, cached per weight config (O(n) per tensor)
    /// - Sign + merge: parallel across modules, runs every call (cheap on sparse tensors)
    ///
    /// base_weights: list[np.ndarray]  — one float32 array per module, CPU
    /// weights:      list[float]       — one per adapter, same order as task_vectors
    ///
    /// Returns list[np.ndarray] — merged weights, same order as base_weights
    fn merge<'py>(
        &mut self,
        py: Python<'py>,
        base_weights: Vec<PyReadonlyArray1<'py, f32>>,
        weights: Vec<f32>,
    ) -> Vec<Py<PyArray1<f32>>> {
        let key = weights_key(&weights);

        // ── Trim: compute once, cache forever for this weight config ──────
        if !self.trimmed_cache.contains_key(&key) {
            let density = self.density;

            let trimmed: Vec<Vec<Vec<f32>>> = self
                .task_vectors
                .par_iter()
                .map(|module_vecs| {
                    module_vecs
                        .iter()
                        .zip(weights.iter())
                        .map(|(tau, &w)| {
                            let weighted: Vec<f32> = tau.iter().map(|&x| x * w).collect();
                            trim_vec(&weighted, density)
                        })
                        .collect()
                })
                .collect();

            self.trimmed_cache.insert(key.clone(), trimmed);
        }

        let trimmed_by_module = self.trimmed_cache.get(&key).unwrap();

        // ── Sign election + disjoint merge: parallel across modules ───────
        // Already fast — trimmed tensors are 20% sparse
        let bases: Vec<&[f32]> = base_weights
            .iter()
            .map(|a| a.as_slice().unwrap())
            .collect();

        let results: Vec<Vec<f32>> = bases
            .par_iter()
            .zip(trimmed_by_module.par_iter())
            .map(|(base, trimmed)| {
                let signs = elect_sign_vec(trimmed);
                let delta = disjoint_merge_vec(trimmed, &signs);
                base.iter().zip(delta.iter()).map(|(&b, &d)| b + d).collect()
            })
            .collect();

        results
            .into_iter()
            .map(|v| v.into_pyarray(py).into())
            .collect()
    }

    /// Free trimmed delta cache to reclaim RAM.
    /// Returns number of entries freed.
    fn clear_cache(&mut self) -> usize {
        let n = self.trimmed_cache.len();
        self.trimmed_cache.clear();
        n
    }

    fn cache_size(&self) -> usize {
        self.trimmed_cache.len()
    }
}

#[pymodule]
fn ties_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TIESMerger>()?;
    Ok(())
}