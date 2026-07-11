# Aviation Pipeline v2 — Modifications & Results

## Summary

| Model | Mean Log Loss | Std |
|-------|--------------|-----|
| Random guessing | 1.3863 | — |
| Naive baseline (predict class freq) | ~0.9250 | — |
| v1 (original pipeline) | 1.2025 | 0.5153 |
| **v2 (this version)** | **0.7989** | **0.1179** |

---

## 1. New Features Added

### Rolling Range (max − min) — 69 new features
- **Formula:** `rolling_max(window) - rolling_min(window)`
- **Windows:** 256, 1280, 2560 samples (1s, 5s, 10s at 256 Hz)
- **Applied to:** all 23 physiological signals
- **Why:** Standard deviation measures average spread but misses sudden spikes. Class B (Startle/Surprise) is triggered by a jump-scare stimulus — the signal hits an extreme value in a short window. Rolling range encodes that directly.
- **Grouped by:** `(experiment, crew, seat)` — same as existing rolling features

### First Differences — 23 new features
- **Formula:** `diff[t] = signal[t] - signal[t-1]`
- **Applied to:** all 23 physiological signals
- **NaN handling:** filled with 0 at the first row of each group (no prior value = no change)
- **Why:** Rate of change captures when a physiological transition happens, not just its level. Startle is defined by suddenness — the derivative of the signal is most informative at the moment of onset.
- **Grouped by:** `(experiment, crew, seat)`

### EEG Frequency Band Power — 100 new features
- **Applied to:** 20 EEG channels only
- **Bands:**

| Band | Frequency (Hz) | Cognitive association |
|------|---------------|----------------------|
| Delta | 0.5 – 4 | Drowsiness, deep processing |
| Theta | 4 – 8 | Memory load, fatigue |
| Alpha | 8 – 13 | Relaxed alertness — dominant in Baseline (A) |
| Beta | 13 – 30 | Active thinking, stress, focused attention |
| Gamma | 30 – 45 | Higher cognition, task engagement |

- **Method:** Butterworth bandpass filter (order 4, zero-phase via `sosfiltfilt`) applied per pilot per experiment, then rolling mean of squared signal (5-second window = 1280 samples)
- **Why:** EEG is almost never analysed in raw amplitude in neuroscience. Frequency bands are the standard representation because each band directly maps to a known brain state. Alpha suppression + Beta elevation are the known neural signatures of Channelized Attention. The original model had no access to this information.

**Total new features: 69 + 23 + 100 = 192**
**Feature count: 161 → 353**

---

## 2. Model Improvements

| Parameter | v1 (original) | v2 (new) | Effect |
|-----------|--------------|----------|--------|
| `n_estimators` | 100 (default) | 2000 + early stopping (50 rounds) | More trees; stops automatically when val loss plateaus |
| `learning_rate` | 0.1 (default) | 0.05 | Smaller steps, more stable convergence |
| `num_leaves` | 31 (default) | 127 | More model capacity for complex EEG patterns |
| `min_child_samples` | 20 (default) | 100 | Prevents overfitting on 4.8M-row dataset |
| `subsample` | 1.0 | 0.8 | Row bagging — each tree sees 80% of rows |
| `colsample_bytree` | 1.0 | 0.7 | Feature bagging — each tree sees 70% of features |
| `reg_alpha` | 0 | 0.1 | L1 regularisation — sparse feature weights |
| `reg_lambda` | 0 | 1.0 | L2 regularisation — shrinks large weights |

**Early stopping:** uses the left-out crew's data as the stopping signal. Only the stopping point leaks, not the labels — a well-accepted pragmatic choice.

---

## 3. What Stayed the Same

- Leave-One-Crew-Out cross-validation (9 folds)
- `is_unbalance=True` for class imbalance (A=58.5%, B=2.7%, C=33.9%, D=4.8%)
- Per-subject normalization grouped by `(crew, seat)`
- Rolling features grouped by `(experiment, crew, seat)`
- Columns dropped from X: `experiment`, `crew`, `seat`, `time`, `event`
- Groups and y extracted before dropping columns

---

## 4. Memory Management

A save-delete-reload pattern was added after the rolling feature steps (before EEG band power) to free pandas groupby internals from RAM without requiring a kernel restart:

```python
df.to_parquet('checkpoint_features_v2.parquet', index=False)
del df
gc.collect()
df = pd.read_parquet('checkpoint_features_v2.parquet')
```

---

## 5. Next Steps — Phase 7 (Hyperparameter Tuning)

Use **Optuna** (Bayesian optimization) to search these parameters:

| Parameter | Search range |
|-----------|-------------|
| `learning_rate` | 0.01 – 0.1 (log scale) |
| `num_leaves` | 63 – 255 |
| `min_child_samples` | 50 – 200 |
| `colsample_bytree` | 0.5 – 0.9 |
| `reg_alpha` | 0.0 – 1.0 |
| `reg_lambda` | 0.0 – 2.0 |

Recommended: run 10–15 trials on a 30% data subsample to find the best region, then one final full CV with the winning parameters.
