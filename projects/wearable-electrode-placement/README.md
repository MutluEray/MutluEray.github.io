# Can physics tell you where to put a wearable EEG electrode?

**One-line result.** On a public 15-person dataset with simultaneous scalp, around-ear and in-ear EEG, a template head model predicts *where on the scalp* a two-electrode wearable should sit for auditory attention decoding. It was checked against real electrode pairs the model never saw (rank agreement ρ = 0.68, noise ceiling 0.89), and the placement survives conductivity and co-registration errors. The same model predicts around-ear *signals* better than interpolation, but it cannot yet rank around-ear electrode pairs for decoding (ρ = 0.34 after calibration, below the pre-registered 0.35).

This is a methods study on public data, not a device. Every step had a pre-registered prediction or go/no-go criterion, and the failures are reported with the successes.

---

## Why

A wearable EEG product has room for a few electrodes, not a cap. Choosing where they go is usually done by selecting channels from a full cap recording. That approach can only choose among positions the cap measured, and it says nothing about positions a product might use (behind the ear, between cap sites). A physics forward model can evaluate *any* position. The open question is whether its predictions hold up against real recordings.

## Data

**Ear-SAAD** (Geirnaert, Kappel & Kidmose, *Scientific Reports* 2025; Zenodo [10.5281/zenodo.16536441](https://doi.org/10.5281/zenodo.16536441)):
- 15 participants, 6 × 10-minute trials each, attending to one of two competing talkers.
- Simultaneous scalp EEG (29 channels incl. mastoids), around-ear EEG (19 cEEGrid-style electrodes) and in-ear EEG (12).
- The authors' preprocessed release was used: 1–9 Hz, 20 Hz, speech envelopes included.

**Licence.** The Zenodo page lists CC BY 4.0, but the dataset's own `dataset_description.json` lists CC BY-NC-SA 4.0. This repo treats it as non-commercial and contains **code only, no data or derived signals**.

## Method

| Component | Choice |
|---|---|
| Task | Auditory attention decoding: linear backward decoder, 0–400 ms lags, leave-one-trial-out, 60 s decision windows |
| Head model | MNE `fsaverage`, 3-layer BEM, ico-5 cortical sources (fixed orientation); standard conductivities 0.3 / 0.006 / 0.3 S/m |
| Virtual electrode | Minimum-norm source estimate from the scalp channels, forward-projected to any skin position (a physics-based interpolation) |
| Baselines | Spherical-spline interpolation; nearest electrode; data-driven ridge regression as a ceiling |
| Placement search | All candidate pairs on a 7.4 mm skin grid within 2.5 cm of a cap electrode (≈ 20k pairs 1–3 cm apart, ≈ 92k pairs 3–7 cm apart), scored with a closed-form version of the decoder (agrees with the full pipeline at ρ = 0.98) |
| Generalisation | One layout for every user, chosen leave-one-subject-out; decoder fitted per user |
| Sensor noise | Independent per-electrode noise, band-limited like the data; swept 0–4 µV |

## Results

### 1. Pipeline reproduces the published decoder

Mean accuracy on 60 s windows: scalp **84.0 %** (paper 83.4 %), around-ear **66.7 %** (67.2 %), in-ear **60.6 %** (61.1 %).

### 2. Around the ear: signals yes, decoding rank not yet

- **Signals.** Around-ear signals were predicted from scalp EEG alone, with the mastoid electrodes withheld (they sit under the cEEGrid). Median correlation with the measured signals:
  - physics 0.29
  - spline 0.12
  - nearest electrode 0.14
  - data-driven ceiling 0.37

  Physics beats the spline in 12 of 15 subjects (p = 0.0004). The gain is largest at the bottom of the arc, furthest below the cap. With the mastoids included, physics and spline tie (0.31 vs 0.32). Physics helps where it has to extrapolate.
- **Decoding rank.** For the 81 within-ear electrode pairs (13 subjects), physics ranks the pairs at ρ = 0.22.
  - The ranking is not mainly limited by noise: a data-driven ridge model reaches 0.46, and the split-half noise ceiling is 0.70.
  - The errors are geometric: physics over-rates the top-front electrodes.
  - Fitting the template's arc geometry leave-one-subject-out raises ρ to **0.34**, just below the pre-registered threshold of 0.35. **NO-GO for around-ear placement claims**, kept as decided.
  - The fitted parameters sit at the edge of the search range, and they absorb head-model error (fsaverage has no pinna), so they are not measured electrode positions.

### 3. On the cap: the model ranks real pairs it never saw

Each of 55 real scalp pairs (≤ 7 cm) was removed in turn and predicted from the remaining channels.

| | Physics | Spline | Noise ceiling |
|---|---|---|---|
| Rank agreement, group means | **0.68** | 0.80 | 0.89 |
| Rank agreement, per-subject median | **0.65** | 0.60 | — |
| Bipolar signal r (subject medians) | **0.61** | 0.55 | — |

- Physics is better on signals (p = 6 × 10⁻⁵) and gets the size of the decoding effect right (Δr 0.019 vs measured 0.017).
- The spline is better on the group-mean ranking.
- On the cap, physics is a good interpolator, not a uniquely better one. Its distinct value is extrapolation.

### 4. Placement

- **Population optimum:** over the left lateral centro-parietal scalp, around CP5 between C3 and P7. Physics and spline optima agree within 0.9 cm.
- **The optimum is broad:** off-grid refinement adds 0.6 %. With 1 µV sensor noise, a 3–7 cm pair keeps ≥ 97 % of its decoding score with each electrode misplaced by up to 1 cm.
- **Miniaturisation:**
  - Without sensor noise, 1–3 cm pairs lose nothing against 3–7 cm pairs. This is an artifact: virtual electrodes are noise-free.
  - With independent electrode noise, 1–3 cm pairs lose 21 % at 1 µV, 33 % at 2 µV and 43 % at 4 µV.
  - A data-based upper bound on that noise is 1.55 µV.
  - Under noise, the best pair in each class moves to the widest allowed spacing.
- **Against the best real 10-20 pair (P3–C3)**, both pairs run through physics with 1 µV noise, held-out (leave-one-subject-out):
  - The model-placed pair is better in 12 of 15 subjects: median paired gain +1.7 accuracy points (p = 0.006).
  - The pre-registered statistic (difference of medians, +5.0 points, threshold ≤ 3) fails, and is reported as failed.
  - At 1 µV, physics + noise reproduces the real pair's measured accuracy (median 70.0 % vs 70.0 %). This is a post-hoc check of the noise level, not a pre-registered one.

### 5. Credibility

| Perturbation | 3–7 cm optimum shift | Regret* | On-cap ρ |
|---|---|---|---|
| Skull 0.0042 / 0.0105 / 0.015 S/m, scalp 0.43 S/m | ≤ 0.7 cm (one grid step) | ≤ 0.1 % | 0.69–0.70 |
| ~1 cm rigid co-registration error (10 draws) | median 1.2 cm (max 1.9) | median 0.9 % (max 1.8 %) | 0.68–0.70 |

\*Regret: decoding lost if the perturbed model's optimum is used under the baseline model.

1–3 cm pairs are more fragile. Under co-registration error, 2 of 10 draws jumped 3–4 cm to a different near-equal optimum, costing up to 5 %.

### Pre-registered predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Reproduction within 5 points of the paper | True (≤ 0.6 points) |
| 2 | Physics beats spline for off-cap signals, most below the cap | True |
| 3 | Off-cap decoding rank preserved (ρ > 0.5) | **False** (0.22; 0.34 after calibration, go ≥ 0.35 not met) |
| 4 | Optimum near FC5/C3/T7 | Partly true (left lateral, but centro-parietal) |
| 4′ | Population optimum within ~2 cm of CP5/C3/P7 | True |
| 5 | Continuous placement gains ≤ 20 % Δr and ≤ 3 accuracy points | **False** as tested (+3.3 points; confounded, see 4) |
| 6 | 1–3 cm pairs lose ≥ 30 % vs 3–7 cm (noise-free model) | **False** (−0.5 %; untestable without sensor noise) |
| 6′ | With sensor noise, ≥ 30 % loss at σ ≤ 1 µV | **False** (21 % at 1 µV; 33 % at 2 µV) |
| 7 | 3–7 cm pairs keep ≥ 90 % within 1 cm misplacement | True (97 %) |
| 8 | Physics-vs-physics gain over the best real pair ≤ 3 points (difference of medians) | **False** (+5.0; paired median +1.7) |
| 9 | Optimum moves < 1.5 cm under conductivity changes; on-cap ρ ≥ 0.6 | True |
| 10 | ~1 cm co-registration error: median shift < 1.5 cm, regret < 5 % | True |

## Limitations

- **Information bound.** Virtual electrodes are linear combinations of the 29 recorded scalp channels. The model can say where a small device captures what the cap sees. It cannot find information the cap missed.
- **Virtual, not measured.** Accuracies for model-placed pairs come from physics-predicted signals (plus modelled noise), not from a device. They were checked on the cap against real pairs; nothing here replaces a recording with the proposed layout.
- **Template head.** fsaverage with standard conductivities; no pinna, no ear canal (in-ear EEG is outside the physics). Individual anatomy is not tested.
- **Around-ear geometry.** There are no measured electrode coordinates; the dataset manual has a sketch only. The template arc and its calibration absorb head-model error.
- **Noise model.** Independent band-limited noise at a nominal 1 µV (not taken from a source; a data-based upper bound of 1.55 µV is reported). Electrode–skin effects such as motion, drift and dry contact are not modelled.
- **Small sample, one task.** 15 participants (13 for around-ear decoding), one session each, auditory attention only. Placement results are task-specific.
- **Statistics.** Several pre-registered thresholds were missed narrowly (0.343 vs 0.35; +3.3 vs ≤ 3 points). They are reported as missed, and no thresholds were moved.

## Reproduce

Apple Silicon Mac, 16 GB, CPU only; `python3` with `numpy scipy scikit-learn matplotlib mne pymatreader remotezip requests`.

```
python3 step0_inspect_data.py        # download preprocessed data, read BIDS metadata remotely
python3 step1_baseline_aad.py        # reproduce published AAD accuracies
python3 step2_headmodel.py           # fsaverage BEM, candidate grid, lead fields
python3 step3_validity.py            # scalp -> around-ear prediction (answer key)
python3 step3b_diagnostics.py        # noise ceiling, ridge ceiling, subject-level tests
python3 step3c_calibrate_template.py # LOSO template calibration (go/no-go for off-cap)
python3 step4a_oncap_validity.py     # on-cap answer key (leave-pair-out)
python3 step4b_oncap_placement.py    # exhaustive on-cap placement, LOSO
python3 step4c_noise_tolerance.py    # sensor noise, fair comparison, tolerance, boundary
python3 step5_credibility.py         # conductivity and co-registration perturbations
python3 step6_figures.py             # figures for the write-up
```

Each script prints its result and writes `results/*.json` and `figures/*.png`. The total runtime is under an hour; step 5 is the longest at ~15 min.

## References

- Geirnaert, Kappel, Kidmose. A direct comparison of simultaneously recorded scalp, around-ear, and in-ear EEG for neural selective auditory attention decoding to speech. *Scientific Reports* 2025. [arXiv:2505.14478](https://arxiv.org/abs/2505.14478)
- Mundanad Narayanan, Zink, Bertrand. EEG miniaturization limits for stimulus decoding with EEG sensor networks. *J. Neural Eng.* 2021. [doi:10.1088/1741-2552/ac2629](https://doi.org/10.1088/1741-2552/ac2629)
- Strypsteen, Bertrand. End-to-end learnable EEG channel selection for deep neural networks with Gumbel-softmax. [arXiv:2102.09050](https://arxiv.org/abs/2102.09050)
- Kappel, Makeig, Kidmose. Ear-EEG forward models: improved head-models for ear-EEG. *Front. Neurosci.* 2019. [doi:10.3389/fnins.2019.00943](https://doi.org/10.3389/fnins.2019.00943)
- Meiser, Tadel, Debener, Bleichner. The sensitivity of ear-EEG: evaluating the source-sensor relationship using forward modeling. *Brain Topogr.* 2020. [doi:10.1007/s10548-020-00793-2](https://doi.org/10.1007/s10548-020-00793-2)
- Beltrachini et al. Optimal design of on-scalp electromagnetic sensor arrays for brain source localisation. *Hum. Brain Mapp.* 2021.
- Gramfort et al. MNE software for processing MEG and EEG data. *NeuroImage* 2014.
