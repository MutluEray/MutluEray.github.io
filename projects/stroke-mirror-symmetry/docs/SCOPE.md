# Scope: Does the brain's mirror symmetry survive stroke?

**One-line claim.** Many motor-imagery (MI) decoders assume that the left and right hemispheres mirror each other. This study tests whether that prior still helps, or starts to hurt, when a stroke has made the hemispheres unequal.

**Status.** Hypotheses and protocol written 2026-11-17, before any stroke analysis was run. Revised the same day, still before any stroke result: the primary model became a tangent-space linear decoder (§6) after EEGNet proved too slow for repeated cross-validation on a laptop.

**What is new and what is not.** This is not a replication and not a new method. Symmetry priors for MI decoding exist (EEGSym, Channel Reflection, mirror losses), and so does lesion-side flipping in stroke EEG analysis. The contribution is an empirical test that has not been reported: do healthy-brain symmetry priors still help in stroke patients, and does lesion-aware canonicalization do better? The test runs under one controlled protocol, with a matched healthy reference and an interpolation from unconstrained to exactly equivariant.

---

## 1. The question

The MIT 6.S980 lecture on geometric deep learning ends with: *"for groups other than the translation group, it is not obvious that we want perfect equivariance."* This project makes that concrete for one small group and one clinical population.

- **Group:** Z2 = {e, m}, where m is the reflection across the mid-sagittal plane. It swaps each left electrode with its right partner (C3↔C4, FC5↔FC6, …) and leaves midline electrodes (Cz, FCz, …) where they are.
- **Task:** left-hand vs right-hand MI. Mirroring the brain should swap the label.
- **Population where it should hold:** healthy adults (contralateral ERD).
- **Population where it might break:** stroke patients. The lesion weakens ERD over one hemisphere, and the contralesional hemisphere can take over part of the work for the affected hand (Mansour et al., 2022).

## 2. Prior work and gap

Mirror symmetry is already used in three ways in MI decoding:

| Form | Example | Evaluated on |
|---|---|---|
| Architecture | EEGSym: siamese network across the mid-sagittal plane (Pérez-Velasco et al., IEEE TNSRE 2022) | 5 public datasets, 280 subjects, all healthy-population datasets |
| Augmentation | Channel Reflection: swap hemispheres, flip MI label (Wang et al., 2024); braindecode `ChannelsSymmetry` (Rommel et al., 2022) | public BCI datasets (healthy) |
| Loss | Mirror contrastive loss (Luo et al., 2024) | healthy |

Separately, stroke EEG studies routinely *flip* the data of left-lesion patients so that every lesion sits in the same hemisphere. That is an analysis convention, and in geometric-deep-learning terms it is canonicalization.

**Gap (as far as four searches, one of them extended, could find):** no study tests whether symmetry priors built for healthy brains help or hurt cross-patient MI decoding in stroke, or compares them with lesion-side canonicalization. **To do:** a 10-minute Google Scholar pass on "EEGSym stroke", "channel reflection stroke", and "symmetry augmentation stroke motor imagery" before publishing.

## 3. Theory (in the course's language)

- **Input representation.** ρ_in(m) acts on an EEG trial x ∈ R^(C×T) by a channel permutation matrix P (P² = I). This is a valid representation of Z2.
- **Output representation.** For left/right MI the binary logit transforms by the **sign representation**: ρ_out(m) = −1. (For a task without lateralized labels it would be the trivial representation, which gives invariance.)
- **Equivariance.** f(Px) = −f(x).
- **Exact construction (lifting + projection).** Lift x to a function on the group, (g(x), g(Px)). m acts on this by swapping the two entries (the regular representation). Then project onto the sign irrep: f(x) = g(x) − g(Px). This mirrors the lecture's "lifting layer → projection layer" recipe. Mean-pooling (the trivial irrep) would give invariance instead.
- **Every network splits in two.** Any g can be written as g = A + S, with A(x) = ½(g(x) − g(Px)) (equivariant) and S(x) = ½(g(x) + g(Px)) (invariant). Exact equivariance forces S ≡ 0. Augmentation pushes S toward 0. A soft penalty λ·E[S(x)²] interpolates between the two. The fraction of a trained model's decision that comes from S gives a model-based asymmetry index.
- **Canonicalization** is the alternative to constraining the function. Use a known group element (the patient's hemiplegic side) to move every input into one frame, then fit an unconstrained model (Kaba et al., ICML 2023, for the general idea).
- **Pre-processing must commute with P,** otherwise the symmetry test is meaningless. We use per-channel filtering, resampling, common-average reference over a mirror-closed channel set, and one scalar scale per subject. All of these commute with P. Per-channel z-scoring would remove the C3-vs-C4 power difference that carries the MI signal, so it is not used.

## 4. Hypotheses (fixed before stroke analysis)

Within-stroke tests are primary because they carry no dataset confound.

- **H1 (primary, within stroke).** Lesion-side canonicalization gives higher cross-patient balanced accuracy than the healthy-brain symmetry priors (reflection augmentation, exact equivariance).
- **H2 (primary, within stroke).** Exact equivariance does not beat the unconstrained model in stroke patients.
- **H3 (reference, healthy).** In matched healthy subjects, reflection augmentation and exact equivariance beat the unconstrained model, which replicates the premise of prior work.
- **H4 (cross-dataset, confounded).** The gain from symmetry priors is smaller in stroke than in healthy. This contrast is confounded by device, cue design and protocol, and is reported with that caveat.
- **Exploratory.** (a) The accuracy-vs-a curve peaks at a larger a (more mirror-invariant weight allowed) in stroke than in healthy. (d) The effects differ between supratentorial and infratentorial lesions. (b) The per-patient gain from equivariance falls with NIHSS and with model-free ERD laterality. (c) The model-based asymmetry index (S-fraction) correlates with clinical scores.

## 5. Data

| | Stroke | Healthy reference |
|---|---|---|
| Source | Liu et al., *Sci Data* 2024; figshare 10.6084/m9.figshare.21679035 (CC BY 4.0) | PhysioNet EEG Motor Movement/Imagery (Schalk et al., 2004), runs 4/8/12 |
| Subjects | 50 acute ischemic stroke patients (1–30 days post-stroke), all originally right-handed | 50 randomly drawn subjects (fixed seed) from 104 usable, to match N |
| Channels | 29 EEG on a 10-10 cap, ref CPz | 64; restricted to the stroke montage ∩ PhysioNet, mirror-closed |
| Clinical | hemiplegic side, NIHSS, MBI, mRS | none |

Resolved from the downloaded files (`participants.tsv`, EDF headers):
- **Paralysis side:** 28 left / 22 right (the paper says 23 R/27 L; figshare matches the files).
- **Trials:** 40 per patient, 20 per class, 8-s trials (2 s instruction, 4 s imagery with a hand video, 2 s break).
- **Channels:** 29 EEG plus CPz (reference, dropped), 2 EOG and a marker channel. All 29 are mirror-closed and present in PhysioNet, so `shared` = 29 channels and `shared_sensorimotor` = FC3, FCz, FC4, C3, Cz, C4, CP3, CP4.
- **Handedness:** 2 patients are left-handed, although the paper says all were right-handed.
- **Labels:** the EDFs carry only a stage marker. Left/right comes from the task-level `events.tsv`, which gives a fixed alternating order. `prep_stroke.py` checks every recording's timing against it and runs a within-subject label sanity check. If that fails, labels come from `sourcedata` instead.
- **Lesion sites:** mostly brainstem (pons, medulla), cerebellum, or subcortical (basal ganglia, paraventricular, thalamus). Only ~10 of 50 involve cortex. Many lesions therefore leave the cortex intact while cutting its output, which weakens the prior expectation that cortical ERD symmetry breaks. This is reported as a limitation, with an exploratory supratentorial vs infratentorial split.

## 6. Treatments

**Primary model (`run_ts.py`): Riemannian tangent-space logistic regression.**
- Each trial becomes a covariance matrix per band (mu 8–13 Hz, beta 13–30 Hz).
- Covariances are re-centred per subject, label-free (Zanini et al., 2018), then mapped to the tangent space at the identity.
- The mirror acts on tangent vectors by a permutation Π. Each vector splits into an antisymmetric part t_a = (t − Πt)/2 (sign irrep) and a symmetric part t_s = (t + Πt)/2 (trivial irrep). This is the isotypic decomposition from the lecture.
- A linear decoder is exactly equivariant if and only if it uses only t_a and has no intercept. Each treatment is therefore a statement about how much weight t_s may carry.
- Fits take seconds and need no training epochs or seeds.

| Treatment | Linear decoder | Symmetry assumption |
|---|---|---|
| `none` | f = w·t + b | none |
| `augment` | `none` trained on {t, y} ∪ {Πt, 1−y} | healthy-brain, approximate |
| `equivariant` | f = w·t_a | healthy-brain, exact |
| `soft_a` (a = 0.5, 0.2, 0.1, 0.05) | f = w₁·t_a + w₂·(a·t_s) + b; the L2 penalty on t_s weights grows as 1/a² | adjustable (a = 1 is `none`, a → 0 is `equivariant`) |
| `canonical` (stroke only) | mirror right-hemiplegic patients and flip their labels, then `none` | lesion-aware |

**Secondary model (`run_cv.py`): EEGNet-sized CNN** with the same treatments (`equivariant` = g(x) − g(Px), `soft` = penalty λ·E[(f(x) + f(Px))²]). It is run only on the final comparisons, with tight early stopping, to check that conclusions do not depend on the linear model.

## 7. Evaluation

- **Cross-subject.** 10-fold grouped CV over subjects (5 test subjects per fold). Folds are fixed across treatments, so comparisons are paired on identical trials. The regularisation C is chosen by 3-fold grouped inner CV on the training subjects.
- **EEGNet early stopping** (secondary model), on validation subjects held out from the training fold:
  - stall: no improvement in validation loss for 8 epochs;
  - stuck at chance: best validation balanced accuracy < 0.55 at epoch 12;
  - success: validation balanced accuracy ≥ 0.95;
  - maximum 60 epochs.
- **Metric:** per-subject balanced accuracy (and AUC).
- **Statistics:**
  - paired Wilcoxon signed-rank tests across subjects, with the mean Δ and a bootstrap 95% CI (10,000 resamples over subjects);
  - H4: Mann–Whitney U on per-subject Δ between cohorts, plus a bootstrap CI;
  - exploratory correlations: Spearman with permutation p;
  - Holm correction within the primary family (H1–H3).
- **Power.** The minimum detectable effect is computed from the observed per-subject Δ variance after the first full run and reported. Effects below it are called inconclusive, not null.

**Controls (pre-specified):**
1. **Sensorimotor-only channels (FC/C/CP).** Both datasets use lateralized visual cues: PhysioNet shows the target on one side of the screen, and the stroke protocol plays a video of the cued hand. Lateralized visual responses are also mirror-symmetric, so occipital channels could make the symmetry prior look useful for non-motor reasons.
2. **Matched N** (50 vs 50) and identical pre-processing and epoch length across cohorts.
3. **Classical anchors,** to check that the deep pipeline is not broken. The dataset paper reports CSP + LDA at 55.6% on its own protocol.

## 8. Planned figures

1. Concept: a left-MI ERD topomap, its mirror image, and how a lesion breaks the pairing.
2. Main result: per-subject Δ balanced accuracy vs `none`, for each treatment and cohort (dots + mean ± CI).
3. Accuracy vs equivariance strength λ (from `none` to exact), healthy vs stroke.
4. Class-average mu/beta ERD topomaps: healthy, stroke, stroke after canonicalization.
5. Per-patient gain from equivariance vs NIHSS.
6. Website: an interactive scalp map with a mirror toggle and λ slider, if it adds something beyond the static figures.

## 9. Risks and how they will be reported

- **Near-chance decoding in acute stroke** (published baselines are 55–61%). If every treatment sits at chance, that is the finding, and it gets reported with the power analysis.
- **Few trials per patient** make per-patient accuracies noisy (±8 pp at 40 trials). Inference is at the population level, and per-patient plots show CIs.
- **Hemiplegic side is a proxy for lesion hemisphere.** No lesion maps are available.
- **The cross-dataset contrast (H4) is confounded,** so the conclusions rest on the within-stroke tests.
- **Acute patients only.** A chronic-stroke MI dataset (2026) could serve as replication if time allows.

## 10. Timeline (≈7 working days)

| Day | Work |
|---|---|
| 0 | Scope, repo, PhysioNet prep, stroke download + inspection |
| 1 | Stroke loader, channel harmonization, sanity checks (ERD topomaps, classical anchors) |
| 2 | Main runs, both cohorts, all treatments (overnight) |
| 3 | λ sweep + sensorimotor-only control |
| 4 | Statistics and figures |
| 5 | Write-up (`docs/WRITEUP.md`) |
| 6 | Website section on projects.html |
| 7 | Buffer and review |

## References

- MIT 6.S980 *Machine Learning for Inverse Graphics* (Fall 2023), lecture "Representation Theory & Symmetries" (slides adapted from E. Bekkers, UvA). Self-study of public materials.
- Pérez-Velasco S. et al. EEGSym. *IEEE TNSRE* 30:1766–1775, 2022. doi:10.1109/TNSRE.2022.3186442
- Wang Z. et al. Channel Reflection: knowledge-driven data augmentation for EEG-based BCIs. arXiv:2412.03224, 2024.
- Rommel C. et al. Data augmentation for learning predictive models on EEG: a systematic comparison. arXiv:2206.14483, 2022.
- Luo J. et al. Mirror contrastive loss based sliding window transformer for subject-independent MI EEG recognition. arXiv:2409.00130, 2024.
- Mansour S. et al. Exploring the ability of stroke survivors in using the contralesional hemisphere to control a BCI. *Sci Rep*, 2022. doi:10.1038/s41598-022-20345-x
- Liu H. et al. An EEG motor imagery dataset for brain computer interface in acute stroke patients. *Sci Data*, 2024. doi:10.1038/s41597-023-02787-8
- Schalk G. et al. BCI2000. *IEEE TBME* 51:1034–1043, 2004 (PhysioNet EEGMMIDB).
- Lawhern V. et al. EEGNet. *J Neural Eng* 15:056013, 2018.
- Kaba S.-O. et al. Equivariance with learned canonicalization functions. *ICML*, 2023.
- Zanini P. et al. Transfer learning: a Riemannian geometry framework with applications to BCIs. *IEEE TBME* 65:1107–1116, 2018.
- Finzi M. et al. Residual pathway priors for soft equivariance constraints. *NeurIPS*, 2021.
