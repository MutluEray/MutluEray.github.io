# neuroprosthetic-bp-control

**Does a blood-pressure neuroprosthesis need a smarter controller, or just a posture sensor?**
A virtual-patient study of closed-loop epidural spinal cord stimulation for blood pressure after spinal cord injury (SCI).

In 20 held-out virtual patients, adding a posture signal (tilt angle) to the clinical PID controller cut the time systolic blood pressure spent above 150 mmHg by about 85 % (95 % CI 76–97 %). Model-predictive control with the same posture signal reached 89 %; without it, only 24 %. Time in the 110–120 mmHg target barely changed (56–60 % for every tuned PID, LQI and MPC variant; 52 % for the rule-based controller). A 5–10-minute calibration session was enough for the model-based controllers. The benefit disappears when the posture sensor freezes after the patient lies back down.

Project page with figures and an interactive demo: https://mutlueray.github.io/projects/neuroprosthetic-bp-control.html

> Simulation only. The results compare controller designs in a model; they do not predict what any patient would experience.

## What is in here

| Piece | Files |
|---|---|
| Healthy-subject reference data (PhysioNet PRCP): download, beat extraction, event-locked envelopes | `c10b_step0a_prcp.py`, `c10b_step1a_prcp_envelopes.py` |
| CVSim 21-compartment model, compiled from its own C code with three asserted patches (external tilt, two stimulation hooks) | `c10b_step0b_cvsim_inspect.py`, `c10b_cvsim.py`, `c10b_cvsim_ext.c` |
| Verification against CVSim, validation and Bayesian calibration against PRCP | `c10b_step1b_verify.py`, `c10b_step1c_calibrate.py` |
| SCI plant (fitted to Corbett et al. 1971), stimulation actuator, 60-patient virtual population | `c10b_plant.py`, `c10b_step2_sci_population.py` |
| Evaluation scenario, metrics, rule-based and PID controllers | `c10b_harness.py`, `c10b_step3_baselines.py` |
| System identification from simulated calibration sessions (ARX + delay + posture input) | `c10b_sysid.py` |
| LQI, offset-free MPC, prefiltered PID, posture feedforward | `c10b_controllers.py` |
| Equal-budget Bayesian-optimisation tuning per controller family | `c10b_step4b_tune.py` |
| Held-out comparison, pre-specified decision rule, calibration-length sweep | `c10b_step4c_evaluate.py` |
| Robustness: 15 plant, sensor and actuator conditions | `c10b_step5_robustness.py` |
| Figures, captions and demo data | `c10b_step6_showcase.py`, `bp-demo.html` |
| Results (tables, figures) | `results/` |

## Run it

Python 3.11, CPU only (developed on an Apple Silicon laptop, 16 GB). A C compiler is needed (`xcode-select --install` on macOS).

```
python3 -m pip install -r requirements.txt
```

1. Download the CVSim archive from https://physionet.org/content/cvsim/ and save it as `data/cvsim.zip`.
2. Run the steps in order (wall times where measured on the laptop above):

```
python3 c10b_step0a_prcp.py                       # PRCP download + beats
python3 c10b_step0b_cvsim_inspect.py data/cvsim.zip
python3 c10b_step1a_prcp_envelopes.py
python3 c10b_step1b_verify.py                     # builds build/libcvsim21.so on first use
python3 c10b_step1c_calibrate.py                  # Bayesian calibration to PRCP
python3 c10b_step2_sci_population.py              # SCI fit + 60 virtual patients
python3 c10b_step3_baselines.py
python3 c10b_sysid.py                             # ~11 min
for f in C1 C2 C2f C2ff C3 C3ff C4 C4ff; do python3 c10b_step4b_tune.py $f; done   # ~7–15 min each
python3 c10b_step4c_evaluate.py                   # ~11 min
python3 c10b_step4c_evaluate.py --cal-sweep       # ~12 min
python3 c10b_step5_robustness.py --group plant    # ~30 min
python3 c10b_step5_robustness.py --group sensor   # ~35 min
python3 c10b_step5_robustness.py --group robust   # ~25 min
python3 c10b_step6_showcase.py                    # ~15 min
```

To view the demo locally: `cp bp-demo.html results/c10b_step6/ && cd results/c10b_step6 && python3 -m http.server 8000`, then open http://localhost:8000/bp-demo.html.

## Controllers

| Label | Controller |
|---|---|
| C0 | fixed amplitude |
| C1 | rule-based with timers (after Christie et al. 2026) |
| C2 / C2f | PID / PID with measurement prefilter (clinical baseline) |
| C2ff | C2f + posture feedforward |
| C3 / C3ff | LQI on the identified model, without / with posture feedforward |
| C4 / C4ff | offset-free linear MPC on the identified model, without / with posture feedforward |

All families were tuned with the same Bayesian-optimisation budget (8 tuning patients, 25 evaluations, stall stopping) and evaluated on 20 held-out patients. Model-based controllers use only a model identified from a simulated calibration session, never the true plant.

## Main results (20 held-out virtual patients)

| Controller | % time in 110–120 | % time > 150 | amplitude changes |
|---|---|---|---|
| Rule-based (C1) | 51.5 | 0.71 | 60 |
| PID + prefilter (C2f, reference) | 56.3 | 0.58 | 239 |
| PID + posture (C2ff) | 56.2 | 0.09 | 376 |
| LQI + posture (C3ff) | 58.9 | 0.12 | 224 |
| MPC (C4) | 58.8 | 0.44 | 154 |
| MPC + posture (C4ff) | 57.7 | 0.07 | 198 |

Full tables: `results/c10b_step4c/summary.md`, `g1.md`, `cal_sweep.md`, `results/c10b_step5/step5.md`.

## Pre-specified decisions

- **G0 (healthy model credibility):** passed after calibration, with stated limitations (overshoot after rapid tilt, missing slow heart-rate component, weak tilt-down fit, no muscle pump).
- **G-SCI (injury model):** passed at 30° and 45°; 60° was held out and is overpredicted (−47 vs −26 mmHg).
- **G1 (publication):** passed on time above 150 mmHg (MPC + posture vs tuned PID); no controller reached the 30 % threshold on time out of range.
- **G2 (robustness ≥ 50 % in every condition):** failed, because of the frozen-posture-sensor condition.

Two fairness bugs found in the first comparison (tuning stall counter started at evaluation 0; PID had no measurement prefilter) both favoured the model-based controllers. They were fixed before the final comparison.

## Limitations

Simulation only; SCI plant fitted to one 1971 study; stimulation delay and lag assumed and varied (2–12 s, 5–30 s), not measured; the model understates clinical hypertensive episodes; the effect sits in the return-to-lying phase, the least validated part of the model (stress-tested, not validated against human data); no human stimulation data.

## Data and third-party code

- PRCP database (Heldt et al. 2003) and CVSim (Heldt et al. 2010) are from PhysioNet and are **not** redistributed here; the scripts download or read your local copy. Check their licences on PhysioNet.
- SCI targets: Corbett, Frankel & Harris, J Physiol 1971 (PMC1331891).
- Method background: RWTH Aachen MOOC *Learning-based Control* (LQR, learned dynamics, Bayesian tuning, MPC).

## Author

Eray Mutlu — https://mutlueray.github.io
