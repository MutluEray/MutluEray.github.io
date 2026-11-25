# C10-B — CVSim plant: verification and validation (G0)

Build + load: 1.3 s. Parameters: CVSim default (initial.c + mapping_ptr, as used by CVSim's main and GUI). Both reflexes on.

## Baseline (supine)

Mean over 250–300 s: SAP 117.0, MAP 95.0, DAP 72.3 mmHg, HR 65.5 bpm. Drift vs 200–250 s: MAP -0.00 mmHg, HR +0.00 bpm.

## Timing

300 s simulated: 0.40 s wall. 30 min simulated: 2.2 s wall.

## Verification: external-angle tilt vs CVSim's original tilt

| phase | max abs dSAP | max abs dMAP | max abs dHR | scored |
|---|---|---|---|---|
| supine | 0.00 | 0.00 | 0.00 | PASS |
| tilt up + upright | 1.11 | 0.83 | 0.45 | PASS |
| tilt down + supine | 38.89 | 30.68 | 16.32 | by design (see header) |

Verification: **PASS**

## Validation vs PRCP (Δ from the −60…−10 s baseline)

Transition durations taken from the PRCP Angle channel (median): rapid_tilt_up 2.0 s, slow_tilt_up 42.0 s, rapid_tilt_down 2.0 s, slow_tilt_down 41.0 s. Stand-up approximated as a 1 s tilt to 90° (not scored).

| event | signal | model early | PRCP early median [IQR] | ok | model steady | PRCP steady median [IQR] | ok | envelope coverage 0–180 s | ok |
|---|---|---|---|---|---|---|---|---|---|
| rapid_tilt_up | sbp | -12.2 | -15.1 [-18.1, -10.2] | (info) | -0.3 | -2.9 [-4.8, 1.4] | (info) | 99 % | (info) |
| rapid_tilt_up | map | -7.1 | -9.0 [-14.1, -5.5] | PASS | +6.8 | 4.4 [-0.2, 7.1] | PASS | 93 % | PASS |
| rapid_tilt_up | hr | +35.2 | 15.2 [12.7, 20.0] | FAIL | +16.7 | 12.1 [8.4, 16.4] | PASS | 93 % | PASS |
| slow_tilt_up | sbp | -9.6 | -7.9 [-13.2, -1.2] | (info) | -0.1 | 2.7 [-4.7, 4.0] | (info) | 98 % | (info) |
| slow_tilt_up | map | -5.2 | -6.0 [-8.4, -3.2] | PASS | +6.9 | 5.6 [0.9, 8.3] | PASS | 99 % | PASS |
| slow_tilt_up | hr | +15.6 | 9.7 [6.6, 10.6] | PASS | +16.4 | 14.0 [10.8, 18.2] | PASS | 85 % | PASS |
| rapid_tilt_down | sbp | -7.6 | -13.1 [-22.3, -6.8] | (info) | -1.2 | -0.4 [-6.9, 6.9] | (info) | 100 % | (info) |
| rapid_tilt_down | map | -14.6 | -12.1 [-17.9, -7.0] | (info) | -7.4 | -7.3 [-9.6, -0.3] | (info) | 99 % | (info) |
| rapid_tilt_down | hr | +0.5 | 7.4 [3.4, 11.9] | (info) | -16.1 | -15.6 [-21.4, -9.0] | (info) | 97 % | (info) |
| slow_tilt_down | sbp | -0.4 | -5.0 [-12.1, -3.5] | (info) | -1.5 | 1.5 [-7.4, 5.9] | (info) | 100 % | (info) |
| slow_tilt_down | map | -2.8 | -6.4 [-12.1, -4.9] | (info) | -7.5 | -4.6 [-10.2, 0.7] | (info) | 100 % | (info) |
| slow_tilt_down | hr | +0.5 | 2.3 [0.8, 5.8] | (info) | -16.0 | -17.5 [-19.9, -11.2] | (info) | 98 % | (info) |
| stand_up | sbp | -12.1 | -28.2 [-40.2, -23.2] | (info) | -0.1 | 3.1 [-1.0, 7.1] | (info) | 98 % | (info) |
| stand_up | map | -7.6 | -23.9 [-27.0, -19.6] | (info) | +7.1 | 6.0 [3.7, 10.0] | (info) | 93 % | (info) |
| stand_up | hr | +36.1 | 32.0 [28.3, 37.2] | (info) | +17.3 | 15.1 [11.0, 18.1] | (info) | 93 % | (info) |

Model baselines (absolute, aortic): rapid_tilt_up: MAP 95, HR 66; slow_tilt_up: MAP 95, HR 66; rapid_tilt_down: MAP 102, HR 82; slow_tilt_down: MAP 102, HR 82; stand_up: MAP 95, HR 66. PRCP baselines are finger (Finapres) pressures, so only changes are compared.


**Verification PASS; validation FAIL → G0 not yet (see step 1c)**

