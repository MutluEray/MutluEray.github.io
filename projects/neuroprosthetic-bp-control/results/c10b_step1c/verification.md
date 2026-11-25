# C10-B — CVSim plant: verification and validation (G0)

Build + load: 0.0 s. Parameters: calibrated to PRCP median (step 1c v2). Both reflexes on.

## Baseline (supine)

Mean over 250–300 s: SAP 116.9, MAP 94.7, DAP 71.6 mmHg, HR 67.3 bpm. Drift vs 200–250 s: MAP +0.00 mmHg, HR -0.00 bpm.

## Timing

300 s simulated: 0.37 s wall. 30 min simulated: 2.2 s wall.

## Verification: external-angle tilt vs CVSim's original tilt

| phase | max abs dSAP | max abs dMAP | max abs dHR | scored |
|---|---|---|---|---|
| supine | 0.00 | 0.00 | 0.00 | PASS |
| tilt up + upright | 1.10 | 0.81 | 0.23 | PASS |
| tilt down + supine | 46.42 | 38.87 | 8.96 | by design (see header) |

Verification: **PASS**

## Validation vs PRCP (Δ from the −60…−10 s baseline)

Transition durations taken from the PRCP Angle channel (median): rapid_tilt_up 2.0 s, slow_tilt_up 42.0 s, rapid_tilt_down 2.0 s, slow_tilt_down 41.0 s. Stand-up approximated as a 1 s tilt to 90° (not scored).

| event | signal | model early | PRCP early median [IQR] | ok | model steady | PRCP steady median [IQR] | ok | envelope coverage 0–180 s | ok |
|---|---|---|---|---|---|---|---|---|---|
| rapid_tilt_up | sbp | -12.9 | -15.1 [-18.1, -10.2] | (info) | -5.4 | -2.9 [-4.8, 1.4] | (info) | 99 % | (info) |
| rapid_tilt_up | map | -10.5 | -9.0 [-14.1, -5.5] | PASS | +2.8 | 4.4 [-0.2, 7.1] | PASS | 100 % | PASS |
| rapid_tilt_up | hr | +16.0 | 15.2 [12.7, 20.0] | PASS | +9.5 | 12.1 [8.4, 16.4] | PASS | 100 % | PASS |
| slow_tilt_up | sbp | -9.1 | -7.9 [-13.2, -1.2] | (info) | -5.1 | 2.7 [-4.7, 4.0] | (info) | 98 % | (info) |
| slow_tilt_up | map | -5.8 | -6.0 [-8.4, -3.2] | PASS | +3.0 | 5.6 [0.9, 8.3] | PASS | 100 % | PASS |
| slow_tilt_up | hr | +8.0 | 9.7 [6.6, 10.6] | PASS | +9.3 | 14.0 [10.8, 18.2] | PASS | 96 % | PASS |
| rapid_tilt_down | sbp | -6.2 | -13.1 [-22.3, -6.8] | (info) | +4.0 | -0.4 [-6.9, 6.9] | (info) | 98 % | (info) |
| rapid_tilt_down | map | -16.0 | -12.1 [-17.9, -7.0] | (info) | -3.3 | -7.3 [-9.6, -0.3] | (info) | 98 % | (info) |
| rapid_tilt_down | hr | +0.3 | 7.4 [3.4, 11.9] | (info) | -9.2 | -15.6 [-21.4, -9.0] | (info) | 100 % | (info) |
| slow_tilt_down | sbp | -0.6 | -5.0 [-12.1, -3.5] | (info) | +3.7 | 1.5 [-7.4, 5.9] | (info) | 99 % | (info) |
| slow_tilt_down | map | -0.4 | -6.4 [-12.1, -4.9] | (info) | -3.4 | -4.6 [-10.2, 0.7] | (info) | 100 % | (info) |
| slow_tilt_down | hr | +0.3 | 2.3 [0.8, 5.8] | (info) | -9.1 | -17.5 [-19.9, -11.2] | (info) | 100 % | (info) |
| stand_up | sbp | -12.2 | -28.2 [-40.2, -23.2] | (info) | -5.4 | 3.1 [-1.0, 7.1] | (info) | 90 % | (info) |
| stand_up | map | -9.6 | -23.9 [-27.0, -19.6] | (info) | +2.9 | 6.0 [3.7, 10.0] | (info) | 98 % | (info) |
| stand_up | hr | +16.2 | 32.0 [28.3, 37.2] | (info) | +9.9 | 15.1 [11.0, 18.1] | (info) | 93 % | (info) |

Model baselines (absolute, aortic): rapid_tilt_up: MAP 95, HR 67; slow_tilt_up: MAP 95, HR 67; rapid_tilt_down: MAP 97, HR 77; slow_tilt_down: MAP 97, HR 77; stand_up: MAP 95, HR 67. PRCP baselines are finger (Finapres) pressures, so only changes are compared.


**Verification PASS; validation PASS → G0 PASS**

