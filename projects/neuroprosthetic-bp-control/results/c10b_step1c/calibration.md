# C10-B step 1c (v2) — BO calibration of the nominal subject to PRCP

Method: BoTorch GP (Matern 5/2 ARD) + LogEI; 60 evaluations; stopped: **stall (15 evaluations without >1 % improvement)**; wall 105 s.

Default model loss 1.208 (G0 criteria not met); best model loss 0.273 (G0 criteria met; best at evaluation 57).

| parameter | indices | default | calibrated | bounds |
|---|---|---|---|---|
| rr_symp_gain | [2] | 0.012 | 0.008714 | [0.0, 0.02] |
| rr_para_gain | [3] | 0.009 | 0.003303 | [0.003, 0.018] |
| carotid_offset | [14] | -25 | -19.1 | [-30.0, -5.0] |
| abr_scale | [1] | 18 | 21.09 | [10.0, 30.0] |
| abr_res_gain_x | [4, 5, 6, 7] | ×1 | ×1.500 | [0.3, 1.5] |
| abr_vt_gain_x | [8, 9, 10, 11] | ×1 | ×0.300 | [0.3, 1.5] |

Parameters within 2 % of a bound: abr_res_gain_x, abr_vt_gain_x.

## Shape metrics (descriptive, not pass/fail)

| metric | PRCP median curve | CVSim default | calibrated |
|---|---|---|---|
| rapid_map_overshoot | 5.4 | 15.9 | 9.1 |
| slow_hr_t63_s | 56.0 | 13.0 | 14.0 |

Calibrated on rapid and slow tilt up only; tilt down and stand-up in the re-run below are held out.

