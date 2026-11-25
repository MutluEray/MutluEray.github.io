# C10-B step 2 (v2) — SCI plant, EES actuator, virtual population

Healthy base: step 1c v2 calibrated CVSim. SCI = complete cervical (all sympathetic efferent reflex arms cut, vagal HR arm kept).

## A. SCI calibration to Corbett et al. 1971 (complete cervical SCI)

BoTorch BO; 18 evaluations; stopped: stall (12 evaluations without >1 % improvement); 37 s. Best loss 1.397.

| parameter | value | bounds |
|---|---|---|
| base_res_x | 0.9086 | [0.4, 1.0] |
| zpfv_add_ml | 139.9 | [0.0, 1000.0] |
| rr_para | 0.01189 | [0.003, 0.015] |
| abr_setpoint | 69.71 | [60.0, 95.0] |
| hr_nominal | 63.85 | [50.0, 80.0] |

Parameters within 2 % of a bound: none.

| angle | role | supine MAP model / target | supine HR model / target | ΔMAP model / target | ΔHR model / target | SBP drop in 3 min | within tolerance |
|---|---|---|---|---|---|---|---|
| 30° | calibrated | 71 / 71 | 63 / 64 | -28 / -24 | +24 / +18 | 37 mmHg | yes (x1.0 tol); peak HR 88 |
| 45° | calibrated | 71 / 71 | 63 / 64 | -39 / -38 | +29 / +35 | 51 mmHg | yes (x1.0 tol); peak HR 93 |
| 60° | held out (not gating) | 71 / 71 | 63 / 64 | -47 / -26 | +31 / +41 | 59 mmHg | NO (x1.5 tol); peak HR 95 |

OH definition at 60° (SBP drop ≥ 20 mmHg within 3 min): met.

**G-SCI: PASS**

## B. EES actuator (nominal SCI patient)

Gain for +30 mmHg supine SBP at u = 1 (delay 5 s, tau 15 s, frac_res 0.5): **1.547**.

Static amplitude → ΔSBP: linear fit R² = 1.000 for u ≤ 1, 1.000 up to u = 1.5. Linear largely by construction (additive actuator on a near-linear plant in this range); consistent with, not validated by, the linear relation in Phillips 2025.

| u | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1 | 1.2 | 1.5 |
|---|---|---|---|---|---|---|---|---|
| ΔSBP (mmHg) | -0.0 | +5.8 | +11.8 | +17.7 | +23.7 | +29.9 | +36.1 | +45.7 |

60° tilt, plateau ΔSBP: -60.0 mmHg without EES, -27.2 with u = 1, +6.1 with u = 2 (constant from tilt onset). Controllers in step 4 get u ∈ [0, 2].

## C. Virtual population

60 patients (3 redrawn by the plausibility filter: supine MAP outside 50–100 3, MAP rises on tilt 0, peak HR > 150 0; a draw can fail several); 0–39 tuning, 40–59 test.

| quantity | median | 10th pct | 90th pct |
|---|---|---|---|
| supine_sbp | 88.1 | 66.1 | 108.8 |
| supine_map | 70.6 | 54.1 | 85.0 |
| supine_hr | 63.9 | 56.8 | 77.2 |
| hr_max_tilt60 | 93.2 | 80.8 | 109.9 |
| tilt60_dmap | -48.2 | -56.9 | -40.4 |
| tilt60_sbp_drop_3min | 60.3 | 55.0 | 64.5 |
| ees_u1_dsbp | 28.3 | 19.7 | 41.1 |
| delay_s | 7.3 | 2.7 | 11.1 |
| tau_s | 17.3 | 8.4 | 27.2 |

Share meeting the OH definition at 60°: 100 %.


Total wall time 180 s.
