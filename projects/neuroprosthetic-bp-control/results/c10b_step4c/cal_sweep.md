# C10-B — calibration-length sweep (test patients)

Mean J of tuned PID (C2f): 81.9. Advantage = mean J(PID) − mean J(model, T_cal).

| family | T_cal (min) | mean J | in_range | hyper | advantage over PID | share of 60-min advantage |
|---|---|---|---|---|---|---|
| C4ff | 5 | 76.7 | 56.1 | 0.06 | +5.1 | 89 % |
| C4ff | 10 | 75.5 | 57.5 | 0.05 | +6.4 | 111 % |
| C4ff | 20 | 75.6 | 57.7 | 0.07 | +6.2 | 109 % |
| C4ff | 60 | 76.1 | 57.3 | 0.05 | +5.7 | 100 % |
| C4 | 5 | 78.0 | 57.5 | 0.50 | +3.9 | 63 % |
| C4 | 10 | 76.2 | 58.7 | 0.45 | +5.7 | 92 % |
| C4 | 20 | 76.1 | 58.8 | 0.44 | +5.8 | 94 % |
| C4 | 60 | 75.7 | 59.2 | 0.42 | +6.2 | 100 % |

**G1 calibration part (20-min model keeps ≥ 70 % of the 60-min advantage, C4 or C4ff): PASS**
