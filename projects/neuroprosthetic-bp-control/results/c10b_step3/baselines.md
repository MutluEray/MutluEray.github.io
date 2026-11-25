# C10-B step 3 (v2) — harness check and baseline controllers (hand-set gains, 40 tuning patients)

Episode: 30 min simulated, 2.9 s wall per episode (160 episodes).

Median [IQR] across patients. Gains are hand-set, not tuned — no controller comparison is claimed from this table.

| controller | in_range | hyper | hypo | peak_tilt | recovery | ad_peak | ad_hyper | n_changes | tv | u_mean | sat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 1.0 [0.0, 4.6] | 0.0 [0.0, 0.0] | 73.6 [60.9, 95.1] | 96.1 [80.3, 102.0] | 600.0 [600.0, 600.0] | 113.8 [97.8, 122.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| C0 fixed | 17.5 [16.0, 19.2] | 0.0 [0.0, 0.0] | 53.9 [53.8, 54.0] | 115.0 [114.5, 115.3] | 600.0 [600.0, 600.0] | 133.9 [133.0, 135.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.8 [0.4, 1.2] | 0.0 [0.0, 0.0] |
| C1 rule | 18.8 [9.5, 31.5] | 6.3 [4.6, 8.1] | 8.5 [6.9, 12.4] | 113.2 [110.3, 115.2] | 231.5 [160.8, 600.0] | 162.9 [160.7, 165.3] | 30.8 [27.7, 38.4] | 62.0 [55.8, 65.0] | 6.2 [5.6, 6.5] | 1.7 [1.3, 2.0] | 0.0 [0.0, 20.5] |
| C2 PID | 59.7 [34.2, 72.9] | 0.9 [0.3, 1.1] | 1.9 [1.5, 2.7] | 114.8 [114.7, 115.0] | 65.5 [54.8, 99.2] | 146.7 [143.2, 149.6] | 0.0 [0.0, 0.0] | 1053.0 [877.0, 1174.0] | 72.8 [60.7, 79.7] | 1.7 [1.3, 2.1] | 2.1 [0.0, 29.1] |

### Per phase (median [IQR])

| controller | in_range_supine | in_range_tilt60 | in_range_return | in_range_tilt30 | hypo_supine | hypo_tilt60 | hypo_return | hypo_tilt30 | hyper_return |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 100] | 100 [100, 100] | 92 [31, 100] | 100 [100, 100] | 0 [0, 0] |
| C0 fixed | 100 [100, 100] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 100 [100, 100] | 2 [2, 2] | 100 [100, 100] | 0 [0, 0] |
| C1 rule | 46 [2, 83] | 9 [0, 36] | 1 [0, 1] | 0 [0, 16] | 0 [0, 0] | 20 [15, 33] | 0 [0, 0] | 0 [0, 8] | 17 [0, 24] |
| C2 PID | 100 [100, 100] | 75 [0, 86] | 62 [48, 78] | 69 [47, 78] | 0 [0, 0] | 2 [1, 5] | 0 [0, 0] | 0 [0, 0] | 5 [0, 6] |

Metric definitions: see c10b_harness.py header. in_range/hyper/hypo/ad_hyper/sat in % time; peak_tilt, ad_peak in mmHg; recovery in s (600 = never within 10 min). Amplitude range u in [0, 3].

