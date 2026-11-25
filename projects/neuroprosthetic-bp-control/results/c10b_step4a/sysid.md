# C10-B step 4a — system identification from calibration sessions

60 patients × [5, 10, 20, 60] min; ARX(1) + input delay + posture input; 642 s.

Median [IQR] across patients. Delay error = identified d − true actuator delay (the ARX delay also absorbs part of the actuator lag, so a positive bias is expected).

| T_cal (min) | identified delay (s) | delay error (s) | gain (mmHg per u) | tau (s) | free-run RMSE in session | free-run RMSE validation |
|---|---|---|---|---|---|---|
| 5 | 14.5 [11.8, 17.2] | 7.3 [5.7, 8.9] | 24.2 [17.0, 30.7] | 8.3 [6.5, 10.5] | 3.6 [3.2, 4.5] | 5.1 [4.4, 5.8] |
| 10 | 13.5 [10.8, 15.0] | 5.8 [4.4, 7.2] | 21.7 [17.3, 31.4] | 9.7 [7.9, 14.0] | 3.9 [3.4, 4.4] | 4.9 [4.5, 5.7] |
| 20 | 13.0 [10.0, 15.0] | 5.1 [4.2, 6.6] | 22.7 [17.6, 34.2] | 11.2 [8.0, 15.1] | 4.2 [3.8, 4.5] | 5.6 [5.1, 6.6] |
| 60 | 12.0 [9.0, 15.0] | 4.2 [3.7, 6.3] | 23.5 [18.2, 35.3] | 12.2 [8.8, 16.9] | 4.2 [3.9, 4.6] | 7.6 [7.2, 8.2] |
