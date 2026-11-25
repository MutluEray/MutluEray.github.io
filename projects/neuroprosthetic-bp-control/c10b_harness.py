"""
C10-B — evaluation harness: clinic-like scenario, disturbances, metrics, baseline controllers.
Used by step 3 (baselines) and step 4 (tuning and comparison).

Scenario (30 min, identical for every patient and controller):
     0– 300 s  supine
   300– 900 s  rapid head-up tilt to 60° (10 s ramp), held 10 min
   900–1200 s  back to supine (10 s ramp)
  1200–1500 s  slow tilt to 30° over 40 s (sitting-up-like), held
  1500–1800 s  back to supine (40 s ramp); AD-like hypertensive surge 1560–1680 s
  Sensor: SBP noise SD 2 mmHg; Finapres-style dropouts (value held) at 600–630 s and 1350–1365 s.
  The AD-like surge is an uncontrolled sympathetic vasoconstriction below the lesion
  (same pathways as EES, 80 % resistance), shaped to add ~+40 mmHg in the nominal patient;
  the controller only sees its effect on BP. No claim that a controller treats AD.

Control target: SBP 110–120 mmHg (Harkema 2018; Christie et al. 2026), set point 115.
Device constraints applied to every controller: u in [0, U_MAX], resolution 0.05.
U_MAX = 3 (v2; was 2): with u <= 2 every closed-loop controller sat at the ceiling for
the whole 60° phase in step 3 v1 and SBP still fell to the 70s, so a third of the
scenario measured actuator saturation rather than control. u is normalised (u = 1 ≈
+30 mmHg supine in the nominal patient); the clinical amplitude range is programmable,
so a wider range is a device setting, not a physiological claim. Time at the ceiling
is reported as `sat`.

Metrics (scored 120–1800 s, true SBP):
  in_range   % time SBP in 110–120
  hyper      % time SBP > 150          (hypertension / AD-risk proxy)
  hypo       % time SBP < 90
  peak_tilt  max SBP 300–420 s (overshoot after the rapid tilt)
  recovery   s from tilt onset until SBP stays in 100–130 for 30 s (cap 600)
  ad_peak    max SBP 1560–1760 s; ad_hyper % time > 150 in that window
  n_changes  number of amplitude changes; tv = total variation of u; u_mean
  sat        % time at U_MAX (actuator authority exhausted)
  per phase  in_range and hypo for supine (120–300), tilt60 (330–900), return (910–1200),
             tilt30 (1240–1500); hyper for the return phase (Christie et al. saw >150 mmHg
             around posture changes)
"""

import json
import os

import numpy as np
import pandas as pd

from c10b_plant import EESActuator, R_UNIT, V_UNIT, W_R, W_V, sci_theta, simulate

T_END = 1800.0
SET, LO, HI = 115.0, 110.0, 120.0
U_MIN, U_MAX, U_RES = 0.0, 3.0, 0.05
SCORE_FROM = 120.0
ANGLE_BP = [(0, 0), (300, 0), (310, 60), (900, 60), (910, 0), (1200, 0), (1240, 30), (1500, 30), (1540, 0), (1800, 0)]
DROPOUTS = [(600, 630), (1350, 1365)]
AD = (1560.0, 20.0, 60.0, 40.0)     # start, rise, hold, decay (s)
AD_DRIVE = 1.3                      # in nominal-gain drive units
NOISE_SD = 2.0
PHASES = [("supine", 120, 300), ("tilt60", 330, 900), ("return", 910, 1200), ("tilt30", 1240, 1500)]


def angle_fn(t):
    ts, a = zip(*ANGLE_BP)
    return float(np.interp(t, ts, a))


def dropout_fn(t):
    return any(a <= t < b for a, b in DROPOUTS)


def make_dist_fn(g_nom):
    t0, rise, hold, decay = AD

    def shape(t):
        if t < t0 or t > t0 + rise + hold + decay:
            return 0.0
        if t < t0 + rise:
            return 0.5 * (1 - np.cos(np.pi * (t - t0) / rise))
        if t < t0 + rise + hold:
            return 1.0
        return 0.5 * (1 + np.cos(np.pi * (t - t0 - rise - hold) / decay))

    def f(t):
        s = AD_DRIVE * shape(t)
        return g_nom * 0.8 * R_UNIT * s * W_R, g_nom * 0.2 * V_UNIT * s * W_V
    return f


def quantise(u):
    return float(np.clip(np.round(u / U_RES) * U_RES, U_MIN, U_MAX))


# ------------------------------------------------------------------ population
def load_population(path=os.path.join("data", "c10b_population.json")):
    pop = json.load(open(path))
    base = np.load(pop["base_theta"])
    csv = os.path.join("results", "c10b_step2", "population.csv")
    info = pd.read_csv(csv).set_index("id") if os.path.exists(csv) else None
    return pop, base, info


def patient_plant(pop, base, pid):
    p = pop["patients"][pid]
    return sci_theta(base, **p["plant"]), EESActuator(**p["actuator"])


# ------------------------------------------------------------------ metrics
def metrics(o):
    t, s, u = o["t"], o["sbp"], o["u"]
    m = t >= SCORE_FROM
    r = {"in_range": 100 * np.mean((s[m] >= LO) & (s[m] <= HI)),
         "hyper": 100 * np.mean(s[m] > 150), "hypo": 100 * np.mean(s[m] < 90),
         "peak_tilt": float(np.nanmax(s[(t >= 300) & (t <= 420)]))}
    rec = 600.0
    ok = (s >= 100) & (s <= 130)
    idx = np.where(t >= 300)[0]
    run = 0
    for k in idx:
        run = run + 1 if ok[k] else 0
        if run >= 30:
            rec = float(t[k] - 30 - 300)
            break
        if t[k] > 900:
            break
    r["recovery"] = rec
    w = (t >= 1560) & (t <= 1760)
    r["ad_peak"] = float(np.nanmax(s[w]))
    r["ad_hyper"] = 100 * np.mean(s[w] > 150)
    du = np.abs(np.diff(u))
    r["n_changes"] = int(np.sum(du > 1e-9))
    r["tv"] = float(np.sum(du))
    r["u_mean"] = float(np.mean(u))
    r["sat"] = 100 * np.mean(u[m] >= U_MAX - 1e-9)
    for name, a, b in PHASES:
        w = (t >= a) & (t < b)
        r[f"in_range_{name}"] = 100 * np.mean((s[w] >= LO) & (s[w] <= HI))
        r[f"hypo_{name}"] = 100 * np.mean(s[w] < 90)
    w = (t >= 910) & (t < 1200)
    r["hyper_return"] = 100 * np.mean(s[w] > 150)
    return r


# ------------------------------------------------------------------ controllers
class Controller:
    name = "base"

    def reset(self, info=None):
        pass

    def __call__(self, t, obs):
        raise NotImplementedError


class NoStim(Controller):
    name = "none"

    def __call__(self, t, obs):
        return 0.0


class FixedAmplitude(Controller):
    """C0: amplitude programmed once in clinic so that supine SBP sits at the set point
    (uses the patient's supine SBP and u = 1 response from the step 2 table)."""
    name = "C0 fixed"

    def reset(self, info=None):
        if info is None:
            self.u0 = 1.0
        else:
            self.u0 = float(np.clip((SET - info["supine_sbp"]) / max(info["ees_u1_dsbp"], 1.0), U_MIN, U_MAX))

    def __call__(self, t, obs):
        return self.u0


class RuleBased(Controller):
    """C1: Christie et al. 2026 v3-style rule: outside 110–120, wait T(|err|) then step
    the amplitude by `step` in the corrective direction; T shrinks sigmoidally with the
    deviation (Tmin for large, Tmax for small errors). Holds during sensor dropouts."""
    name = "C1 rule"

    def __init__(self, t_min=10.0, t_max=70.0, e_mid=15.0, k=0.3, step=0.1):
        self.t_min, self.t_max, self.e_mid, self.k, self.step = t_min, t_max, e_mid, k, step

    def reset(self, info=None):
        self.u, self.t_last = 0.0, -1e9

    def __call__(self, t, obs):
        y = obs["sbp"]
        if obs["valid"] and not (LO <= y <= HI):
            err = abs(y - SET)
            wait = self.t_min + (self.t_max - self.t_min) / (1 + np.exp(self.k * (err - self.e_mid)))
            if t - self.t_last >= wait:
                self.u = float(np.clip(self.u + (self.step if y < LO else -self.step), U_MIN, U_MAX))
                self.t_last = t
        return self.u


class PID(Controller):
    """C2: PID on SBP error to 115 (Phillips 2025 regulator class), derivative on the
    filtered measurement, conditional-integration anti-windup, integrator frozen during
    sensor dropouts, optional output rate limit."""
    name = "C2 PID"

    def __init__(self, kp=0.02, ki=0.0007, kd=0.0, d_tau=5.0, rate=0.2):
        self.kp, self.ki, self.kd, self.d_tau, self.rate = kp, ki, kd, d_tau, rate

    def reset(self, info=None):
        self.i, self.y_f, self.u = 0.0, None, 0.0

    def __call__(self, t, obs):
        y = obs["sbp"]
        if self.y_f is None:
            self.y_f = y
        if obs["valid"]:
            a = 1.0 / (1.0 + self.d_tau)
            y_prev = self.y_f
            self.y_f += a * (y - self.y_f)
            dy = self.y_f - y_prev
            e = SET - y
            u_unsat = self.kp * e + self.ki * (self.i + e) - self.kd * dy
            if U_MIN < u_unsat < U_MAX or (u_unsat >= U_MAX and e < 0) or (u_unsat <= U_MIN and e > 0):
                self.i += e
            u = self.kp * e + self.ki * self.i - self.kd * dy
            u = float(np.clip(u, U_MIN, U_MAX))
            self.u = float(np.clip(u, self.u - self.rate, self.u + self.rate))
        return self.u


# ------------------------------------------------------------------ runner
def run_patient(sim, pop, base, pid, controller, info=None, seed=0):
    th, act = patient_plant(pop, base, pid)
    controller.reset(info)
    g_nom = pop["ees_gain_nominal"]
    o = simulate(sim, th, T_END, angle_fn, lambda t, obs: quantise(controller(t, obs)), actuator=act,
                 sbp_noise_sd=NOISE_SD, rng=np.random.default_rng(1000 + pid + seed),
                 dist_fn=make_dist_fn(g_nom), dropout_fn=dropout_fn)
    return o, metrics(o)
