"""
C10-B — controller families, their BO parameter spaces, and the tuning cost.

Families (all see the same measurements and device limits; see c10b_harness.py):
  C1   rule-based timed steps (Christie et al. 2026 v3 style)          5 parameters
  C2   PID (Phillips et al. 2025 regulator class)                      5 parameters
  C3   LQI on the patient's identified ARX model (week 2), Δu form     3 parameters
  C3ff C3 + static posture feedforward from the measured tilt angle    3 parameters
  C4   offset-free linear MPC on the identified model (week 6 recipe:
       shooting over move-blocked inputs, L-BFGS-B with bounds),
       soft SBP limits, Δu penalty                                     4 parameters
  C4ff C4 with the measured tilt angle in the prediction (held over
       the horizon: the controller does not know future posture)      4 parameters
  C0 (fixed amplitude) has no tunable parameter.

Tuning cost per episode (pre-registered 2026-11-25, before any tuning run):
  J = (100 − in_range) + 5·hyper + 2·hypo + 0.5·tv
  in % time (scored 120–1800 s, true SBP) and total variation of u. Hypertension is
  weighted most (autonomic-dysreflexia risk), hypotension next, and tv penalises
  amplitude chatter (clinically: many changes are uncomfortable and were counted as a
  cost in Christie et al.). Population cost = mean J over the tuning subset.
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

import c10b_harness as H


def cost_J(m):
    return (100.0 - m["in_range"]) + 5.0 * m["hyper"] + 2.0 * m["hypo"] + 0.5 * m["tv"]


def _prev_u(obs):
    return obs["hist"][-1][2] if obs["hist"] else 0.0


def _u_hist(obs, n):
    """Applied amplitudes u[k-1], u[k-2], ..., u[k-n] (0 before the episode start)."""
    h = obs["hist"]
    return np.array([h[-j][2] if len(h) >= j else 0.0 for j in range(1, n + 1)])


# ------------------------------------------------------------------ C3 LQI
class LQI(H.Controller):
    """Δu-form LQI on y[k+1] = a y + b u[k-d] (+ e sin(angle) + c, rejected by the integrator
    or, in the ff variant, compensated statically). State z = [y − r, u[k-1..k-D], I].
    Gains from the discrete Riccati equation with Q = diag(1, 0…, q_i), R = r."""

    def __init__(self, log_r=3.0, log_qi=-2.0, tau_f=2.0, ff=False):
        self.log_r, self.log_qi, self.tau_f, self.ff = log_r, log_qi, tau_f, ff
        self.name = "C3ff LQI+ff" if ff else "C3 LQI"

    def reset(self, info=None):
        m = info["model"]
        self.m = m
        a, b, d = m["a"], m["b"], m["d"]
        D = max(d, 1)
        n = 1 + D + 1
        A = np.zeros((n, n)); B = np.zeros((n, 1))
        A[0, 0] = a
        if d == 0:
            A[0, 1] = b; B[0, 0] = b
        else:
            A[0, d] = b                      # u[k-d] is state index d
        A[1, 1] = 1.0; B[1, 0] = 1.0         # u[k] = u[k-1] + Δu becomes the new u[k-1]
        for j in range(2, D + 1):
            A[j, j - 1] = 1.0                # shift register
        A[n - 1, 0] = 1.0; A[n - 1, n - 1] = 1.0
        Q = np.zeros((n, n)); Q[0, 0] = 1.0; Q[n - 1, n - 1] = 10.0 ** self.log_qi
        R = np.array([[10.0 ** self.log_r]])
        P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A).ravel()
        self.D, self.I, self.yf, self.s_prev = D, 0.0, None, None

    def __call__(self, t, obs):
        u_prev = _prev_u(obs)
        y = obs["sbp"]
        s = np.sin(np.deg2rad(obs["angle"]))
        if self.yf is None:
            self.yf, self.s_prev = y, s
        if not obs["valid"]:
            self.s_prev = s
            return u_prev
        al = 1.0 / (1.0 + self.tau_f)
        self.yf += al * (y - self.yf)
        e = self.yf - H.SET
        z = np.concatenate([[e], _u_hist(obs, self.D), [self.I]])
        du = -float(self.K @ z)
        if self.ff and self.m["b"] > 1e-6:
            du += -(self.m["e"] / self.m["b"]) * (s - self.s_prev)
        u = u_prev + du
        if (H.U_MIN < u < H.U_MAX) or (u >= H.U_MAX and e > 0) or (u <= H.U_MIN and e < 0):
            self.I += e                       # conditional integration (anti-windup)
        self.s_prev = s
        return u


# ------------------------------------------------------------------ C4 MPC
class MPC(H.Controller):
    """Offset-free linear MPC on the identified ARX model.
    Horizon N = 60 s, 10 move blocks of 6 s, inputs bounded to [0, U_MAX].
    Cost: mean (y − 115)² + W_SOFT·[relu(y − y_hi)² + relu(95 − y)²] + r_du·Σ(Δv)².
    Offset-free: a filtered one-step prediction error d̂ is added to every prediction.
    ff variant: e·sin(measured angle) enters the prediction, held constant over the horizon."""
    N, NB, W_SOFT, Y_LO = 60, 10, 10.0, 95.0

    def __init__(self, log_rdu=2.0, y_hi=140.0, tau_f=2.0, alpha_d=0.2, ff=False):
        self.log_rdu, self.y_hi, self.tau_f, self.alpha_d, self.ff = log_rdu, y_hi, tau_f, alpha_d, ff
        self.name = "C4ff MPC+ff" if ff else "C4 MPC"

    def reset(self, info=None):
        m = info["model"]
        self.m = m
        a, b, d, N = m["a"], m["b"], m["d"], self.N
        # y_j (j = 1..N) forced by future u[k+i] (i = 0..N-1): b a^(j-1-i-d) for i <= j-1-d
        M = np.zeros((N, N))
        for j in range(1, N + 1):
            for i in range(0, j - d):
                M[j - 1, i] = b * a ** (j - 1 - i - d)
        blk = np.zeros((N, self.NB))
        L = N // self.NB
        for q in range(self.NB):
            blk[q * L:(q + 1) * L, q] = 1.0
        self.G = M @ blk
        self.apow = a ** np.arange(1, N + 1)
        self.geo = np.array([(1 - a ** j) / (1 - a) if a < 1 else j for j in range(1, N + 1)])
        self.r_du = 10.0 ** self.log_rdu
        self.yf, self.dhat, self.v, self.y_pred_prev = None, 0.0, None, None

    def _free(self, y0, u_past, const):
        """Free response: y0 propagated, past inputs u[k-1..k-d] still in the pipeline, constants."""
        a, b, d, N = self.m["a"], self.m["b"], self.m["d"], self.N
        y = self.apow * y0 + self.geo * const
        # past input u[k-p] (p = 1..d) reaches y at step j when j-1 = d-p+... : y_j += b a^(j-1-(d-p)) for j-1 >= d-p
        for p in range(1, d + 1):
            lag = d - p
            j = np.arange(1, N + 1)
            mask = (j - 1) >= lag
            y[mask] += b * a ** (j[mask] - 1 - lag) * u_past[p - 1]
        return y

    def __call__(self, t, obs):
        u_prev = _prev_u(obs)
        y = obs["sbp"]
        s = np.sin(np.deg2rad(obs["angle"])) if self.ff else 0.0
        if self.yf is None:
            self.yf = y
        if not obs["valid"]:
            return u_prev
        al = 1.0 / (1.0 + self.tau_f)
        self.yf += al * (y - self.yf)
        m = self.m
        # offset-free correction from the one-step prediction error
        if self.y_pred_prev is not None:
            self.dhat += self.alpha_d * (self.yf - self.y_pred_prev)   # prediction already contains d-hat
        const = m["c"] + m["e"] * s + self.dhat
        u_past = _u_hist(obs, max(m["d"], 1))
        yfree = self._free(self.yf, u_past, const)
        G, r_du = self.G, self.r_du
        v0 = np.full(self.NB, u_prev) if self.v is None else np.r_[self.v[1:], self.v[-1]]

        def f(v):
            yp = yfree + G @ v
            e = yp - H.SET
            hi = np.maximum(yp - self.y_hi, 0.0)
            lo = np.maximum(self.Y_LO - yp, 0.0)
            dv = np.diff(np.r_[u_prev, v])
            J = np.mean(e ** 2) + self.W_SOFT * (np.sum(hi ** 2) + np.sum(lo ** 2)) / self.N + r_du * np.sum(dv ** 2)
            gy = 2 * e / self.N + self.W_SOFT * (2 * hi - 2 * lo) / self.N
            gdv = 2 * r_du * dv
            gv = G.T @ gy + gdv - np.r_[gdv[1:], 0.0]
            return J, gv

        res = minimize(f, v0, jac=True, method="L-BFGS-B", bounds=[(H.U_MIN, H.U_MAX)] * self.NB,
                       options={"maxiter": 50})
        self.v = res.x
        u = float(res.x[0])
        # model one-step prediction for the next d-hat update (with the input that will be applied)
        ud = u_past[m["d"] - 1] if m["d"] >= 1 else H.quantise(u)
        self.y_pred_prev = m["a"] * self.yf + m["b"] * ud + const
        return u


# ------------------------------------------------------------------ families and parameter spaces
# each entry: (name, lo, hi, log10?) ; BO works on [0, 1]^p
class PIDF(H.PID):
    """C2f: the clinical PID (C2) plus the same first-order measurement prefilter (time
    constant tau_f, s) that LQI and MPC get, applied to SBP before P, I and D. Added
    2026-11-25 so the PID baseline is not handicapped by acting on raw noisy SBP."""
    name = "C2f PID + prefilter"

    def __init__(self, tau_f=2.0, **kw):
        super().__init__(**kw)
        self.tau_f = tau_f

    def reset(self, info=None):
        super().reset(info)
        self.yp = None

    def __call__(self, t, obs):
        if obs["valid"]:
            y = obs["sbp"]
            self.yp = y if self.yp is None else self.yp + (y - self.yp) / (1.0 + self.tau_f)
        o = dict(obs)
        if self.yp is not None:
            o["sbp"] = self.yp
        return super().__call__(t, o)


class PIDFF(PIDF):
    """C2ff: C2f plus the same static posture feedforward as C3ff/C4ff,
    u_ff = -(e/b)·sin(measured angle) from the patient's identified ARX model, added outside
    the PID (anti-windup sees the total). Step 5a control: is the G1 effect the feedforward
    signal or the model-based controller?"""
    name = "C2ff PID + prefilter + ff"

    def reset(self, info=None):
        super().reset(info)
        m = (info or {}).get("model")
        self.kff = -(m["e"] / m["b"]) if (m and m["b"] > 1e-6) else 0.0

    def __call__(self, t, obs):
        if obs["valid"]:
            y = obs["sbp"]
            self.yp = y if self.yp is None else self.yp + (y - self.yp) / (1.0 + self.tau_f)
            if self.y_f is None:
                self.y_f = self.yp
            a = 1.0 / (1.0 + self.d_tau)
            y_prev = self.y_f
            self.y_f += a * (self.yp - self.y_f)
            dy = self.y_f - y_prev
            e = H.SET - self.yp
            base = self.kp * e - self.kd * dy + self.kff * np.sin(np.deg2rad(obs["angle"]))
            u_unsat = base + self.ki * (self.i + e)
            if H.U_MIN < u_unsat < H.U_MAX or (u_unsat >= H.U_MAX and e < 0) or (u_unsat <= H.U_MIN and e > 0):
                self.i += e
            u = float(np.clip(base + self.ki * self.i, H.U_MIN, H.U_MAX))
            self.u = float(np.clip(u, self.u - self.rate, self.u + self.rate))
        return self.u


FAMILIES = {
    "C1": {"cls": lambda p: H.RuleBased(t_min=p["t_min"], t_max=p["t_max"], e_mid=p["e_mid"], k=p["k"], step=p["step"]),
           "space": [("t_min", 3, 30, False), ("t_max", 30, 150, False), ("e_mid", 3, 30, False),
                     ("k", 0.05, 1.0, True), ("step", 0.05, 0.3, False)]},
    "C2": {"cls": lambda p: H.PID(kp=p["kp"], ki=p["ki"], kd=p["kd"], d_tau=p["d_tau"], rate=p["rate"]),
           "space": [("kp", 0.002, 0.1, True), ("ki", 1e-5, 0.01, True), ("kd", 1e-4, 0.5, True),
                     ("d_tau", 0.5, 20, True), ("rate", 0.01, 0.5, True)]},
    "C2f": {"cls": lambda p: PIDF(kp=p["kp"], ki=p["ki"], kd=p["kd"], d_tau=p["d_tau"], rate=p["rate"],
                                  tau_f=p["tau_f"]),
            "space": [("kp", 0.002, 0.1, True), ("ki", 1e-5, 0.01, True), ("kd", 1e-4, 0.5, True),
                      ("d_tau", 0.5, 20, True), ("rate", 0.01, 0.5, True), ("tau_f", 0, 10, False)]},
    "C2ff": {"cls": lambda p: PIDFF(kp=p["kp"], ki=p["ki"], kd=p["kd"], d_tau=p["d_tau"], rate=p["rate"],
                                    tau_f=p["tau_f"]),
             "space": [("kp", 0.002, 0.1, True), ("ki", 1e-5, 0.01, True), ("kd", 1e-4, 0.5, True),
                       ("d_tau", 0.5, 20, True), ("rate", 0.01, 0.5, True), ("tau_f", 0, 10, False)]},
    "C3": {"cls": lambda p: LQI(log_r=p["log_r"], log_qi=p["log_qi"], tau_f=p["tau_f"], ff=False),
           "space": [("log_r", 0, 6, False), ("log_qi", -5, 0, False), ("tau_f", 0, 10, False)]},
    "C3ff": {"cls": lambda p: LQI(log_r=p["log_r"], log_qi=p["log_qi"], tau_f=p["tau_f"], ff=True),
             "space": [("log_r", 0, 6, False), ("log_qi", -5, 0, False), ("tau_f", 0, 10, False)]},
    "C4": {"cls": lambda p: MPC(log_rdu=p["log_rdu"], y_hi=p["y_hi"], tau_f=p["tau_f"], alpha_d=p["alpha_d"], ff=False),
           "space": [("log_rdu", -2, 4, False), ("y_hi", 122, 150, False), ("tau_f", 0, 10, False),
                     ("alpha_d", 0.02, 1.0, True)]},
    "C4ff": {"cls": lambda p: MPC(log_rdu=p["log_rdu"], y_hi=p["y_hi"], tau_f=p["tau_f"], alpha_d=p["alpha_d"], ff=True),
             "space": [("log_rdu", -2, 4, False), ("y_hi", 122, 150, False), ("tau_f", 0, 10, False),
                       ("alpha_d", 0.02, 1.0, True)]},
}
NEEDS_MODEL = {"C2ff", "C3", "C3ff", "C4", "C4ff"}
# hand-set defaults (step 3 values for C1/C2; reasonable starts for the others) — always evaluated first
DEFAULTS = {"C1": {"t_min": 10, "t_max": 70, "e_mid": 15, "k": 0.3, "step": 0.1},
            "C2": {"kp": 0.02, "ki": 0.0007, "kd": 1e-4, "d_tau": 5.0, "rate": 0.2},
            "C2f": {"kp": 0.02, "ki": 0.0007, "kd": 1e-4, "d_tau": 5.0, "rate": 0.2, "tau_f": 2.0},
            "C2ff": {"kp": 0.02, "ki": 0.0007, "kd": 1e-4, "d_tau": 5.0, "rate": 0.2, "tau_f": 2.0},
            "C3": {"log_r": 3.0, "log_qi": -2.0, "tau_f": 2.0},
            "C3ff": {"log_r": 3.0, "log_qi": -2.0, "tau_f": 2.0},
            "C4": {"log_rdu": 2.0, "y_hi": 140.0, "tau_f": 2.0, "alpha_d": 0.2},
            "C4ff": {"log_rdu": 2.0, "y_hi": 140.0, "tau_f": 2.0, "alpha_d": 0.2}}


def to_params(fam, x01):
    p = {}
    for (n, lo, hi, lg), x in zip(FAMILIES[fam]["space"], np.clip(x01, 0, 1)):
        p[n] = float(10 ** (np.log10(lo) + x * (np.log10(hi) - np.log10(lo)))) if lg else float(lo + x * (hi - lo))
    return p


def to_x01(fam, p):
    x = []
    for n, lo, hi, lg in FAMILIES[fam]["space"]:
        v = p[n]
        x.append((np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) if lg else (v - lo) / (hi - lo))
    return np.clip(np.array(x), 0, 1)


def make(fam, p):
    return FAMILIES[fam]["cls"](p)
