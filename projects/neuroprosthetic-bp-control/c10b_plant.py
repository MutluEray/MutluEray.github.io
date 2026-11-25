"""
C10-B — plant layer on top of the CVSim wrapper: SCI parameterisation, EES actuator,
1 Hz episode runner. Reused by steps 2–5.

SCI (complete cervical, default): every sympathetic efferent arm of the arterial and
cardiopulmonary reflexes is cut (resistance, venous tone, RR-sympathetic, contractility);
the vagal (parasympathetic) heart-rate arm is kept. Loss of tonic sympathetic drive is
modelled as lower arteriolar resistance (x base_res_x) and larger splanchnic/leg
zero-pressure filling volume (+ zpfv_add_ml). The arterial baroreflex set point is
allowed to reset towards the chronically low pressure (abr_setpoint), as reported in
chronic SCI, and the heart rate at the set point (hr_nominal) drops without tonic
cardiac sympathetic drive. These are fitted to Corbett et al. 1971 (J Physiol; complete cervical SCI,
head-up tilt) in step 2.

EES actuator (assumption, swept in the virtual population):
  amplitude u (normalised; u = 1 gives the nominal patient ~+30 mmHg supine SBP)
  -> pure delay (delay_s) -> first-order lag (tau_s) -> drive s
  -> + resistance  (frac_res share)      in kidney / splanchnic / legs (weights W_R)
  -> - unstressed volume (1 - frac_res)  in splanchnic / legs (weights W_V)
  This mirrors the mechanism reported for lower-thoracic EES (sympathetic
  preganglionic recruitment -> vasoconstriction and splanchnic venous mobilisation);
  the numbers are not from data.
"""

import numpy as np

from c10b_cvsim import CVSim21

SYMP_IDX = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24]
RES_IDX = [53, 55, 57, 59]          # arteriolar resistances: upper body, kidney, splanchnic, legs
W_R = np.array([0.0, 0.2, 0.5, 0.3])  # EES resistance effect per compartment
W_V = np.array([0.0, 0.0, 0.7, 0.3])  # EES venous (unstressed-volume) effect per compartment
R_UNIT, V_UNIT = 2.0, 400.0         # mmHg·s/mL and mL per unit drive, before the gain


def sci_theta(base, base_res_x=0.7, zpfv_add_ml=400.0, rr_para=None, abr_setpoint=None, hr_nominal=None,
              blood_vol_x=1.0):
    th = np.array(base, dtype=float).copy()
    th[SYMP_IDX] = 0.0
    th[RES_IDX] = th[RES_IDX] * base_res_x
    th[79] += 0.6 * zpfv_add_ml       # splanchnic ZPFV
    th[80] += 0.4 * zpfv_add_ml       # leg ZPFV
    if rr_para is not None:
        th[3] = rr_para
    if abr_setpoint is not None:
        th[0] = abr_setpoint          # baroreflex resetting to chronically low pressure
    if hr_nominal is not None:
        th[90] = hr_nominal           # HR at the set point; lower without tonic cardiac sympathetic drive
    th[70] *= blood_vol_x
    return th


class EESActuator:
    def __init__(self, gain=1.0, delay_s=5.0, tau_s=15.0, frac_res=0.5, dt=1.0):
        self.gain, self.delay_s, self.tau_s, self.frac_res, self.dt = gain, delay_s, tau_s, frac_res, dt
        self.reset()

    def reset(self):
        n = max(0, int(round(self.delay_s / self.dt)))
        self.buf = [0.0] * n
        self.s = 0.0

    def step(self, u):
        """Apply amplitude u for one control period; returns (ees_dr, ees_dv) for that period."""
        self.buf.append(float(u))
        u_del = self.buf.pop(0)
        a = 1.0 - np.exp(-self.dt / max(self.tau_s, 1e-6))
        self.s += a * (u_del - self.s)
        dr = self.gain * self.frac_res * R_UNIT * self.s * W_R
        dv = self.gain * (1.0 - self.frac_res) * V_UNIT * self.s * W_V
        return dr, dv


def simulate(sim, theta, t_end, angle_fn, u_fn, actuator=None, settle=400.0, dt=1.0,
             sbp_noise_sd=0.0, rng=None, dist_fn=None, dropout_fn=None):
    """Run one episode at a fixed control period dt.
    angle_fn(t)   -> tilt angle (deg) at time t (s, t = 0 at the end of settling)
    u_fn(t, obs)  -> amplitude; obs = latest measured beat values, 'valid' flag, history
    dist_fn(t)    -> (dr, dv) arrays added to the actuator effect (e.g. an AD-like surge);
                     not seen by the controller
    dropout_fn(t) -> True while the BP sensor is unavailable (measured SBP is held,
                     obs['valid'] = False), e.g. Finapres recalibration
    Returns dict of 1 Hz arrays: t, sbp (true), sbp_meas, map, hr, u, s (actuator drive), angle."""
    rng = rng if rng is not None else np.random.default_rng(0)
    act = actuator if actuator is not None else EESActuator(gain=0.0)
    act.reset()
    sim.init(theta, mode=1)
    b = sim.run(settle, angle_end=angle_fn(0.0))
    last = b[-1] if len(b) else np.array([settle, np.nan, np.nan, np.nan, np.nan])
    n = int(round(t_end / dt))
    out = {k: np.full(n, np.nan) for k in ("t", "sbp", "map", "hr", "u", "s", "angle", "sbp_meas", "valid")}
    hist = []
    meas_prev = last[1]
    for i in range(n):
        t = i * dt
        valid = not (dropout_fn is not None and dropout_fn(t))
        if valid:
            meas = last[1] + (rng.normal(0, sbp_noise_sd) if sbp_noise_sd > 0 else 0.0)
            meas_prev = meas
        else:
            meas = meas_prev
        obs = {"t": t, "sbp": meas, "valid": valid, "map": last[2], "hr": last[4],
               "angle": angle_fn(t), "hist": hist}
        u = float(u_fn(t, obs))
        dr, dv = act.step(u)
        if dist_fn is not None:
            ddr, ddv = dist_fn(t)
            dr, dv = dr + ddr, dv + ddv
        b = sim.run(settle + t + dt, angle_end=angle_fn(t + dt), ees_dr=dr, ees_dv=dv)
        if len(b):
            last = b[-1]
        out["t"][i], out["sbp"][i], out["map"][i], out["hr"][i] = t + dt, last[1], last[2], last[4]
        out["u"][i], out["s"][i], out["angle"][i], out["sbp_meas"][i] = u, act.s, angle_fn(t + dt), meas
        out["valid"][i] = float(valid)
        hist.append((t, meas, u, valid))
    return out


def tilt_profile(t_tilt, angle, ramp_s=10.0, t_back=None):
    """Angle function: supine, linear ramp to `angle` at t_tilt, optional return at t_back."""
    def f(t):
        if t < t_tilt:
            return 0.0
        if t_back is not None and t >= t_back:
            return max(0.0, angle * (1 - (t - t_back) / ramp_s))
        return angle * min(1.0, (t - t_tilt) / ramp_s)
    return f


def new_sim():
    return CVSim21()
