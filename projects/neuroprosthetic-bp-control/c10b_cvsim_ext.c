/*
 * C10-B — thin extension driver around the CVSim 21-compartment C back end
 * (Heldt, Mukkamala, Moody, Mark; PhysioNet CVSim 1.0.0).
 *
 * This file adds, without touching the hemodynamic or reflex equations:
 *   1. a stepping API callable from Python (ctypes): init, run-until-t, beat numerics
 *   2. an "external tilt" mode: the tilt angle follows any commanded profile
 *      (e.g. the PRCP Angle channel) instead of CVSim's fixed 2-phase tilt.
 *      Gravity terms, intrathoracic-pressure shift, carotid offset and the
 *      interstitial fluid shift use CVSim's own coefficients; the fluid shift is
 *      written as the equivalent first-order lag (tau = 276 s, total loss
 *      vec[74]*sin(angle)/sin(85 deg), split 7/40/16 over sp/ll/ab as in CVSim).
 *   3. EES actuator hooks: ees_dr[k] adds to the arteriolar resistance and
 *      ees_dv[k] subtracts from the zero-pressure filling volume of compartment
 *      k (0 upper body, 1 kidney, 2 splanchnic, 3 legs). They are applied inside
 *      CVSim's own reflex update (see the patches in c10b_cvsim.py), so the
 *      pressure re-adjustment for venous-tone changes is CVSim's.
 *
 * Mode 0 runs CVSim's original tilt routine unchanged (for verification).
 */
#include <string.h>
#include <math.h>
#include "simulator.h"

int tilt_orig(Data_vector *, Parameter_vector *, Tilt_vector *, double, double);

/* actuator hooks read by the patched reflex.c */
double ees_dr[4] = {0.0, 0.0, 0.0, 0.0};
double ees_dv[4] = {0.0, 0.0, 0.0, 0.0};

/* ------------------------------------------------------------ state */
static Parameter_vector TH;
static Data_vector P;
static Reflex_vector R;
static Impulse_vector IMP;
static int ab_on = 1, cp_on = 1, tilt_on = 1, ext_mode = 1;
static double t_start = 0.0, t_stop = 0.0, htry = 1.0e-5;

/* commanded tilt profile: linear from (ang_t0, ang0) to (ang_t1, ang1) */
static double ang_t0 = 0.0, ang_t1 = 0.0, ang0 = 0.0, ang1 = 0.0;
static double vloss = 0.0;
#define TAU_IF 276.0
/* step 5 sensitivity: multiplier on the fluid-RETURN time constant (vloss falling),
   1.0 = CVSim's symmetric 276 s lag. Set with cv_set_refill(). */
static double refill_x = 1.0;
void cv_set_refill(double x) { refill_x = (x > 0.0) ? x : 1.0; }

/* beat accumulators (same definitions as CVSim's numerics_new) */
static double b_sys = 0.0, b_dia = 5000.0, b_int = 0.0, b_dt = 0.0;

static double angle_at(double t)
{
  if (t <= ang_t0 || ang_t1 <= ang_t0) return (t >= ang_t1) ? ang1 : ang0;
  if (t >= ang_t1) return ang1;
  return ang0 + (ang1 - ang0) * (t - ang_t0) / (ang_t1 - ang_t0);
}

static double angle_rate(double t)   /* deg/s */
{
  if (ang_t1 <= ang_t0 || t < ang_t0 || t > ang_t1) return 0.0;
  return (ang1 - ang0) / (ang_t1 - ang_t0);
}

static double vloss_target(double angdeg)
{
  return TH.vec[74] * sin(angdeg * M_PI / 180.0) / sin(85.0 * M_PI / 180.0);
}

/* ------------------------------------------------------------ external tilt */
static int tilt_ext(Data_vector *p, Parameter_vector *theta, Tilt_vector *tv)
{
  double a_deg = angle_at(p->time[5]);
  double a = a_deg * M_PI / 180.0;
  double a_dot = angle_rate(p->time[5]) * M_PI / 180.0;
  double gravity = 0.738 * sin(a);
  double vt = vloss_target(a_deg);
  double q_loss = (vt - vloss) / (TAU_IF * ((vt < vloss) ? refill_x : 1.0));

  /* CVSim: x24 = Pth + con1*gravity with con1 = -3.5/0.738*sin(alpha_max).
     With a time-varying angle we use the current angle: x24 = Pth - 3.5 sin^2(a),
     identical at steady tilt. */
  p->x[24] = theta->vec[31] - 3.5 * sin(a) * sin(a);
  p->dxdt[24] = -7.0 * sin(a) * cos(a) * a_dot;

  p->grav[0]  = theta->vec[121]/2.0 * gravity;
  p->grav[1]  = theta->vec[122]/2.0 * gravity;
  p->grav[2]  = theta->vec[123]/2.0 * gravity;
  p->grav[3]  = theta->vec[124]/2.0 * gravity;
  p->grav[4]  = theta->vec[125]/2.0 * gravity;
  p->grav[5]  = theta->vec[126]/2.0 * gravity;
  p->grav[6]  = theta->vec[127]/3.0 * gravity;
  p->grav[7]  = theta->vec[128]/2.0 * gravity;
  p->grav[8]  = theta->vec[129]/2.0 * gravity;
  p->grav[9]  = theta->vec[130]/2.0 * gravity;
  p->grav[10] = theta->vec[131]/2.0 * gravity;
  p->grav[11] = theta->vec[132]/3.0 * gravity;
  p->grav[12] = theta->vec[133]/3.0 * gravity;
  p->grav[13] = theta->vec[134]/3.0 * gravity;
  p->grav[14] = theta->vec[135]/2.0 * gravity;

  tv->flow[0] = 7.0 / 63.0 * q_loss;
  tv->flow[1] = 40.0 / 63.0 * q_loss;
  tv->flow[2] = 16.0 / 63.0 * q_loss;

  p->tilt[0] = theta->vec[14] * gravity;
  p->tilt[1] = vloss;
  p->tilt_angle = a_deg;
  return 0;
}

int tilt_dispatch(Data_vector *p, Parameter_vector *theta, Tilt_vector *tv,
                  double ts, double tstop)
{
  if (ext_mode) return tilt_ext(p, theta, tv);
  return tilt_orig(p, theta, tv, ts, tstop);
}

/* ------------------------------------------------------------ API */
void cv_default_theta(double *out)
{
  Hemo hemo; Cardiac cardiac; Micro_r micro_r; System_parameters sys;
  Reflex reflex; Timing timing; Parameter_vector tmp;
  memset(&tmp, 0, sizeof(tmp));
  initial_ptr(&hemo, &cardiac, &micro_r, &sys, &reflex, &timing);
  mapping_ptr(&hemo, &cardiac, &micro_r, &sys, &reflex, &timing, &tmp);
  memcpy(out, tmp.vec, sizeof(double) * N_PARAMETER);
}

int cv_n_parameter(void) { return N_PARAMETER; }

int cv_init(const double *theta, int ab, int cp, int mode, double ts, double tstop)
{
  memcpy(TH.vec, theta, sizeof(double) * N_PARAMETER);
  memset(&P, 0, sizeof(P));
  memset(&R, 0, sizeof(R));
  memset(&IMP, 0, sizeof(IMP));
  queue_reset();

  ab_on = ab; cp_on = cp; ext_mode = mode;
  t_start = ts; t_stop = tstop;
  tilt_on = ext_mode ? 1 : (ts >= 0.0 ? 1 : 0);
  ang_t0 = ang_t1 = 0.0; ang0 = ang1 = 0.0; vloss = 0.0;
  for (int k = 0; k < 4; k++) { ees_dr[k] = 0.0; ees_dv[k] = 0.0; }

  estimate_ptr(&P, &TH, &R);
  elastance_ptr(&P, &TH);
  fixvolume_ptr(&P, &R, &TH);
  eqns_ptr(&P, &TH, &R, tilt_on, t_start, t_stop);
  makeImp(&IMP, &TH);
  htry = 1.0e-5;
  b_sys = 0.0; b_dia = 5000.0; b_int = 0.0; b_dt = 0.0;
  return 0;
}

/* Advance to t_end. Angle ramps linearly from its current value to a_end.
   EES hooks held constant over the interval. Writes one row per completed
   beat: [t_end_of_beat, SAP, MAP, DAP, HR]. Returns the number of beats. */
int cv_run(double t_end, double a_end, const double *dr, const double *dv,
           double *beats, int maxb)
{
  double yscale[21];
  double hdid = 0.0, hnext = 0.0;
  int n = 0;
  for (int i = 0; i < 21; i++) yscale[i] = 1.0;
  for (int k = 0; k < 4; k++) { ees_dr[k] = dr[k]; ees_dv[k] = dv[k]; }

  double now = P.time[0];
  double cur = angle_at(now);
  ang_t0 = now; ang0 = cur; ang_t1 = t_end; ang1 = a_end;

  while (P.time[0] < t_end) {
    rkqc_ptr(&P, &R, &TH, htry, 0.001, yscale, &hdid, &hnext,
             tilt_on, t_start, t_stop);
    P.time[1] = sanode(&P, &R, &TH, hdid);
    elastance_ptr(&P, &TH);
    eqns_ptr(&P, &TH, &R, tilt_on, t_start, t_stop);
    P.time[0] += hdid;
    htry = (hnext > 0.001 ? 0.001 : hnext);
    queue_ptr(&P, &IMP, &R, &TH, hdid, ab_on, cp_on);
    fixvolume_ptr(&P, &R, &TH);

    if (ext_mode) {
      double tgt = vloss_target(angle_at(P.time[0]));
      vloss += hdid * (tgt - vloss) / TAU_IF;
    }

    if (R.step_cnt == 1) {
      if (b_dt > 0.0 && n < maxb) {
        beats[5*n + 0] = P.time[0];
        beats[5*n + 1] = b_sys;
        beats[5*n + 2] = b_int / b_dt;
        beats[5*n + 3] = b_dia;
        beats[5*n + 4] = R.hr[2];
        n++;
      }
      b_sys = 0.0; b_dia = 5000.0; b_int = 0.0; b_dt = 0.0;
    } else {
      if (P.x[0] > b_sys) b_sys = P.x[0];
      if (P.x[0] < b_dia) b_dia = P.x[0];
      b_int += P.x[0] * hdid;
      b_dt += hdid;
    }
  }
  return n;
}

double cv_time(void) { return P.time[0]; }

/* effector snapshot: resistance[4], volume[4], hr_reflex, angle, vloss */
void cv_effectors(double *out)
{
  for (int k = 0; k < 4; k++) { out[k] = R.resistance[k]; out[4 + k] = R.volume[k]; }
  out[8] = R.hr[0];
  out[9] = angle_at(P.time[0]);
  out[10] = vloss;
}
