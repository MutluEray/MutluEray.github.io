#!/usr/bin/env python3
"""
cbg_model.py — evaluation oracle around the Fleming, Dunn & Lowery (2020)
cortico-basal-ganglia model (ModelDB 262046), for drift-aware aDBS tuning.

Network construction mirrors Fleming's run_CBG_Model_Amplitude_Modulation_Controller.py
(same populations, bias currents, topology files, synaptic weights, LFP and
beta-ARV biomarker). Changes relative to the original:
  * DBS pulses are delivered as scheduled NEURON events (two per pulse) instead of
    a dt-resolution Vector.play. Memory no longer grows with simulation length,
    and amplitude/frequency can be changed online between segments.
  * Output paths are relative (the original wrote to /Simulation_Output_Results).
  * Only the STN synaptic currents needed for the LFP are recorded.
  * Two drift knobs:
      beta_drive  - scales the cortical beta-burst modulation current
                    (Fleming default 0.02; stands in for dopaminergic state)
      stim_gain   - multiplies the delivered DBS amplitude
                    (stands in for electrode shift / encapsulation)
    Burst statistics can also be swapped by passing a different burst schedule.

All time units are ms, amplitudes mA, frequencies Hz.
"""
import math
import os
import sys
import time

import numpy as np
from scipy import signal

SS_MS = 6000.0          # Fleming's steady-state duration
DT = 0.01               # integration step (ms)
REC_DT = 0.5            # recording interval (ms) -> 2 kHz LFP
POP = 100
PULSE_WIDTH = 0.06      # ms
INTERP_AMPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.50, 3, 4, 5])
INTERP_GPE_PCT = np.array([0, 0, 0, 1, 4, 8, 19, 30, 43, 59, 82, 100, 100, 100])
SIGMA = 0.27            # S/m, tissue conductivity


def find_model_root(start):
    """Folder containing Fleming's model files (searches subfolders, e.g. a nested ModelDB zip)."""
    start = os.path.abspath(start)
    for root, _, files in os.walk(start):
        if "Cortical_Basal_Ganglia_Cell_Classes.py" in files and "Striatal_Spike_Times.npy" in files:
            return root
    raise FileNotFoundError(f"Fleming 2020 model files not found under {start}")


def beta_filter(fs=2000.0, low=21.0, high=29.0):
    nyq = 0.5 * fs
    return signal.cheby1(4, 0.5, [low / nyq, high / nyq], "band")


def beta_arv(lfp, fs=2000.0, discard_ms=100.0):
    """Beta average rectified value of an LFP window (Fleming's biomarker)."""
    b, a = beta_filter(fs)
    x = np.abs(signal.filtfilt(b, a, lfp))
    k = int(discard_ms * fs / 1000.0)
    return float(np.mean(x[k:-k])) if len(x) > 2 * k + 10 else float(np.mean(x))


def teed(amp_ma, freq_hz, pw_ms=PULSE_WIDTH):
    """Total electrical energy delivered per second, up to a constant (amp^2 * f * pw)."""
    return float(amp_ma ** 2 * freq_hz * pw_ms)


class CBGModel:
    def __init__(self, model_dir, mode="eval", ss_ms=SS_MS, burst_file="burst_times_1.txt",
                 level_file="burst_level_1.txt", beta_drive=0.02, seed=3695, horizon_ms=None):
        """mode: 'steady_state' (bursts off, noise from t=0) or 'eval' (load steady state).
        horizon_ms: last simulated time needed. PyNN precomputes membrane noise up to it,
        so memory grows ~3 MB per simulated second; default = steady state + 40 s."""
        self.horizon_ms = horizon_ms if horizon_ms is not None else ss_ms + 40000.0
        self.model_dir = find_model_root(model_dir)
        os.chdir(self.model_dir)                     # model loads its txt/npy files and .bin from cwd
        self._ensure_compiled()                      # NEURON auto-loads ./arm64 (or ./x86_64) at import
        if self.model_dir not in sys.path:
            sys.path.insert(0, self.model_dir)
        self.mode, self.ss_ms, self.seed = mode, ss_ms, seed
        self.stim_gain = 1.0
        self._t_wall_sim = 0.0
        self._sim_ms = 0.0
        self._build(burst_file, level_file, beta_drive)

    # ------------------------------------------------------------------ build
    def _ensure_compiled(self):
        """Compile the model's .mod files once (nrnivmodl) if no build exists yet.
        Must run before NEURON is imported, so NEURON's auto-load picks the build up."""
        import platform
        import subprocess
        if "neuron" in sys.modules:
            return
        arch = platform.machine()
        built = any(os.path.exists(os.path.join(self.model_dir, a, f))
                    for a in (arch, "arm64", "x86_64")
                    for f in ("libnrnmech.dylib", "libnrnmech.so", os.path.join(".libs", "libnrnmech.so")))
        if built:
            return
        print("compiling model mechanisms (nrnivmodl, one time) ...", flush=True)
        r = subprocess.run(["nrnivmodl"], cwd=self.model_dir, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise RuntimeError("nrnivmodl failed in the model folder (Xcode command line tools installed?)")

    def _build(self, burst_file, level_file, beta_drive):
        import neuron
        from pyNN.neuron import (setup, Population, SpikeSourceArray, Projection, FromFileConnector,
                                 StaticSynapse, NoisyCurrentSource, StepCurrentSource, simulator)
        from pyNN.random import RandomDistribution, NumpyRNG
        from pyNN import space
        from pyNN.parameters import Sequence
        from Cortical_Basal_Ganglia_Cell_Classes import (Cortical_Neuron_Type, Interneuron_Type,
                                                         STN_Neuron_Type, GP_Neuron_Type, Thalamic_Neuron_Type)
        from Electrode_Distances import distances_to_electrode, collateral_distances_to_electrode
        import Global_Variables as GV

        self.h = neuron.h
        self.sim = simulator
        setup(timestep=DT, rngseed=self.seed)
        ss = self.ss_ms
        horizon = ss if self.mode == "steady_state" else self.horizon_ms

        cvode = self.h.CVode()
        cvode.active(0)
        cvode.use_fast_imem(1)
        self.cvode = cvode
        v_init = -68
        STN_space = space.RandomStructure(boundary=space.Sphere(2000))

        striatal = np.load("Striatal_Spike_Times.npy", allow_pickle=True)
        if self.mode == "eval":
            for i in range(POP):
                st = striatal[i][0].value
                striatal[i][0] = Sequence(st[st > ss])
        self._striatal = striatal

        self.Cortical = Population(POP, Cortical_Neuron_Type(soma_bias_current_amp=0.245), structure=STN_space, label="Cortical")
        self.Inter = Population(POP, Interneuron_Type(bias_current_amp=0.070), initial_values={"v": v_init}, label="Interneurons")
        self.STN = Population(POP, STN_Neuron_Type(bias_current=-0.125), structure=STN_space, initial_values={"v": v_init}, label="STN")
        self.GPe = Population(POP, GP_Neuron_Type(bias_current=-0.009), initial_values={"v": v_init}, label="GPe")
        self.GPi = Population(POP, GP_Neuron_Type(bias_current=0.006), initial_values={"v": v_init}, label="GPi")
        self.Striatal = Population(POP, SpikeSourceArray(spike_times=striatal[0][0]), label="Striatal")
        self.Thal = Population(POP, Thalamic_Neuron_Type(), initial_values={"v": v_init}, label="Thalamic")
        for i in range(POP):
            self.Striatal[i].spike_times = striatal[i][0]

        # cortical beta-burst modulation (off while reaching steady state, as in Fleming)
        bt = np.loadtxt(burst_file, delimiter=",") + (ss - SS_MS)
        bl = np.loadtxt(level_file, delimiter=",")
        scale = 0.0 if self.mode == "steady_state" else beta_drive
        self.Cortical.inject(StepCurrentSource(times=bt, amplitudes=scale * bl))

        noise_start = 0.0 if self.mode == "steady_state" else ss
        for cell in self.Cortical:
            cell.inject(NoisyCurrentSource(mean=0, stdev=0.005, start=noise_start, stop=horizon, dt=1.0))
        for cell in self.Inter:
            cell.inject(NoisyCurrentSource(mean=0, stdev=0.005, start=noise_start, stop=horizon, dt=1.0))

        xy = np.loadtxt("cortical_xy_pos.txt", delimiter=",")
        for cid, c in enumerate(self.Cortical):
            c.position[0], c.position[1] = xy[0, cid], xy[1, cid]
        xy = np.loadtxt("STN_xy_pos.txt", delimiter=",")
        for cid, c in enumerate(self.STN):
            c.position[0], c.position[1], c.position[2] = xy[0, cid], xy[1, cid], 500

        rec1, rec2, stim_pos = np.array([0, -1500, 250]), np.array([0, 1500, 250]), np.array([0, 0, 250])
        self.d1 = distances_to_electrode(rec1, self.STN)
        self.d2 = distances_to_electrode(rec2, self.STN)
        coll_d = collateral_distances_to_electrode(stim_pos, self.Cortical, L=500, nseg=11)

        gCtxInt = RandomDistribution("uniform", (0, 2.5e-3), rng=NumpyRNG(seed=3695))
        gIntCtx = RandomDistribution("uniform", (0, 6.0e-3), rng=NumpyRNG(seed=3695))
        S = StaticSynapse
        self._prj = [
            Projection(self.Cortical, self.Inter, FromFileConnector("CorticalAxonInterneuron_Connections.txt"), S(weight=gCtxInt, delay=2), source="middle_axon_node", receptor_type="AMPA"),
            Projection(self.Inter, self.Cortical, FromFileConnector("InterneuronCortical_Connections.txt"), S(weight=gIntCtx, delay=2), receptor_type="GABAa"),
            Projection(self.Cortical, self.STN, FromFileConnector("CorticalSTN_Connections.txt"), S(weight=0.12, delay=1), source="collateral(0.5)", receptor_type="AMPA"),
            Projection(self.STN, self.GPe, FromFileConnector("STNGPe_Connections.txt"), S(weight=0.111111, delay=4), source="soma(0.5)", receptor_type="AMPA"),
            Projection(self.GPe, self.GPe, FromFileConnector("GPeGPe_Connections.txt"), S(weight=0.015, delay=4), source="soma(0.5)", receptor_type="GABAa"),
            Projection(self.GPe, self.STN, FromFileConnector("GPeSTN_Connections.txt"), S(weight=0.111111, delay=3), source="soma(0.5)", receptor_type="GABAa"),
            Projection(self.Striatal, self.GPe, FromFileConnector("StriatalGPe_Connections.txt"), S(weight=0.01, delay=1), source="soma(0.5)", receptor_type="GABAa"),
            Projection(self.STN, self.GPi, FromFileConnector("STNGPi_Connections.txt"), S(weight=0.111111, delay=2), source="soma(0.5)", receptor_type="AMPA"),
            Projection(self.GPe, self.GPi, FromFileConnector("GPeGPi_Connections.txt"), S(weight=0.111111, delay=2), source="soma(0.5)", receptor_type="GABAa"),
            Projection(self.GPi, self.Thal, FromFileConnector("GPiThalamic_Connections.txt"), S(weight=3.0, delay=2), source="soma(0.5)", receptor_type="GABAa"),
            Projection(self.Thal, self.Cortical, FromFileConnector("ThalamicCorticalSoma_Connections.txt"), S(weight=5, delay=2), source="soma(0.5)", receptor_type="AMPA"),
            Projection(self.Cortical, self.Thal, FromFileConnector("CorticalSomaThalamic_Connections.txt"), S(weight=0.0, delay=2), source="soma(0.5)", receptor_type="AMPA"),
        ]

        self.STN.record("AMPA.i", sampling_interval=REC_DT)
        self.STN.record("GABAa.i", sampling_interval=REC_DT)

        rho = 1 / (SIGMA * 1e-2)
        rx = (rho / (4 * math.pi)) * (1 / coll_d) * 0.01
        for ii, c in enumerate(self.Cortical):
            c.collateral_rx = Sequence(rx[ii, :].flatten())

        # DBS hooks: global is_xtra (cortical collaterals) and per-GPe IClamps
        self.h.is_xtra = 0.0
        self.gpe_clamps = GV.GPe_stimulation_iclamps[:POP]
        self.gpe_order = [int(i) for i in np.loadtxt("GPe_Stimulation_Order.txt", delimiter=",")]
        self._lfp = []

    # -------------------------------------------------------------- running
    def make_steady_state(self):
        from pyNN.neuron import run_to_steady_state
        t0 = time.perf_counter()
        run_to_steady_state(self.ss_ms)
        wall = time.perf_counter() - t0
        self._t_wall_sim += wall
        self._sim_ms += self.ss_ms
        return wall

    def load_steady_state(self):
        from pyNN.neuron import run_until
        if not os.path.exists(os.path.join(self.model_dir, "steady_state.bin")):
            raise FileNotFoundError("steady_state.bin not found: run step0_probe.py --make-ss first")
        run_until(self.ss_ms + DT, run_from_steady_state=True)
        for i in range(POP):                         # as in Fleming: reload striatal spikes after restore
            self.Striatal[i].spike_times = self._striatal[i][0]
        self.STN.get_data(["AMPA.i", "GABAa.i"], clear=True)   # drop the restore transient

    def _pulse_on(self, amp_eff, n_gpe):
        def f():
            self.h.is_xtra = -amp_eff                # cathodic, as in Fleming
            for i in range(n_gpe):
                self.gpe_clamps[self.gpe_order[i]].amp = 100.0
        return f

    def _pulse_off(self, n_gpe):
        def f():
            self.h.is_xtra = 0.0
            for i in range(n_gpe):
                self.gpe_clamps[self.gpe_order[i]].amp = 0.0
        return f

    def _schedule_segment(self, t_start, dur, amp, freq):
        amp_eff = max(0.0, amp * self.stim_gain)
        if amp_eff <= 0 or freq <= 0:
            return 0
        n_gpe = int(np.interp(amp_eff, INTERP_AMPS, INTERP_GPE_PCT))
        period = 1000.0 / freq
        on, off = self._pulse_on(amp_eff, n_gpe), self._pulse_off(n_gpe)
        n = 0
        t = t_start
        while t + PULSE_WIDTH < t_start + dur:
            self.cvode.event(round(t, 2), on)
            self.cvode.event(round(t + PULSE_WIDTH, 2), off)
            t += period
            n += 1
        return n

    def _lfp_since_last(self):
        a = np.array(self.STN.get_data("AMPA.i", clear=False).segments[0].analogsignals[0])
        g = np.array(self.STN.get_data("GABAa.i", clear=True).segments[0].analogsignals[0])
        syn = a + g
        k = 1 / (4 * math.pi * SIGMA)
        lfp1 = k * np.sum((1 / (self.d1 * 1e-6)) * syn.T, axis=0) * 1e-6
        lfp2 = k * np.sum((1 / (self.d2 * 1e-6)) * syn.T, axis=0) * 1e-6
        return lfp1 - lfp2

    def evaluate(self, amp, freq, dur_ms=2000.0, settle_ms=500.0, stim_gain=None):
        """Apply DBS (amp mA, freq Hz) for dur_ms from the current time; return metrics.
        Beta ARV is computed on the window after settle_ms (stimulation stays on throughout)."""
        from pyNN.neuron import run_until
        if stim_gain is not None:
            self.stim_gain = stim_gain
        t_now = self.sim.state.t
        if t_now + dur_ms > self.horizon_ms:
            raise ValueError(f"evaluation would pass horizon_ms={self.horizon_ms:.0f}; build the model with a larger horizon_ms")
        n_pulses = self._schedule_segment(t_now + DT, dur_ms, amp, freq)
        t0 = time.perf_counter()
        run_until(t_now + dur_ms)
        wall = time.perf_counter() - t0
        self._t_wall_sim += wall
        self._sim_ms += dur_ms
        lfp = self._lfp_since_last()
        self._lfp.append(lfp)
        k = int(settle_ms / REC_DT)
        b = beta_arv(lfp[k:]) if len(lfp) > k + 400 else beta_arv(lfp)
        return {"t_start_ms": t_now, "amp_mA": amp, "freq_Hz": freq, "stim_gain": self.stim_gain,
                "beta_arv": b, "teed": teed(amp * self.stim_gain, freq) if amp > 0 and freq > 0 else 0.0,
                "n_pulses": n_pulses, "wall_s": wall, "wall_s_per_sim_s": wall / (dur_ms / 1000.0)}

    def lfp(self):
        return np.concatenate(self._lfp) if self._lfp else np.array([])
