"""
Simulated Brain Organoid — Izhikevich Spiking Neural Network
=============================================================

A lightweight, vectorized spiking neural network that maps to the
FinalSpark 128-electrode MEA layout and produces biophysically plausible
spike responses to electrical stimulation parameters.

Network architecture
--------------------
- 4 MEAs x 8 electrodes = 32 electrodes (matches FinalSpark platform).
- ``n_neurons_per_electrode`` neurons per electrode (default 16 -> 512 total).
- 80% excitatory (Regular Spiking) / 20% inhibitory (Fast Spiking).
- Random within-MEA connectivity with distance-dependent probability.
- No cross-MEA connections (organoids are physically isolated).

Neuron model
------------
Izhikevich (2003) simple model::

    v' = 0.04 v^2 + 5 v + 140 - u + I
    u' = a (b v - u)
    if v >= 30 mV:  v <- c,  u <- u + d

Parameters (a, b, c, d) are drawn from the standard distributions for
Regular Spiking (excitatory) and Fast Spiking (inhibitory) neuron types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_MEAS = 4
ELECTRODES_PER_MEA = 8
N_ELECTRODES = N_MEAS * ELECTRODES_PER_MEA  # 32

WAVEFORM_N_SAMPLES = 90
WAVEFORM_DURATION_MS = 3.0  # -1 ms to +2 ms around spike peak

# Izhikevich neuron type parameter distributions (mean values)
#   Regular Spiking:  a=0.02, b=0.2, c=-65, d=8
#   Fast Spiking:     a=0.1,  b=0.2, c=-65, d=2
EXCITATORY_FRACTION = 0.8

MIN_CROSS_ELECTRODE_LATENCY_MS = 5.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Spikes produced by a single stimulation event."""

    spike_times_ms: List[float] = field(default_factory=list)
    spike_channels: List[int] = field(default_factory=list)
    spike_amplitudes_uv: List[float] = field(default_factory=list)
    waveforms: List[List[float]] = field(default_factory=list)


@dataclass
class PathwayProfile:
    """Describes the discrete neural pathways between an electrode pair.

    Each pathway produces a tight cluster of spike times (a vertical line
    on a raster plot) at a characteristic latency, matching the multi-path
    response patterns observed in real organoid recordings.
    """

    latencies_ms: List[float]
    jitter_std_ms: List[float]
    reliabilities: List[float]
    spontaneous_rate: float = 0.05


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class SimulatedOrganoid:
    """Izhikevich spiking neural network mapped to a 32-electrode MEA.

    Parameters
    ----------
    n_neurons_per_electrode : int
        Number of simulated neurons assigned to each electrode.
    seed : int or None
        RNG seed for reproducibility.
    dt : float
        Integration timestep in milliseconds.
    connection_probability : float
        Base probability of a synapse between two neurons on the
        same MEA.  Actual probability decays with electrode distance.
    max_connection_dist : int
        Hard electrode-distance cutoff.  Neurons on electrodes
        farther apart than this have zero connection probability,
        preventing long-range cascade chains.
    background_current : float
        Mean thalamic / noise current injected into every neuron
        at each timestep (produces spontaneous background activity).
    synaptic_gain : float
        Multiplier applied to post-synaptic current from firing
        pre-synaptic neurons.  Lower values reduce cascade activity.
    stdp_enabled : bool
        Whether spike-timing-dependent plasticity is active.
    A_plus : float
        LTP learning rate (pre-before-post).
    A_minus : float
        LTD learning rate (post-before-pre).
    tau_plus : float
        LTP time constant in ms.
    tau_minus : float
        LTD time constant in ms.
    w_max : float
        Hard ceiling for excitatory synaptic weights.
    """

    def __init__(
        self,
        n_neurons_per_electrode: int = 16,
        seed: int = 42,
        dt: float = 0.5,
        connection_probability: float = 0.15,
        max_connection_dist: int = 2,
        background_current: float = 1.0,
        synaptic_gain: float = 15.0,
        stdp_enabled: bool = False,
        A_plus: float = 0.005,
        A_minus: float = 0.005,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_max: float = 0.5,
    ) -> None:
        self._rng = np.random.RandomState(seed)
        self._dt = dt
        self._bg_current = background_current
        self._syn_gain = synaptic_gain
        self._max_conn_dist = max_connection_dist
        self._n_per_elec = n_neurons_per_electrode
        self._n_total = N_ELECTRODES * n_neurons_per_electrode

        # STDP configuration
        self._stdp_enabled = stdp_enabled
        self._A_plus = A_plus
        self._A_minus = A_minus
        self._tau_plus = tau_plus
        self._tau_minus = tau_minus
        self._w_max = w_max

        # Which electrode each neuron belongs to (shape: [N])
        self._neuron_electrode = np.repeat(
            np.arange(N_ELECTRODES), n_neurons_per_electrode
        )
        # Which MEA each neuron belongs to
        self._neuron_mea = self._neuron_electrode // ELECTRODES_PER_MEA

        self._build_neuron_parameters()
        self._build_connectivity(connection_probability, max_connection_dist)

        # Persistent state so successive stimulations see ongoing dynamics
        self._v = np.full(self._n_total, -65.0)
        self._u = self._b * self._v

        # STDP state: last spike time per neuron (global clock)
        self._last_spike_time = np.full(self._n_total, -np.inf)
        self._global_time_ms = 0.0

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_neuron_parameters(self) -> None:
        """Assign Izhikevich (a, b, c, d) per neuron."""
        N = self._n_total
        rng = self._rng

        is_excitatory = rng.rand(N) < EXCITATORY_FRACTION
        self._is_excitatory = is_excitatory

        # Randomised parameters following Izhikevich (2003) distributions
        re = rng.rand(N)  # per-neuron random factor

        self._a = np.where(is_excitatory, 0.02, 0.02 + 0.08 * re)
        self._b = np.where(is_excitatory, 0.2, 0.25 - 0.05 * re)
        self._c = np.where(is_excitatory, -65.0 + 15.0 * re ** 2, -65.0)
        self._d = np.where(is_excitatory, 8.0 - 6.0 * re ** 2, 2.0)

    def _build_connectivity(self, base_prob: float, max_dist: int = 2) -> None:
        """Build sparse within-MEA synaptic weight matrix.

        Connections are only possible between neurons whose electrodes
        are at most *max_dist* apart, preventing long-range multi-hop
        cascade chains across the entire MEA.
        """
        N = self._n_total
        rng = self._rng
        elec = self._neuron_electrode

        rows, cols, weights = [], [], []

        for mea_id in range(N_MEAS):
            mask = self._neuron_mea == mea_id
            idx = np.where(mask)[0]
            n_mea = len(idx)
            if n_mea < 2:
                continue

            mea_elecs = elec[idx]
            elec_i = mea_elecs[:, None]
            elec_j = mea_elecs[None, :]
            dist = np.abs(elec_i - elec_j).astype(float)

            # Hard cutoff: zero probability beyond max_dist electrodes
            prob = np.where(dist <= max_dist,
                            base_prob * np.exp(-dist),
                            0.0)
            np.fill_diagonal(prob, 0.0)

            connected = rng.rand(n_mea, n_mea) < prob

            pre_local, post_local = np.where(connected)
            pre_global = idx[pre_local]
            post_global = idx[post_local]

            w = np.where(
                self._is_excitatory[pre_global],
                rng.rand(len(pre_global)) * 0.5,    # excitatory: [0, 0.5]
                -rng.rand(len(pre_global)) * 0.8,   # inhibitory: [-0.8, 0]
            )

            rows.append(pre_global)
            cols.append(post_global)
            weights.append(w)

        rows = np.concatenate(rows) if rows else np.array([], dtype=int)
        cols = np.concatenate(cols) if cols else np.array([], dtype=int)
        weights = np.concatenate(weights) if weights else np.array([], dtype=float)

        # COO arrays for fast per-synapse STDP updates
        self._W_row = rows.astype(np.int32)
        self._W_col = cols.astype(np.int32)
        self._W_data = weights.astype(np.float64)
        self._W_initial = self._W_data.copy()

        # Boolean mask: only excitatory synapses undergo STDP
        self._W_exc_mask = self._is_excitatory[self._W_row]

        self._W = sparse.csr_matrix(
            (self._W_data.copy(), (self._W_row, self._W_col)), shape=(N, N)
        )

        self._build_pathway_profiles()

    # ------------------------------------------------------------------
    # Pathway profiles
    # ------------------------------------------------------------------

    def _build_pathway_profiles(self) -> None:
        """Pre-compute multi-path latency profiles for connected electrode pairs.

        A pair (stim, resp) is "connected" when at least one synapse exists
        from a neuron on *stim* to a neuron on *resp*.  Each connected pair
        gets 1-3 discrete pathways with characteristic latency, jitter, and
        reliability drawn to match patterns observed in real organoid data.
        """
        rng = self._rng
        elec = self._neuron_electrode

        pre_elecs = elec[self._W_row]
        post_elecs = elec[self._W_col]
        connected_pairs = set(zip(pre_elecs.tolist(), post_elecs.tolist()))
        connected_pairs = {(a, b) for a, b in connected_pairs if a != b}

        n_paths_weights = [0.50, 0.35, 0.15]  # P(1 path), P(2), P(3)

        profiles: Dict[Tuple[int, int], PathwayProfile] = {}
        for stim_e, resp_e in connected_pairs:
            n_paths = rng.choice([1, 2, 3], p=n_paths_weights)

            latencies: List[float] = []
            jitters: List[float] = []
            reliabilities: List[float] = []

            lat = float(rng.uniform(10.0, 25.0))
            latencies.append(lat)
            jitters.append(float(rng.uniform(0.3, 1.0)))
            reliabilities.append(float(rng.uniform(0.70, 0.95)))

            if n_paths >= 2:
                lat = lat + float(rng.uniform(5.0, 15.0))
                latencies.append(lat)
                jitters.append(float(rng.uniform(0.3, 1.0)))
                reliabilities.append(float(rng.uniform(0.30, 0.70)))

            if n_paths >= 3:
                lat = lat + float(rng.uniform(5.0, 12.0))
                latencies.append(lat)
                jitters.append(float(rng.uniform(0.3, 1.0)))
                reliabilities.append(float(rng.uniform(0.15, 0.40)))

            profiles[(stim_e, resp_e)] = PathwayProfile(
                latencies_ms=latencies,
                jitter_std_ms=jitters,
                reliabilities=reliabilities,
            )

        self._pathway_profiles = profiles

    # ------------------------------------------------------------------
    # Stimulation
    # ------------------------------------------------------------------

    def _rebuild_W(self) -> None:
        """Rebuild the CSR weight matrix from the mutable COO arrays."""
        self._W = sparse.csr_matrix(
            (self._W_data.copy(), (self._W_row, self._W_col)),
            shape=(self._n_total, self._n_total),
        )

    def _apply_stdp(self, fired: np.ndarray, t_now: float) -> None:
        """Vectorized STDP weight update for all synapses involving *fired* neurons.

        Only excitatory synapses are modified.  Inhibitory weights are left unchanged.
        """
        fired_set = np.zeros(self._n_total, dtype=bool)
        fired_set[fired] = True
        exc = self._W_exc_mask
        cutoff = 5.0 * max(self._tau_plus, self._tau_minus)

        # -- LTP: fired neurons are post-synaptic --------------------------
        # For each excitatory synapse where col (post) just fired, check
        # when its pre (row) last spiked.
        post_mask = fired_set[self._W_col] & exc
        if np.any(post_mask):
            dt_ltp = t_now - self._last_spike_time[self._W_row[post_mask]]
            valid = (dt_ltp > 0) & (dt_ltp < cutoff)
            dw = np.zeros_like(dt_ltp)
            dw[valid] = self._A_plus * np.exp(-dt_ltp[valid] / self._tau_plus)
            self._W_data[post_mask] += dw
            np.clip(self._W_data[post_mask], 0.0, self._w_max,
                    out=self._W_data[post_mask])

        # -- LTD: fired neurons are pre-synaptic ---------------------------
        # For each excitatory synapse where row (pre) just fired, check
        # when its post (col) last spiked.
        pre_mask = fired_set[self._W_row] & exc
        if np.any(pre_mask):
            dt_ltd = self._last_spike_time[self._W_col[pre_mask]] - t_now
            valid = (dt_ltd < 0) & (dt_ltd > -cutoff)
            dw = np.zeros_like(dt_ltd)
            dw[valid] = -self._A_minus * np.exp(dt_ltd[valid] / self._tau_minus)
            self._W_data[pre_mask] += dw
            np.clip(self._W_data[pre_mask], 0.0, self._w_max,
                    out=self._W_data[pre_mask])

    def stimulate(
        self,
        electrode: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: str = "NegativeFirst",
        window_ms: float = 100.0,
    ) -> SimulationResult:
        """Inject current into neurons on *electrode* and simulate *window_ms*.

        Parameters
        ----------
        electrode : int
            Electrode index [0-127].
        amplitude_ua : float
            Stimulation amplitude in micro-amps (max 4.0).
        duration_us : float
            Phase duration in micro-seconds (max 400).
        polarity : str
            ``"NegativeFirst"`` or ``"PositiveFirst"``.
        window_ms : float
            Simulation window after the stimulus onset.

        Returns
        -------
        SimulationResult
            Spike times, channels, amplitudes, and waveforms.
        """
        electrode = int(electrode) % N_ELECTRODES
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        charge = amplitude_ua * duration_us  # µA·µs
        stim_peak = (charge / 1600.0) * 120.0
        stim_spread_steps = max(1, int(2.0 / self._dt))

        n_steps = int(window_ms / self._dt)

        target_mask = self._neuron_electrode == electrode
        stim_mea = electrode // ELECTRODES_PER_MEA

        # Rebuild CSR from (possibly updated) COO data
        if self._stdp_enabled:
            self._rebuild_W()

        v = self._v.copy()
        u = self._u.copy()

        all_fired_steps: List[np.ndarray] = []
        all_fired_neurons: List[np.ndarray] = []

        a, b, c, d = self._a, self._b, self._c, self._d
        dt = self._dt
        W = self._W
        t_offset = self._global_time_ms

        for step in range(n_steps):
            I = self._rng.randn(self._n_total) * self._bg_current

            if step < stim_spread_steps:
                decay = 1.0 - (step / stim_spread_steps)
                I[target_mask] += stim_peak * decay

            # Izhikevich update (half-step Euler for stability)
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
            v += dt * 0.5 * dv
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
            v += dt * 0.5 * dv
            u += dt * a * (b * v - u)

            # Spike detection
            fired = np.where(v >= 30.0)[0]
            if len(fired) > 0:
                all_fired_steps.append(np.full(len(fired), step, dtype=np.int32))
                all_fired_neurons.append(fired.astype(np.int32))

                t_now = t_offset + step * dt
                if self._stdp_enabled:
                    self._apply_stdp(fired, t_now)
                    self._last_spike_time[fired] = t_now

                # Synaptic propagation (use current W; STDP changes take
                # effect next stimulate call via _rebuild_W)
                syn_input = np.asarray(W[fired].sum(axis=0)).flatten()
                v += syn_input * self._syn_gain

                # Reset
                v[fired] = c[fired]
                u[fired] += d[fired]

        self._v = v
        self._u = u
        self._global_time_ms = t_offset + n_steps * dt

        if not all_fired_steps:
            return SimulationResult()

        step_arr = np.concatenate(all_fired_steps)
        neuron_arr = np.concatenate(all_fired_neurons)

        # Only keep spikes from the stimulated MEA
        elec_arr = self._neuron_electrode[neuron_arr]
        mea_arr = elec_arr // ELECTRODES_PER_MEA
        same_mea = mea_arr == stim_mea
        elec_arr = elec_arr[same_mea]

        responding_electrodes = set(elec_arr.tolist())

        # Replace Izhikevich timing with pathway-sampled latencies for
        # cross-electrode responses; keep Izhikevich timing only for
        # the stimulated electrode itself.
        out_times: List[float] = []
        out_elecs: List[int] = []

        for resp_e in responding_electrodes:
            if resp_e == electrode:
                # Self-electrode: keep Izhikevich times (with floor)
                mask = elec_arr == resp_e
                raw_times = step_arr.astype(float)[same_mea][mask] * dt
                for t_ms in raw_times:
                    out_times.append(max(t_ms, 0.0))
                    out_elecs.append(resp_e)
                continue

            profile = self._pathway_profiles.get((electrode, resp_e))
            if profile is None:
                continue

            for path_idx in range(len(profile.latencies_ms)):
                if self._rng.rand() < profile.reliabilities[path_idx]:
                    t_ms = float(
                        profile.latencies_ms[path_idx]
                        + self._rng.randn() * profile.jitter_std_ms[path_idx]
                    )
                    t_ms = max(t_ms, MIN_CROSS_ELECTRODE_LATENCY_MS)
                    if t_ms <= window_ms:
                        out_times.append(round(t_ms, 2))
                        out_elecs.append(resp_e)

            if self._rng.rand() < profile.spontaneous_rate:
                t_ms = float(self._rng.uniform(40.0, 95.0))
                if t_ms <= window_ms:
                    out_times.append(round(t_ms, 2))
                    out_elecs.append(resp_e)

        if not out_times:
            return SimulationResult()

        return self._extract_results_vectorized(
            np.array(out_times, dtype=float),
            np.array(out_elecs, dtype=int),
        )

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def _extract_results_vectorized(
        self,
        time_arr: np.ndarray,
        elec_arr: np.ndarray,
    ) -> SimulationResult:
        """Convert raw neuron spikes into electrode-level results (vectorized)."""
        if len(time_arr) == 0:
            return SimulationResult()

        # Group by (time_bin, electrode) using structured array for speed
        keys = np.stack([time_arr, elec_arr.astype(float)], axis=1)
        unique_keys, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True,
        )

        n_events = len(unique_keys)
        times_out = np.round(unique_keys[:, 0], 2)
        elecs_out = unique_keys[:, 1].astype(int)

        # Amplitude scales with how many neurons on this electrode fired
        base_amps = self._rng.uniform(50.0, 300.0, size=n_events)
        amp_scale = np.minimum(counts, 5) / 3.0
        amps_out = base_amps * amp_scale
        sign = np.where(self._rng.rand(n_events) < 0.5, -1.0, 1.0)
        amps_out *= sign
        amps_out = np.round(amps_out, 2)

        # Vectorized waveform generation
        waveforms = self._generate_waveforms_batch(amps_out)

        return SimulationResult(
            spike_times_ms=times_out.tolist(),
            spike_channels=elecs_out.tolist(),
            spike_amplitudes_uv=amps_out.tolist(),
            waveforms=waveforms,
        )

    @staticmethod
    def _generate_waveforms_batch(peak_uvs: np.ndarray) -> List[List[float]]:
        """Vectorized biphasic waveform generation for multiple spikes."""
        n = WAVEFORM_N_SAMPLES
        centre = n // 3
        width = n / 6.0
        t = (np.arange(n) - centre) / width                     # (90,)
        shape = -t * np.exp(-(t * t) / 2.0)                     # (90,)
        # Outer product: (n_spikes, 90)
        all_waveforms = peak_uvs[:, None] * shape[None, :]
        return np.round(all_waveforms, 3).tolist()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, keep_weights: bool = False) -> None:
        """Reset membrane potentials, spike traces, and optionally weights.

        Parameters
        ----------
        seed : int or None
            If given, reseed the RNG for reproducibility.
        keep_weights : bool
            If True, retain the current (learned) synaptic weights.
            If False (default), restore the initial weight matrix.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._v = np.full(self._n_total, -65.0)
        self._u = self._b * self._v
        self._last_spike_time = np.full(self._n_total, -np.inf)
        self._global_time_ms = 0.0
        if not keep_weights:
            self._W_data = self._W_initial.copy()
            self._rebuild_W()

    @property
    def n_neurons(self) -> int:
        return self._n_total

    @property
    def n_electrodes(self) -> int:
        return N_ELECTRODES
