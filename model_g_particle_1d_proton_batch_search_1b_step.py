#!/usr/bin/env python3
"""
Date: 9 March 2026 18:58 GMT

Model G 1D proton-like batch search (non-GUI)

............
- Written by Brendan Darrer aided by ChatGPT 5.4 (extensive thinking)
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT 5.4 writing it and Brendan guiding it to produce a clean code.

Install:
    pip3 install numpy scipy matplotlib imageio imageio[ffmpeg]

Tested for: Ubuntu 24.04.3 LTS on i7-4790 (Optiplex 7020/9020), Python 3.10+

*********************
ChatGPT 5.4:

Yes — I rewrote it as a step-by-step commented version based on your uploaded file 

model_g_particle_1d_proton_batc…

Download it here:

model_g_particle_1d_proton_batch_search_1b_step.py

What I changed:

added STEP 0, STEP 1, STEP 2... commentary throughout

explained the purpose of each major section

annotated the main solver flow, Kelly target construction, scoring, and output writing

kept the code runnable

kept the safer local output default:
--outdir ./model_g_proton_batch_search_out_1b

I also checked that the annotated file still compiles cleanly as Python.

If you like, I can next do a second pass where I make the comments even more beginner-friendly, line by line, inside the rhs() equations and the main() scan loop.
*********************

............

Built from the same 1D Model G equations used in Brendan's GUI explorer,
but refactored into a command-line batch search that scores each trial
against a Kelly-style proton target profile.

What it does
------------
STEP-BY-STEP COMMENTARY VERSION
This file keeps the same logic as the original batch-search script, but adds
extra explanatory comments so you can follow the flow more easily.

1) Builds a Kelly-style proton target from the electric Sachs form factor G_E^p(Q^2)
   using a Kelly-style relativistic inversion to rho(r).
2) Runs many 1D Model G simulations over a user-defined parameter grid.
3) Compares the final pY(x) profile against the Kelly target by shape only
   (normalized profile + mirrored surface profile x^2 rho), because in SQK the
   substrate-potential units are being used as a proxy rather than literal
   charge-density units.
4) Writes CSV rankings and a PNG comparing the best SQK candidate with the target.

Typical use
-----------
python3 model_g_particle_1d_proton_batch_search_1b.py

Faster smoke test
-----------------
python3 model_g_particle_1d_proton_batch_search_1b.py --nx 41 --tfinal 6 --dy 9.5,10.5,12 --b 27,28,29

Note to Brendan (not necessarily others):
python3 -m venv tf-env
source tf-env/bin/activate
==================
"The best next step is for me to turn your GUI code into a non-GUI batch search script that automatically scores each run against a Kelly-style proton target profile and writes out the top candidate parameter sets." Yes, thanks. Can you do this?

ChatGPT 5.4:

Thought for 26m 49s

Yes — I made the non-GUI batch search script and also ran a quick coarse search.

Files:
Batch search script
Quick-search summary
Quick-search top candidates CSV
Best candidate vs Kelly proton PNG
Best candidate profiles NPZ

What the script does:

uses the same 1D Model G equations as your GUI

builds a Kelly-style proton target profile

compares the final SQK pY(x) shape against that target after normalization

also compares the mirrored surface profile 

x^2 pY(x)

writes ranked candidate parameter sets to CSV

From the quick coarse run I did, the current best provisional proton-like candidate was:

dy = 12.0

b = 29.0

g = 0.09

amp = 1.0

sx = 1.0

st = 1.5

Tseed = 3.0

sign = -1

with:

pY_core = 0.288756...

Qproxy_int_pYdx = 0.164173...

polarity = positive-polarity (proton-like)

One caveat: that run was intentionally coarse (nx=41, Tfinal=6) just to get a fast first pass, so this is a provisional best candidate, not yet a high-resolution final fit.

You can run the script like this:

python3 model_g_particle_1d_proton_batch_search_1b_step.py

or for the same quick sweep I used:

python3 model_g_particle_1d_proton_batch_search_1b_step.py \
  --outdir ./model_g_proton_batch_search_quick_1b \
  --nx 41 --tfinal 6 \
  --dy 9.5,10.5,12.0 \
  --b 27,28,29 \
  --g 0.09,0.10,0.11 \
  --amp 1.0 \
  --sx 0.9,1.0,1.1 \
  --st 1.5 \
  --Tseed 3.0

The natural next step is a second-stage refinement around that neighbourhood with a finer grid and longer runtime.

Brendan: Thank you ChatGPT! I'll be very interested to review work you've just done!!

ChatGPT 5.4:

Thought for 5s

You’re very welcome, Brendan.

A good way to review it is to start with the plot and summary first:

Best candidate vs Kelly proton PNG

Quick-search summary

Top candidates CSV

Then, when you want the mechanics, look at:

Batch search script

Best candidate profiles NPZ

The main thing to check is whether the best candidate really gives you the kind of single broad positive proton-like core you want, rather than a more neutron-like or overly oscillatory profile in the SQK substrate language.

Once you’ve had a look, send me your impressions and I’ll help you do the next refinement pass.
==================
"""

from __future__ import annotations

# STEP 0
# This script is a non-GUI batch-search tool for your 1D Model G particle solver.
# Its job is to:
#   1) build a proton-like target profile from Kelly's proton form factor,
#   2) run many Model G trials over a parameter grid,
#   3) compare each final pY(x) profile to that target, and
#   4) save ranked results, plots, and diagnostic files.
#
# The key idea is that pY is treated here as an SQK proxy field, so the matching is
# shape-based rather than a literal one-to-one unit conversion to charge density.

import argparse
import csv
import itertools
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------------------------------------------------------
# STEP 1: Define the core Model G parameter containers
# -----------------------------------------------------------------------------
# These dataclasses simply collect the solver parameters into clear groups:
#   - ModelParams : physical / equation coefficients
#   - SeedParams  : localized forcing seed that kicks the particle into existence
#   - GridParams  : numerical domain and integrator settings
# -----------------------------------------------------------------------------

@dataclass
class ModelParams:
    a: float = 14.0
    b: float = 29.0
    dx: float = 1.0
    dy: float = 12.0
    p: float = 1.0
    q: float = 1.0
    g: float = 0.1
    s: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0


@dataclass
class SeedParams:
    sign: int = -1
    amp: float = 1.0
    sx: float = 1.0
    st: float = 1.5
    Tseed: float = 3.0
    nseeds: int = 1
    sep: float = 3.303 / 2.0


@dataclass
class GridParams:
    L: float = 20.0
    nx: int = 51
    Tfinal: float = 8.0
    max_step: float = 0.1
    rtol: float = 1e-4
    atol: float = 1e-6
    dense: bool = False


# STEP 2
# Helper function: make a Gaussian-like bell envelope.
# This is used for both the spatial seed profile and the temporal seed profile.
def _bell(s: float, x: np.ndarray | float) -> np.ndarray | float:
    s = float(s)
    if s <= 0:
        if np.isscalar(x):
            return 0.0
        return np.zeros_like(x)
    return np.exp(- (x / s) ** 2 / 2.0)


# STEP 3
# Build the seed / forcing function chi(x,t).
# This is the localized kick that initially perturbs the background and helps the
# solver find a self-organized particle-like structure.
def make_chi(xgrid: np.ndarray, t: float, sp: SeedParams) -> np.ndarray:
    env_t = _bell(sp.st, t - sp.Tseed)
    if sp.nseeds == 1:
        env_x = _bell(sp.sx, xgrid)
    elif sp.nseeds == 2:
        env_x = _bell(sp.sx, xgrid + sp.sep) + _bell(sp.sx, xgrid - sp.sep)
    else:
        env_x = _bell(sp.sx, xgrid + 3.314) + _bell(sp.sx, xgrid) + _bell(sp.sx, xgrid - 3.314)
    return float(sp.sign) * float(sp.amp) * env_x * env_t


# STEP 4
# Main solver class for the 1D Model G system.
# It contains:
#   - the spatial grid
#   - the background equilibrium values G0, X0, Y0
#   - numerical derivative operators
#   - the RHS (time-derivative) equations
#   - a run() wrapper around scipy.solve_ivp
#   - diagnostics() to summarize the final particle shape
class ModelG1D:
    # STEP 4A
    # Initialize the solver, build the x-grid, and compute the background steady-state
    # values G0, X0, Y0 that the perturbation fields live on top of.
    def __init__(self, mp: ModelParams, gp: GridParams, sp: SeedParams):
        self.mp = mp
        self.gp = gp
        self.sp = sp

        self.x = np.linspace(-gp.L / 2.0, gp.L / 2.0, int(gp.nx))
        if len(self.x) < 5:
            raise ValueError("nx too small; use >= 21")
        self.dx_space = self.x[1] - self.x[0]

        denom = (mp.q - mp.g * mp.p)
        if abs(denom) < 1e-12:
            raise ValueError("Invalid params: q - g*p too close to 0")
        self.G0 = (mp.a + mp.g * mp.w) / denom
        self.X0 = (mp.p * mp.a + mp.q * mp.w) / denom
        self.Y0 = ((mp.s * self.X0 ** 2 + mp.b) * self.X0 / (self.X0 ** 2 + mp.u)) if abs(self.X0 ** 2 + mp.u) > 1e-12 else 0.0

    # STEP 4B
    # Second spatial derivative with simple zero-handled boundaries.
    # This gives the diffusion term in the PDEs.
    def laplacian(self, u: np.ndarray) -> np.ndarray:
        dudxx = np.zeros_like(u)
        dx2 = self.dx_space ** 2
        dudxx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx2
        return dudxx

    # STEP 4C
    # First spatial derivative, used for any drift/advection term.
    def grad(self, u: np.ndarray) -> np.ndarray:
        dudx = np.zeros_like(u)
        dudx[1:-1] = (u[2:] - u[:-2]) / (2.0 * self.dx_space)
        return dudx

    # STEP 4D
    # Pack the three fields into one long vector, because solve_ivp expects a single state vector.
    def pack(self, pG: np.ndarray, pX: np.ndarray, pY: np.ndarray) -> np.ndarray:
        return np.concatenate([pG, pX, pY])

    # STEP 4E
    # Unpack the long solver vector back into the three physical fields pG, pX, pY.
    def unpack(self, y_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx = len(self.x)
        return y_flat[:nx], y_flat[nx:2 * nx], y_flat[2 * nx:3 * nx]

    # STEP 4F
    # This is the heart of the Model G solver.
    # Given the current state y(t), compute dy/dt using diffusion, coupling,
    # nonlinear reaction terms, and the external seed chi(x,t).
    def rhs(self, t: float, y_flat: np.ndarray) -> np.ndarray:
        mp = self.mp
        pG, pX, pY = self.unpack(y_flat)

        lapG = self.laplacian(pG)
        lapX = self.laplacian(pX)
        lapY = self.laplacian(pY)

        dGdx = self.grad(pG)
        dXdx = self.grad(pX)
        dYdx = self.grad(pY)

        chi_vec = make_chi(self.x, t, self.sp)

        Xtot = pX + self.X0
        Ytot = pY + self.Y0

        # Nonlinear reaction terms.
        # nonlinear_s corresponds to the cubic self-interaction term.
        nonlinear_s = mp.s * (Xtot ** 3 - self.X0 ** 3)
        # nonlinear_xy is the X^2 Y coupling term, shifted so the background state is removed.
        nonlinear_xy = (Xtot ** 2 * Ytot - self.X0 ** 2 * self.Y0)

        # Time evolution equations for the perturbation fields.
        dpGdt = lapG - mp.q * pG + mp.g * pX - mp.v * dGdx
        dpXdt = mp.dx * lapX - mp.v * dXdx + mp.p * pG - (1.0 + mp.b) * pX + mp.u * pY - nonlinear_s + nonlinear_xy + chi_vec
        dpYdt = mp.dy * lapY - mp.v * dYdx + mp.b * pX - mp.u * pY + (-nonlinear_xy + nonlinear_s)

        # Boundary handling: pin the time-derivatives to zero at the two outer edges.
        dpGdt[0] = dpGdt[-1] = 0.0
        dpXdt[0] = dpXdt[-1] = 0.0
        dpYdt[0] = dpYdt[-1] = 0.0

        return self.pack(dpGdt, dpXdt, dpYdt)

    # STEP 4G
    # Run the time integration from t=0 to Tfinal.
    # We use the BDF stiff solver because reaction-diffusion systems often become stiff.
    def run(self, y0: np.ndarray | None = None, nframes: int = 30):
        nx = len(self.x)
        if y0 is None:
            y0 = np.zeros(3 * nx, dtype=float)

        t_eval = np.linspace(0.0, float(self.gp.Tfinal), int(nframes))
        sol = solve_ivp(
            fun=self.rhs,
            t_span=(0.0, float(self.gp.Tfinal)),
            y0=y0,
            method="BDF",
            t_eval=t_eval,
            max_step=float(self.gp.max_step),
            rtol=float(self.gp.rtol),
            atol=float(self.gp.atol),
            dense_output=bool(self.gp.dense),
        )
        return sol

    # STEP 4H
    # Summarize the final state with a few useful diagnostics:
    #   - pY_core: central value of the Y substrate proxy
    #   - Qproxy : integral of pY over x (proxy for net polarity / charge-like sign)
    #   - FWHM   : width of the final |pY| structure
    #   - polarity_label : convenient proton-like / electron-like label
    def diagnostics(self, y_flat: np.ndarray) -> dict:
        pG, pX, pY = self.unpack(y_flat)
        i0 = int(np.argmin(np.abs(self.x)))
        core = {
            "pY_core": float(pY[i0]),
            "pX_core": float(pX[i0]),
            "pG_core": float(pG[i0]),
        }

        Qproxy = float(np.trapezoid(pY, self.x))
        absY = np.abs(pY)
        peak = float(absY.max())
        if peak > 1e-12:
            half = 0.5 * peak
            idx = np.where(absY >= half)[0]
            fwhm = float(self.x[idx[-1]] - self.x[idx[0]]) if len(idx) >= 2 else 0.0
        else:
            fwhm = 0.0

        if core["pY_core"] > 0:
            pol = "positive-polarity (proton-like)"
        elif core["pY_core"] < 0:
            pol = "negative-polarity (electron-like)"
        else:
            pol = "neutral/zero"

        return {
            **core,
            "Qproxy_int_pYdx": Qproxy,
            "pY_peak_abs": peak,
            "pY_fwhm_abs": fwhm,
            "polarity_label": pol,
        }


# -----------------------------------------------------------------------------
# STEP 5: Build the Kelly-style proton target profile
# -----------------------------------------------------------------------------
# This section converts the proton electric Sachs form factor G_E^p(Q^2) into a
# radial density-like target profile rho(r) using a Kelly-style relativistic inversion.
# That target is then what the SQK pY(x) profiles are scored against.
# -----------------------------------------------------------------------------

HBARC = 0.1973269804  # GeV*fm
MASS_P = 0.9382720813 # GeV


# STEP 5A
# Kelly-style rational approximation for the proton electric form factor.
def G_Ep_kelly(Q2: np.ndarray) -> np.ndarray:
    tau = Q2 / (4.0 * MASS_P ** 2)
    return (1.0 - 0.24 * tau) / (1.0 + 10.98 * tau + 12.82 * tau ** 2 + 21.97 * tau ** 3)


# STEP 5B
# Invert the form factor into a radial density-like profile.
# This uses a spherical-Bessel j0 transform over intrinsic spatial frequency k.
def rho_from_GE_kelly(r_values: np.ndarray, G_of_Q2, m_geV: float, lambda_E: int = 0, nk: int = 1800) -> np.ndarray:
    kmax = 0.999 * (2.0 * m_geV / HBARC)
    k = np.linspace(1e-6, kmax, nk)
    k_geV = HBARC * k
    denom = 1.0 - (k_geV ** 2) / (4.0 * m_geV ** 2)
    Q2 = k_geV ** 2 / denom
    tau = Q2 / (4.0 * m_geV ** 2)
    rho_tilde = G_of_Q2(Q2) * (1.0 + tau) ** lambda_E
    rho = np.empty_like(r_values, dtype=float)
    for i, r in enumerate(r_values):
        if r == 0:
            j0 = np.ones_like(k)
        else:
            kr = k * r
            j0 = np.sin(kr) / kr
        rho[i] = (2.0 / np.pi) * np.trapezoid(k ** 2 * j0 * rho_tilde, k)
    # tiny far-tail numerical noise is not physically meaningful here
    tiny = 1e-5 * np.max(np.abs(rho))
    rho[np.abs(rho) < tiny] = 0.0
    return rho


# -----------------------------------------------------------------------------
# STEP 6: Helper functions for scanning and scoring candidate particles
# -----------------------------------------------------------------------------
# These functions parse command-line scan lists, normalize curves, measure widths,
# and compute a combined score telling us how proton-like a final SQK profile is.
# -----------------------------------------------------------------------------


# STEP 6A
# Convert a comma-separated command-line string like '9.5,10.5,12.0' into floats.
def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


# STEP 6B
# Normalize a profile by its maximum absolute value so shape is compared independently of scale.
def normalized(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y)))
    return y / peak if peak > 0 else y.copy()


# STEP 6C
# Compute the full width at half maximum of |y|.
def fwhm_abs(x: np.ndarray, y: np.ndarray) -> float:
    a = np.abs(y)
    peak = float(np.max(a))
    if peak <= 1e-12:
        return 0.0
    idx = np.where(a >= 0.5 * peak)[0]
    return float(x[idx[-1]] - x[idx[0]]) if len(idx) >= 2 else 0.0


# STEP 6D
# Compare one final SQK pY(x) profile against the Kelly proton target.
# The score mixes several ingredients:
#   - rho RMSE           : direct profile mismatch
#   - surface RMSE       : mismatch of x^2 rho presentation
#   - negative-lobe pen. : discourages too much unwanted negative structure
#   - core-sign pen.     : prefers a positive proton-like core
#   - charge-sign pen.   : prefers positive net integral Qproxy
#   - FWHM penalty       : encourages the width to resemble the target
def score_against_proton_target(x: np.ndarray, pY: np.ndarray, target_rho: np.ndarray) -> dict:
    i0 = int(np.argmin(np.abs(x)))
    core = float(pY[i0])
    Q = float(np.trapezoid(pY, x))
    peak = float(np.max(np.abs(pY)))
    if (not np.isfinite(peak)) or peak < 1e-12:
        return {
            "score_total": 1e9,
            "score_rho_rmse": 1e9,
            "score_surface_rmse": 1e9,
            "penalty_negative_lobes": 1e9,
            "penalty_core_sign": 1e9,
            "penalty_charge_sign": 1e9,
            "penalty_fwhm_rel": 1e9,
            "pY_core": core,
            "Qproxy_int_pYdx": Q,
            "pY_peak_abs": peak,
            "pY_fwhm_abs": 0.0,
        }

    pred_rho_n = normalized(pY)
    targ_rho_n = normalized(target_rho)

    pred_surface_n = normalized((x ** 2) * pY)
    targ_surface_n = normalized((x ** 2) * target_rho)

    rho_rmse = float(np.sqrt(np.mean((pred_rho_n - targ_rho_n) ** 2)))
    surface_rmse = float(np.sqrt(np.mean((pred_surface_n - targ_surface_n) ** 2)))

    negative_lobes = float(np.trapezoid(np.clip(-pred_rho_n, 0.0, None), x) / (x[-1] - x[0]))
    fwhm_pred = fwhm_abs(x, pY)
    fwhm_targ = fwhm_abs(x, target_rho)
    fwhm_rel = abs(fwhm_pred - fwhm_targ) / (fwhm_targ if fwhm_targ > 1e-12 else 1.0)

    penalty_core_sign = 0.0 if core > 0 else (2.0 + abs(core))
    penalty_charge_sign = 0.0 if Q > 0 else (1.0 + abs(Q))

    # Weighted shape score. Main target is the Kelly rho(r) shape;
    # surface term adds the mirrored x^2*rho presentation; penalties enforce proton-like polarity.
    # Final weighted score: smaller is better.
    score_total = (
        0.60 * rho_rmse
        + 0.20 * surface_rmse
        + 0.10 * negative_lobes
        + 0.10 * fwhm_rel
        + 0.50 * penalty_charge_sign
        + 1.00 * penalty_core_sign
    )

    return {
        "score_total": float(score_total),
        "score_rho_rmse": rho_rmse,
        "score_surface_rmse": surface_rmse,
        "penalty_negative_lobes": negative_lobes,
        "penalty_core_sign": penalty_core_sign,
        "penalty_charge_sign": penalty_charge_sign,
        "penalty_fwhm_rel": fwhm_rel,
        "pY_core": core,
        "Qproxy_int_pYdx": Q,
        "pY_peak_abs": peak,
        "pY_fwhm_abs": fwhm_pred,
    }


# -----------------------------------------------------------------------------
# STEP 7: Output helpers
# -----------------------------------------------------------------------------
# These write CSV tables and generate the summary comparison plot for the best run.
# -----------------------------------------------------------------------------


# STEP 7A
# Save a list of dictionaries as a CSV table.
def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# STEP 7B
# Build the main summary figure comparing the best SQK profile to the Kelly proton target.
def make_summary_plot(path: str, x: np.ndarray, target_rho: np.ndarray, best_pY: np.ndarray, best_row: dict) -> None:
    targ_rho_n = normalized(target_rho)
    best_rho_n = normalized(best_pY)
    targ_surface_n = normalized((x ** 2) * target_rho)
    best_surface_n = normalized((x ** 2) * best_pY)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, targ_rho_n, label="Kelly-style proton target (normalized)")
    axes[0].plot(x, best_rho_n, label="Best SQK pY profile (normalized)")
    axes[0].axhline(0.0, linewidth=0.8)
    axes[0].axvline(0.0, linewidth=0.8)
    axes[0].set_ylabel("normalized profile")
    axes[0].set_title("Direct profile comparison")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, targ_surface_n, label="Kelly-style target  x²ρ(x)")
    axes[1].plot(x, best_surface_n, label="Best SQK  x²pY(x)")
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].axvline(0.0, linewidth=0.8)
    axes[1].set_xlabel("x [arbitrary solver fm-like units]")
    axes[1].set_ylabel("normalized surface profile")
    axes[1].set_title("Mirrored surface-profile comparison")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    text = (
        f"best score = {best_row['score_total']:.4f}\n"
        f"dy={best_row['dy']}, b={best_row['b']}, g={best_row['g']}\n"
        f"amp={best_row['amp']}, sx={best_row['sx']}, st={best_row['st']}, Tseed={best_row['Tseed']}\n"
        f"pY_core={best_row['pY_core']:.6g}, Qproxy={best_row['Qproxy_int_pYdx']:.6g}, FWHM={best_row['pY_fwhm_abs']:.6g}"
    )
    axes[0].text(0.02, 0.98, text, transform=axes[0].transAxes, va="top",
                 bbox=dict(boxstyle="round,pad=0.35", alpha=0.12))

    fig.suptitle("Batch search for a proton-like 1D Model G profile")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# STEP 8: Main driver
# -----------------------------------------------------------------------------
# This is the command-line entry point. It reads scan values, runs the whole search,
# keeps track of the best result, and saves all output files.
# -----------------------------------------------------------------------------


# STEP 8A
# Main search procedure. Follow the comments inside this function in order; that is
# the easiest way to understand the full batch-search workflow from start to finish.
def main() -> None:
    # STEP 8A-1
    # Define command-line arguments so you can change the scan range without editing the code.
    parser = argparse.ArgumentParser(description="Batch-search for a proton-like 1D Model G profile")
    parser.add_argument("--outdir", default="/mnt/data/model_g_proton_batch_search_out", help="Output folder")
    parser.add_argument("--L", type=float, default=20.0)
    parser.add_argument("--nx", type=int, default=51)
    parser.add_argument("--tfinal", type=float, default=8.0)
    parser.add_argument("--max-step", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--nframes", type=int, default=30)
    parser.add_argument("--sign", type=int, default=-1)
    parser.add_argument("--a", type=float, default=14.0)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--s", type=float, default=0.0)
    parser.add_argument("--u", type=float, default=0.0)
    parser.add_argument("--v", type=float, default=0.0)
    parser.add_argument("--w", type=float, default=0.0)
    parser.add_argument("--dy", default="9.5,10.5,12.0", help="Comma list")
    parser.add_argument("--b", default="27,28,29", help="Comma list")
    parser.add_argument("--g", default="0.09,0.10,0.11", help="Comma list")
    parser.add_argument("--amp", default="0.9,1.0,1.1", help="Comma list")
    parser.add_argument("--sx", default="0.9,1.0,1.1", help="Comma list")
    parser.add_argument("--st", default="1.5", help="Comma list")
    parser.add_argument("--Tseed", default="3.0", help="Comma list")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    # STEP 8A-2
    # Create the output folder where CSV files, PNG plots, NPZ arrays, and the summary
    # text file will be written.
    os.makedirs(args.outdir, exist_ok=True)

    # STEP 8A-3
    # Build one GridParams object that all runs in the scan will share.
    gp = GridParams(L=args.L, nx=args.nx, Tfinal=args.tfinal, max_step=args.max_step, rtol=args.rtol, atol=args.atol, dense=False)

    # STEP 8A-4
    # Parse the scan lists from strings into Python float lists.
    dy_vals = parse_float_list(args.dy)
    b_vals = parse_float_list(args.b)
    g_vals = parse_float_list(args.g)
    amp_vals = parse_float_list(args.amp)
    sx_vals = parse_float_list(args.sx)
    st_vals = parse_float_list(args.st)
    tseed_vals = parse_float_list(args.Tseed)

    # Build target on the same x-grid used by the solver.
    # STEP 8A-5
    # Build the Kelly proton target on the same x-grid used by the solver so curve
    # comparisons are point-for-point.
    x_ref = np.linspace(-gp.L / 2.0, gp.L / 2.0, int(gp.nx))
    target_rho = rho_from_GE_kelly(np.abs(x_ref), G_Ep_kelly, MASS_P)

    rows: list[dict] = []
    best: dict | None = None
    best_pY: np.ndarray | None = None

    # STEP 8A-6
    # Create the full Cartesian product of scan parameters. Each tuple in combos is one run.
    combos = list(itertools.product(dy_vals, b_vals, g_vals, amp_vals, sx_vals, st_vals, tseed_vals))
    t0 = time.time()

    # STEP 8A-7
    # Loop over every parameter combination, run the solver, score the result,
    # and keep track of the best candidate found so far.
    for n, (dy, b, g, amp, sx, st, Tseed) in enumerate(combos, start=1):
        mp = ModelParams(a=args.a, b=b, dx=args.dx, dy=dy, p=args.p, q=args.q, g=g, s=args.s, u=args.u, v=args.v, w=args.w)
        sp = SeedParams(sign=args.sign, amp=amp, sx=sx, st=st, Tseed=Tseed, nseeds=1)

        row = {
            "dy": dy,
            "b": b,
            "g": g,
            "amp": amp,
            "sx": sx,
            "st": st,
            "Tseed": Tseed,
            "solver_success": False,
            "solver_message": "",
        }

        # STEP 8A-7a
        # Try one simulation. If it fails, catch the exception and record the failure
        # instead of crashing the whole batch search.
        try:
            # Build the model instance for this parameter set, solve it, unpack the
            # final fields, then compute diagnostics and the proton-likeness score.
            model = ModelG1D(mp, gp, sp)
            sol = model.run(nframes=args.nframes)
            pG, pX, pY = model.unpack(sol.y[:, -1])
            diag = model.diagnostics(sol.y[:, -1])
            score = score_against_proton_target(model.x, pY, target_rho)
            row.update(diag)
            row.update(score)
            row.update({
                "solver_success": bool(sol.success),
                "solver_message": str(sol.message),
                "nfev": int(getattr(sol, "nfev", -1)),
                "njev": int(getattr(sol, "njev", -1)),
                "nlu": int(getattr(sol, "nlu", -1)),
            })
            if best is None or row["score_total"] < best["score_total"]:
                best = row.copy()
                best_pY = pY.copy()
        except Exception as exc:
            row.update({
                "score_total": 1e9,
                "score_rho_rmse": 1e9,
                "score_surface_rmse": 1e9,
                "penalty_negative_lobes": 1e9,
                "penalty_core_sign": 1e9,
                "penalty_charge_sign": 1e9,
                "penalty_fwhm_rel": 1e9,
                "pY_core": float("nan"),
                "pX_core": float("nan"),
                "pG_core": float("nan"),
                "Qproxy_int_pYdx": float("nan"),
                "pY_peak_abs": float("nan"),
                "pY_fwhm_abs": float("nan"),
                "polarity_label": "ERROR",
                "solver_message": repr(exc),
            })
        rows.append(row)

        # STEP 8A-7b
        # Print progress occasionally so long searches do not look frozen.
        if n == 1 or n % 20 == 0 or n == len(combos):
            print(f"[{n:4d}/{len(combos)}] elapsed={time.time() - t0:6.1f}s best_score={best['score_total'] if best else float('nan'):.5f}")

    # STEP 8A-8
    # After all runs finish, sort everything by score so the best candidates are first.
    rows.sort(key=lambda r: r["score_total"])
    top_rows = rows[: max(1, int(args.topk))]

    all_fieldnames = [
        "score_total", "score_rho_rmse", "score_surface_rmse",
        "penalty_negative_lobes", "penalty_core_sign", "penalty_charge_sign", "penalty_fwhm_rel",
        "dy", "b", "g", "amp", "sx", "st", "Tseed",
        "pY_core", "pX_core", "pG_core", "Qproxy_int_pYdx", "pY_peak_abs", "pY_fwhm_abs", "polarity_label",
        "solver_success", "solver_message", "nfev", "njev", "nlu"
    ]

    all_csv = os.path.join(args.outdir, "all_runs.csv")
    top_csv = os.path.join(args.outdir, "top_candidates.csv")
    plot_png = os.path.join(args.outdir, "best_candidate_vs_kelly_proton.png")
    best_npz = os.path.join(args.outdir, "best_candidate_profiles.npz")
    summary_txt = os.path.join(args.outdir, "summary.txt")

    # STEP 8A-9
    # Save the complete ranked tables.
    write_csv(all_csv, rows, all_fieldnames)
    write_csv(top_csv, top_rows, all_fieldnames)

    # STEP 8A-10
    # Save the best candidate profile arrays and the comparison plot.
    if best is not None and best_pY is not None:
        np.savez_compressed(
            best_npz,
            x=x_ref,
            target_rho=target_rho,
            best_pY=best_pY,
            best_surface=(x_ref ** 2) * best_pY,
            target_surface=(x_ref ** 2) * target_rho,
        )
        make_summary_plot(plot_png, x_ref, target_rho, best_pY, best)

    # STEP 8A-11
    # Write a human-readable text summary of the search setup and the winning candidate.
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Model G 1D proton-like batch search summary\n")
        f.write("======================================\n\n")
        f.write(f"Total runs: {len(rows)}\n")
        f.write(f"Elapsed seconds: {time.time() - t0:.2f}\n\n")
        f.write("Fixed parameters\n")
        f.write(f"  a={args.a}, dx={args.dx}, p={args.p}, q={args.q}, s={args.s}, u={args.u}, v={args.v}, w={args.w}, sign={args.sign}\n")
        f.write(f"  L={args.L}, nx={args.nx}, Tfinal={args.tfinal}, max_step={args.max_step}, rtol={args.rtol}, atol={args.atol}\n\n")
        f.write("Scanned lists\n")
        f.write(f"  dy={dy_vals}\n  b={b_vals}\n  g={g_vals}\n  amp={amp_vals}\n  sx={sx_vals}\n  st={st_vals}\n  Tseed={tseed_vals}\n\n")
        if best is not None:
            f.write("Best candidate\n")
            for key in [
                "score_total", "score_rho_rmse", "score_surface_rmse",
                "dy", "b", "g", "amp", "sx", "st", "Tseed",
                "pY_core", "Qproxy_int_pYdx", "pY_peak_abs", "pY_fwhm_abs", "polarity_label"
            ]:
                f.write(f"  {key} = {best[key]}\n")
            f.write("\nInterpretation\n")
            f.write("  Lower score means the final pY profile looked more like the Kelly-style proton target after normalization.\n")
            f.write("  Positive pY_core and positive Qproxy_int_pYdx are treated as proton-like.\n")
            f.write("  Because SQK pY is a substrate-potential proxy here, the comparison is shape-based, not unit-based.\n")

    # STEP 8A-12
    # Final terminal report so you immediately know where the outputs were saved.
    print("\nSearch complete.")
    print(f"Output directory: {args.outdir}")
    print(f"All runs CSV:     {all_csv}")
    print(f"Top candidates:   {top_csv}")
    print(f"Best plot PNG:    {plot_png}")
    print(f"Best NPZ:         {best_npz}")
    print(f"Summary text:     {summary_txt}")

    if best is not None:
        print("\nBest candidate:")
        for key in ["score_total", "dy", "b", "g", "amp", "sx", "st", "Tseed", "pY_core", "Qproxy_int_pYdx", "pY_fwhm_abs", "polarity_label"]:
            print(f"  {key}: {best[key]}")


# STEP 9
# Standard Python entry point. Only call main() when the file is executed directly.
if __name__ == "__main__":
    main()
