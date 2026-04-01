"""Empty docstring."""

import os
import pandas as pd
import math
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel
from datetime import datetime

from .. import thermal_fist_binding

NAME = "low_T_table"


# ---- analysis-specific config ----
class Config(BaseModel):
    """Analysis configuration. Overriden by user YAML config."""

    # grid
    Tmin: float = 0.000
    Tmax: float = 0.000
    dT: float = 0.001
    muBmin: float = -0.800
    muBmax: float = 0.800
    dmuB: float = 0.005
    muQmin: float = -0.800
    muQmax: float = 0.800
    dmuQ: float = 0.005
    muSmin: float = -0.800
    muSmax: float = 0.800
    dmuS: float = 0.005

    # model
    model_type: int = 0  # 0: Id-HRG, 1: EV Bag, 2: EV TwoComponent, 3: QvdW

    # IO
    particle_list: Path
    decays_list: Path
    outdir: Path = Path("./output")


# ---- worker globals (set by init_worker) ----
TF = None
BASE_TPS = None
CFG_T = None  # (config, Tmin, dT, muBmin, dmuB, muSmin, dmuS,muQmin, dmuQ)


def make_work(cfg: Config) -> List[Tuple[int, int, int, int]]:
    """Calculate the number of steps and return a list of (iT, iMuB, iMuQ, iMuS) tuples."""
    nT = int((cfg.Tmax - cfg.Tmin) / cfg.dT + 0.5) + 1
    nMuB = int((cfg.muBmax - cfg.muBmin) / cfg.dmuB + 0.5) + 1
    nMuQ = int((cfg.muQmax - cfg.muQmin) / cfg.dmuQ + 0.5) + 1
    nMuS = int((cfg.muSmax - cfg.muSmin) / cfg.dmuS + 0.5) + 1
    return [
        (iT, iMuB, iMuQ, iMuS)
        for iT in range(nT)
        for iMuB in range(nMuB)
        for iMuQ in range(nMuQ)
        for iMuS in range(nMuS)
    ]


def init_worker(tf_inc_build: str, tf_inc_src: str, tf_lib: str, cfg: Config) -> None:
    """Initialize Thermal-FIST binding for the initial worker process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("OMP_DYNAMIC", "false")

    thermal_fist_binding.add_paths(Path(tf_inc_build), Path(tf_inc_src), Path(tf_lib))
    thermal_fist_binding.load_tf(Path(tf_lib))
    thermal_fist_binding.include_tf_headers()

    global TF, BASE_TPS, CFG_T
    TF = thermal_fist_binding.ns()

    BASE_TPS = TF.ThermalParticleSystem(
        str(cfg.particle_list), str(cfg.decays_list), True, -1.0
    )

    CFG_T = (
        int(cfg.model_type),
        float(cfg.Tmin),
        float(cfg.dT),
        float(cfg.muBmin),
        float(cfg.dmuB),
        float(cfg.muQmin),
        float(cfg.dmuQ),
        float(cfg.muSmin),
        float(cfg.dmuS),
    )


def run_task(item: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, str]:
    """Run a single (iT, iMuB, iMuQ, iMuS) task and return (iT, iMuB, iMuQ, iMuS, output_line)."""
    iT, iMuB, iMuQ, iMuS = item
    config, Tmin, dT, muBmin, dmuB, muQmin, dmuQ, muSmin, dmuS = CFG_T
    TF_local, TPS = TF, BASE_TPS

    # ========================================
    #               BUILD MODEL
    # ========================================
    model = TF_local.ThermalModelIdeal(TPS)

    # Set statistics treatment
    # True: quantum statistics
    # False: Boltzmann statistics
    model.SetStatistics(True)

    # Set resonance width treatment
    model.SetUseWidth(TF_local.ThermalParticle.ZeroWidth)

    # ========================================
    #       SET THERMODYNAMIC VARIABLES
    # ========================================
    T = Tmin + iT * dT
    muB = muBmin + iMuB * dmuB
    muQ = muQmin + iMuQ * dmuQ
    muS = muSmin + iMuS * dmuS

    # Temperature
    model.SetTemperature(T)

    # Baryon chemical potential
    model.SetBaryonChemicalPotential(muB)

    # Strangeness chemical potential
    model.ConstrainMuS(False)
    model.SetStrangenessChemicalPotential(muS)

    # Electric chemical potential
    model.ConstrainMuQ(False)
    model.SetElectricChemicalPotential(muQ)

    # Perform calculations
    model.CalculatePrimordialDensities()
    model.CalculateFluctuations()
    model.CalculateTemperatureDerivatives()

    # Compute thermodynamic quantities
    p = model.Pressure()  # in GeV/fm^3
    s = model.EntropyDensity()  # in 1/fm^3
    nB = model.BaryonDensity()  # in 1/fm^3
    nS = model.StrangenessDensity()  # in 1/fm^3
    nQ = model.ElectricChargeDensity()  # in 1/fm^3
    e = model.EnergyDensity()  # in GeV/fm^3
    cs2 = model.cs2()  # speed of sound squared (dimensionless)
    chiBB = model.Susc(
        TF_local.ConservedCharge.BaryonCharge, TF_local.ConservedCharge.BaryonCharge
    )  # Real susceptibility, ∂(p/T^4)/∂(µB/T)^2 (dimensionless)
    chiQQ = model.Susc(
        TF_local.ConservedCharge.ElectricCharge,
        TF_local.ConservedCharge.ElectricCharge,
    )  # Real susceptibility, ∂(p/T^4)/∂(µQ/T)^2 (dimensionless)
    chiSS = model.Susc(
        TF_local.ConservedCharge.StrangenessCharge,
        TF_local.ConservedCharge.StrangenessCharge,
    )  # Real susceptibility, ∂(p/T^4)/∂(µS/T)^2 (dimensionless)
    chiBQ = model.Susc(
        TF_local.ConservedCharge.BaryonCharge,
        TF_local.ConservedCharge.ElectricCharge,
    )  # Real susceptibility, ∂(p/T^4)/∂(µB/T)∂(µQ/T) (dimensionless)
    chiBS = model.Susc(
        TF_local.ConservedCharge.BaryonCharge,
        TF_local.ConservedCharge.StrangenessCharge,
    )  # Real susceptibility, ∂(p/T^4)/∂(µB/T)∂(µS/T) (dimensionless)
    chiQS = model.Susc(
        TF_local.ConservedCharge.ElectricCharge,
        TF_local.ConservedCharge.StrangenessCharge,
    )  # Real susceptibility, ∂(p/T^4)/∂(µQ/T)∂(µS/T) (dimensionless)
    chiTB = model.ConservedChargeDensitydT(
        TF_local.ConservedCharge.BaryonCharge
    )  # Temperature derivative of baryon density at constant, ∂(nB)/∂T in 1/(fm^3 GeV)
    chiTQ = model.ConservedChargeDensitydT(
        TF_local.ConservedCharge.ElectricCharge
    )  # Temperature derivative of electric charge density at constant, ∂(nQ)/∂T in 1/(fm^3 GeV)
    chiTS = model.ConservedChargeDensitydT(
        TF_local.ConservedCharge.StrangenessCharge
    )  # Temperature derivative of strangeness density at constant, ∂(nS)/∂T in 1/(fm^3 GeV)
    chiTT = (
        model.SpecificHeatChem()
    ) / T  # Specific heat at constant chemical potentials, ∂^2p/∂T^2 = (∂s/∂T)_\mu = c_V/T in 1/(fm^3 GeV)

    # Normalize dimensionfull quantities
    pT4 = (p / TF_local.xMath.GeVtoifm3()) / (T**4)
    sT3 = (s / TF_local.xMath.GeVtoifm3()) / (T**3)
    nBT3 = (nB / TF_local.xMath.GeVtoifm3()) / (T**3)
    nQT3 = (nQ / TF_local.xMath.GeVtoifm3()) / (T**3)
    nST3 = (nS / TF_local.xMath.GeVtoifm3()) / (T**3)
    eT4 = (e / TF_local.xMath.GeVtoifm3()) / (T**4)
    chiTBT2 = (chiTB / TF_local.xMath.GeVtoifm3()) / (T**2)
    chiTQT2 = (chiTQ / TF_local.xMath.GeVtoifm3()) / (T**2)
    chiTST2 = (chiTS / TF_local.xMath.GeVtoifm3()) / (T**2)
    chiTTT2 = (chiTT / TF_local.xMath.GeVtoifm3()) / (T**2)

    sep = " "
    line = sep.join(
        [
            f"{round(T*1000):d}",
            f"{round(muB*1000):d}",
            f"{round(muQ*1000):d}",
            f"{round(muS*1000):d}",
            f"{pT4:.8e}",
            f"{sT3:.8e}",
            f"{nBT3:.8e}",
            f"{nST3:.8e}",
            f"{nQT3:.8e}",
            f"{eT4:.8e}",
            f"{cs2:.8e}",
            f"{chiBB:.8e}",
            f"{chiQQ:.8e}",
            f"{chiSS:.8e}",
            f"{chiBQ:.8e}",
            f"{chiBS:.8e}",
            f"{chiQS:.8e}",
            f"{chiTBT2:.8e}",
            f"{chiTQT2:.8e}",
            f"{chiTST2:.8e}",
            f"{chiTTT2:.8e}",
        ]
    )
    return (iT, iMuB, iMuQ, iMuS, line)


def postprocess(results, cfg: Config) -> Path:
    """Use pandas to write results to output file and return its path."""
    rows = []
    for _, _, _, _, line in sorted(results):
        if line is not None:
            rows.append(line)
    df = pd.DataFrame(
        [r.split() for r in rows],
        columns=[
            "T",
            "µB",
            "µQ",
            "µS",
            "p/T^4",
            "s/T^3",
            "nB/T^3",
            "nS/T^3",
            "nQ/T^3",
            "ε/T^4",
            "cs2",
            "chiBB/T^2",
            "chiQQ/T^2",
            "chiSS/T^2",
            "chiBQ/T^2",
            "chiBS/T^2",
            "chiQS/T^2",
            "chiTB/T^2",
            "chiTQ/T^2",
            "chiTS/T^2",
            "chiTT/T^2",
        ],
    )
    outdir = cfg.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = outdir / f"ANALYSIS={NAME}_MODELTYPE={cfg.model_type}_TIMESTAMP={ts}.csv"
    df.to_csv(out_name, index=False)

    return out_name
