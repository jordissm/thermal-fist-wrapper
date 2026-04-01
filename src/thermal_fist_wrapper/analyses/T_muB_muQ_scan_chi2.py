"""Empty docstring."""

import os
import pandas as pd
from math import pi
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel
from datetime import datetime

from .. import thermal_fist_binding

NAME = "T_muB_muQ_scan_chi2"


# ---- analysis-specific config ----
class Config(BaseModel):
    """Analysis configuration. Overriden by user YAML config."""

    # grid
    Tmin: float = 0.1350
    Tmax: float = 0.1601
    dT: float = 0.01
    muBmin: float = 0.0
    muBmax: float = 0.05
    dmuB: float = 0.05
    muQmin: float = -0.02
    muQmax: float = 0.02
    dmuQ: float = 0.05

    # model
    model_type: int = 0  # 0: Id-HRG, 1: EV Bag, 2: EV TwoComponent, 3: QvdW

    # IO
    particle_list: Path
    decays_list: Path
    data_file: Path
    outdir: Path = Path("./output")


# ---- worker globals (set by init_worker) ----
TF = None
QUANT = None
BASE_TPS = None
CFG_T = None  # (config, Tmin, dT, muBmin, dmuB, muQmin, dmuQ)


def make_work(cfg: Config) -> List[Tuple[int, int, int]]:
    """Calculate the number of steps and return a list of (iT, iMuB, iMuQ) tuples."""
    nT = int((cfg.Tmax - cfg.Tmin) / cfg.dT + 0.5) + 1
    nMuB = int((cfg.muBmax - cfg.muBmin) / cfg.dmuB + 0.5) + 1
    nMuQ = int((cfg.muQmax - cfg.muQmin) / cfg.dmuQ + 0.5) + 1
    return [
        (iT, iMuB, iMuQ)
        for iT in range(nT)
        for iMuB in range(nMuB)
        for iMuQ in range(nMuQ)
    ]


def init_worker(tf_inc_build: str, tf_inc_src: str, tf_lib: str, cfg: Config) -> None:
    """Initialize Thermal-FIST binding for the initial worker process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("OMP_DYNAMIC", "false")

    thermal_fist_binding.add_paths(Path(tf_inc_build), Path(tf_inc_src), Path(tf_lib))
    thermal_fist_binding.load_tf(Path(tf_lib))
    thermal_fist_binding.include_tf_headers()

    global TF, QUANT, BASE_TPS, CFG_T
    TF = thermal_fist_binding.ns()
    QUANT = TF.ThermalModelFit.loadExpDataFromFile(str(cfg.data_file))
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
    )


def run_task(item: Tuple[int, int, int]) -> Tuple[int, int, int, int, str]:
    """Run a single (iT, iMuB, iMuQ) task and return (iT, iMuB, iMuQ, output_line)."""
    iT, iMuB, iMuQ = item
    config, Tmin, dT, muBmin, dmuB, muQmin, dmuQ = CFG_T
    TF_local, TPS = TF, BASE_TPS

    # build model
    if config == 1:
        model = TF_local.ThermalModelEVDiagonal(TPS)
        radProton = 0.5
        tps = model.TPS()
        N = tps.ComponentsNumber()
        for i in range(N):
            mRatio = tps.Particle(i).Mass() / 0.938
            model.SetRadius(i, radProton * (mRatio ** (1.0 / 3.0)))
    elif config == 2:
        model = TF_local.ThermalModelEVDiagonal(TPS)
        rad = 0.3
        tps = model.TPS()
        N = tps.ComponentsNumber()
        for i in range(N):
            model.SetRadius(i, rad if tps.Particle(i).BaryonCharge() != 0 else 0.0)
    elif config == 3:
        model = TF_local.ThermalModelVDWFull(TPS)
        a, b = 0.329, 3.42
        tps = model.TPS()
        N = tps.ComponentsNumber()
        baryons, antibaryons = [], []
        for i in range(N):
            B = tps.Particle(i).BaryonCharge()
            (baryons if B > 0 else antibaryons if B < 0 else []).append(i)
        for i in range(N):
            for j in range(N):
                model.SetAttraction(i, j, 0.0)
                model.SetVirial(i, j, 0.0)
        for i in baryons:
            for j in baryons:
                model.SetAttraction(i, j, a)
                model.SetVirial(i, j, b)
        for i in antibaryons:
            for j in antibaryons:
                model.SetAttraction(i, j, a)
                model.SetVirial(i, j, b)
    else:
        model = TF_local.ThermalModelIdeal(TPS)

    model.SetStatistics(True)
    model.SetUseWidth(TF_local.ThermalParticle.ZeroWidth)
    # model.SetUseWidth(TF_local.ThermalParticle.eBW)

    T = Tmin + iT * dT
    muB = muBmin + iMuB * dmuB + 1e-8
    muQ = muQmin + iMuQ * dmuQ

    fitTemperature = False
    fitRadius = False

    # ========================================
    #   SET CHEMICAL POTENTIAL CONSTRAINTS
    # ========================================

    # Baryon chemical potential
    model.SetBaryonChemicalPotential(muB)

    # Strangeness chemical potential
    model.ConstrainMuS(True)

    # Electric chemical potential
    model.ConstrainMuQ(False)
    model.SetElectricChemicalPotential(muQ)

    # ========================================
    #           SET FIT PARAMETERS
    # ========================================

    fitter = TF_local.ThermalModelFit(model)

    # Set experimental data
    fitter.SetQuantities(QUANT)

    # ----- Fitting T logic -----
    if not fitTemperature:
        fitter.SetParameterFitFlag("T", False)
        fitter.SetParameterValue("T", T)
    else:
        fitter.SetParameterFitFlag("T", True)
        fitter.SetParameter("T", 0.01, 0.01, 0.0, 0.2)

    # ----- Fitting R logic -----
    if not fitRadius:
        fitter.SetParameterFitFlag("R", False)
        fitter.SetParameterValue("R", 10.7)
    else:
        fitter.SetParameterFitFlag("R", True)
        fitter.SetParameter("R", 0.01, 0.01, 0.0, 15.0)

    # ----- Fitting muB logic -----
    model.SetBaryonChemicalPotential(muB)
    fitter.SetParameterFitFlag("muB", False)

    # ----- Fitting muS logic -----
    # If fitStrangenessChemicalPotential is True, muS is fitted.
    fitter.SetParameterFitFlag("muS", False)

    # ----- Fitting muQ logic -----
    # If fitElectricChemicalPotential is True, muQ is fitted.
    fitter.SetParameterFitFlag("muQ", False)

    # Perform fit.
    # If PerformFit(False), output to terminal is suppressed but results are still returned.
    res = fitter.PerformFit(False)

    # If energy density is within the desired range, save the result.
    # energyDensityMin = 0.258_442  # GeV/fm^3
    # energyDensityMax = 0.290_340  # GeV/fm^3
    # if not (energyDensityMin <= model.CalculateEnergyDensity() <= energyDensityMax):
    # return (iT, iMuB, iMuQ, None)

    # If chi2/N_dof is within the desired range, save the result.
    chi2ndofMin = 0.0
    chi2ndofMax = 3.0
    if not (chi2ndofMin <= res.chi2 / (res.ndf - 1) <= chi2ndofMax):
        return (iT, iMuB, iMuQ, None)

    line = (
        f"{T*1000:15.10f}"
        f"{res.muB.value*1000:15.8f}"
        f"{res.muS.value*1000:15.8f}"
        f"{res.muQ.value*1000:15.8f}"
        # f"{(4./3)*pi*res.R.value**3:15.6f}"
        f"{model.CalculateBaryonDensity():15.6e}"
        f"{model.CalculateStrangenessDensity():15.6e}"
        f"{model.CalculateChargeDensity():15.6e}"
        f"{model.CalculateChargeDensity()/model.CalculateBaryonDensity():15.6e}"
        f"{model.CalculateEnergyDensity()*1000:15.6e}"
        f"{model.CalculateEntropyDensity():15.6e}"
        f"{model.CalculateEntropyDensity()/model.CalculateBaryonDensity():15.6e}"
        f"{res.chi2:15.6f}"
        f"{res.chi2/(res.ndf-1):15.6f}"
    )
    return (iT, iMuB, iMuQ, line)


def postprocess(results, cfg: Config) -> Path:
    """Use pandas to write results to output file and return its path."""
    rows = []
    for _, _, _, line in sorted(results):
        if line is not None:
            rows.append(line)
    df = pd.DataFrame(
        [r.split() for r in rows],
        columns=[
            "T[MeV]",
            "muB[MeV]",
            "muS[MeV]",
            "muQ[MeV]",
            "nB[1/fm^3]",
            "nS[1/fm^3]",
            "nQ[1/fm^3]",
            "Q/B",
            "e[MeV/fm^3]",
            "s[1/fm^3]",
            "s/nB",
            "chi2",
            "chi2/N_dof",
        ],
    )
    outdir = cfg.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_stem = Path(cfg.data_file).stem
    out_name = (
        outdir
        / f"ANALYSIS={NAME}_MODELTYPE={cfg.model_type}_DATA={data_stem}_TIMESTAMP={ts}.csv"
    )
    df.to_csv(out_name, index=False)

    # """Write results to output file and return its path."""
    # rows = [
    #     "".join(
    #         f"{h:>15s}"
    #         for h in [
    #             "T[MeV]",
    #             "muB[MeV]",
    #             "muS[MeV]",
    #             "muQ[MeV]",
    #             "nB[1/fm^3]",
    #             "nS[1/fm^3]",
    #             "nQ[1/fm^3]",
    #             "Q/B",
    #             "e[MeV/fm^3]",
    #             "s[1/fm^3]",
    #             "s/nB",
    #             "chi2",
    #             "chi2/N_dof",
    #         ]
    #     )
    # ]
    # for _, _, _, line in sorted(results):
    #     if line is not None:
    #         rows.append(line)

    # outdir = cfg.outdir.resolve()
    # outdir.mkdir(parents=True, exist_ok=True)

    # ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # data_stem = Path(cfg.data_file).stem
    # out_name = (
    #     outdir
    #     / f"ANALYSIS={NAME}_MODELTYPE={cfg.model_type}_DATA={data_stem}_TIMESTAMP={ts}.csv"
    # )

    # out_name.write_text("\n".join(rows) + "\n")
    return out_name
