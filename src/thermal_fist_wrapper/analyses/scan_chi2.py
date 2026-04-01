"""Empty docstring."""

import os
from math import pi
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel
from datetime import datetime

from .. import thermal_fist_binding

NAME = "scan_chi2"


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
CFG_T = None  # (config, Tmin, dT, muBmin, dmuB)


def make_work(cfg: Config) -> List[Tuple[int, int]]:
    """Calculate the number of steps and return a list of (iT, iMuB) tuples."""
    nT = int((cfg.Tmax - cfg.Tmin) / cfg.dT + 0.5) + 1
    nMuB = int((cfg.muBmax - cfg.muBmin) / cfg.dmuB + 0.5) + 1
    return [(iT, iMuB) for iT in range(nT) for iMuB in range(nMuB)]


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
    )


def run_task(item: Tuple[int, int]) -> Tuple[int, int, int, str]:
    """Run a single (iT, iMuB) task and return (iT, iMuB, output_line)."""
    iT, iMuB = item
    config, Tmin, dT, muBmin, dmuB = CFG_T
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
    muB = muBmin + iMuB * dmuB + 1e-6

    model.SetBaryonChemicalPotential(muB)
    model.ConstrainMuS(True)
    # model.SetStrangenessChemicalPotential(0.0)
    model.ConstrainMuQ(True)
    # Choose one of the following if ConstrainMuQ(True)
    # model.SetElectricChemicalPotential(0.0)
    model.SetQoverB(0.4)

    fitter = TF_local.ThermalModelFit(model)
    fitter.SetParameterFitFlag("muB", False)
    fitter.SetParameterFitFlag("muS", False)
    fitter.SetParameterFitFlag("muQ", False)
    # fitter.SetParameter("muQ", -0.0005, 0.000001, -0.001, 0.001)
    fitter.SetParameterFitFlag("R", False)
    fitter.SetParameterValue("R", 10.7)
    # Uncomment to fit radius
    # fitter.SetParameterFitFlag("R", True)
    # fitter.SetParameter("R", 0.01, 0.01, 0.0, 15.0)
    fitter.SetQuantities(QUANT)
    fitter.SetParameterFitFlag("T", False)
    fitter.SetParameterValue("T", T)

    res = fitter.PerformFit(False)

    line = (
        f"{T*1000:15.6f}"
        f"{res.muB.value*1000:15.6f}"
        f"{res.muS.value*1000:15.6f}"
        f"{res.muQ.value*1000:15.6f}"
        f"{res.muQ.error*1000:15.6f}"
        f"{(4./3)*pi*res.R.value**3:15.6f}"
        f"{model.CalculateChargeDensity()/model.CalculateBaryonDensity():15.6f}"
        f"{res.chi2:15.6f}"
        f"{res.chi2/(res.ndf-1):15.6f}"
    )
    return (iT, iMuB, line)


def postprocess(results, cfg: Config) -> Path:
    """Write results to output file and return its path."""
    rows = [
        "".join(
            f"{h:>15s}"
            for h in [
                "T[MeV]",
                "muB[MeV]",
                "muS[MeV]",
                "muQ[MeV]",
                "muQ_err[MeV]",
                "V[fm^3]",
                "Q/B",
                "chi2",
                "chi2/N_dof",
            ]
        )
    ]
    for _, _, line in sorted(results):
        rows.append(line)

    outdir = cfg.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_stem = Path(cfg.data_file).stem
    out_name = (
        outdir
        / f"ANALYSIS={NAME}_MODELTYPE={cfg.model_type}_DATA={data_stem}_TIMESTAMP={ts}.dat"
    )

    out_name.write_text("\n".join(rows) + "\n")
    return out_name
