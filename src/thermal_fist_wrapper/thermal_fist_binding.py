"""Bindings for Thermal-FIST using cppyy."""

import os
import cppyy
from pathlib import Path

def add_paths(tf_inc_build: Path, tf_inc_src: Path, tf_lib: Path) -> None:
    cppyy.add_include_path(str(tf_inc_build))
    cppyy.add_include_path(str(tf_inc_src))
    cppyy.add_library_path(str(tf_lib))

def load_tf(tf_lib: Path) -> None:
    cppyy.load_library(str(tf_lib / "libThermalFIST.dylib"))
    try:
        cppyy.load_library("libMinuit2")
    except Exception:
        pass

def include_tf_headers() -> None:
    for h in ("ThermalFISTConfig.h", 
              "HRGBase.h", 
              "HRGEV.h", 
              "HRGFit.h",
              "HRGPCE.h",
              "HRGRealGas.h",
              "HRGVDW.h"):
        cppyy.include(h)

def ns():
    g = cppyy.gbl
    return getattr(g, "thermalfist", g)

def set_parent_openmp_defaults(threads: int | None = None) -> None:
    if threads is None:
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ.setdefault("OMP_DYNAMIC", "false")
    os.environ.setdefault("OMP_PROC_BIND", "true")
    os.environ.setdefault("OMP_PLACES", "cores")
