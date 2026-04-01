from __future__ import annotations
from typing import Dict, Type

# Import analyses here to register them
from . import scan_chi2
from . import T_muB_muQ_scan_chi2
from . import T_muB_muQ_w_lattice_scan_chi2
from . import low_T_table

REGISTRY: Dict[str, object] = {
    scan_chi2.NAME: scan_chi2,
    T_muB_muQ_scan_chi2.NAME: T_muB_muQ_scan_chi2,
    T_muB_muQ_w_lattice_scan_chi2.NAME: T_muB_muQ_w_lattice_scan_chi2,
    low_T_table.NAME: low_T_table,
}


def list_analyses() -> list[str]:
    return sorted(REGISTRY.keys())


def get_analysis(name: str):
    try:
        return REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"Unknown analysis '{name}'. Available: {', '.join(list_analyses())}"
        )
