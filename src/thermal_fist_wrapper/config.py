from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel

from .logger import get_logger

log = get_logger(__name__)

class TFPaths(BaseModel):
    tf_root: Path = Path("/Applications/Thermal-FIST")
    tf_inc: Optional[Path] = None                 # defaults to <tf_root>/include
    tf_lib: Optional[Path] = None                 # defaults to <tf_root>/build-shared/lib

    def resolved(self) -> "TFPaths":
        """Resolve and validate Thermal-FIST paths. Raises on missing/invalid dirs."""
        def _resolve_dir(p: Path, name: str) -> Path:
            # Expand, resolve, and validate a directory path
            rp = p.expanduser().resolve()
            log.info("Resolving %s: %s", name, rp)
            if not rp.exists():
                log.error("%s does not exist: %s", name, rp)
                raise FileNotFoundError(f"{name} does not exist: {rp}")
            if not rp.is_dir():
                log.error("%s is not a directory: %s", name, rp)
                raise NotADirectoryError(f"{name} is not a directory: {rp}")
            return rp

        root = _resolve_dir(self.tf_root, "tf_root")

        # Default includes to <tf_root>/include
        inc_default = root / "include"
        inc = _resolve_dir(self.tf_inc or inc_default, "tf_inc")

        # Default libs to <tf_root>/build-shared/lib
        lib_default = root / "build-shared" / "lib"
        lib = _resolve_dir(self.tf_lib or lib_default, "tf_lib")

        log.info("Using Thermal-FIST paths -> root: %s | inc: %s | lib: %s", root, inc, lib)
        return TFPaths(tf_root=root, tf_inc=inc, tf_lib=lib)

class Runtime(BaseModel):
    workers: int | None = None        # default: cpu count

class ProjectConfig(BaseModel):
    analysis: str                      # e.g. "scan_chi2"
    paths: TFPaths
    params: dict                       # passed to the analysis' ConfigModel
    runtime: Runtime = Runtime()

def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project config from a YAML file."""
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))

    # Validate and return the config
    return ProjectConfig.model_validate(data)
