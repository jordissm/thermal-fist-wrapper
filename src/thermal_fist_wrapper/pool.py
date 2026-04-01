from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
)
import os

from .logger import get_logger

logger = get_logger(__name__)

def run_pool(
    work: list[Any],
    init_fn: Callable[..., None],
    init_args: tuple[Any, ...],
    task_fn: Callable[[Any], Any],
    workers: int | None = None,
    chunksize: int | None = None,
):
    """Generic process-pool runner (rich-based progress).

    - If workers==1, runs serially (still calls init_fn for parity with parallel case).
    - If unordered_log=True and show_pids=True, logs completions (assumes result like (..., pid, ...)).
    - Returns list of task results (unordered); caller can sort if needed.
    """
    total = len(work)
    max_workers = workers or (os.cpu_count() or 1)

    progress_cols = (
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    if max_workers == 1:
        init_fn(*init_args)
        out = []
        with Progress(*progress_cols) as prog:
            task = prog.add_task("running", total=total)
            for item in work:
                res = task_fn(item)
                out.append(res)
                prog.advance(task)
        return out

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_fn, initargs=init_args) as ex:
        # Ordered results; good when tasks are uniform. We still show live progress.
        mapped = ex.map(task_fn, work, chunksize=1 if chunksize is None else chunksize)
        out = []
        with Progress(*progress_cols) as prog:
            task = prog.add_task("running", total=total)
            for res in mapped:
                out.append(res)
                prog.advance(task)
        return out