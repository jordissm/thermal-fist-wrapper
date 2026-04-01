"""Main entry point for thermal-fist-wrapper."""

import time
from .cli import parse_args
from .config import load_config
from .pool import run_pool
from .analyses import list_analyses, get_analysis
from .logger import get_logger, get_console, setup_logging

def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    setup_logging(level=args.log_level)
    log = get_logger(__name__)
    console = get_console()


    if args.cmd == "list":
        console.print("\n".join(list_analyses()))
        return 0

    elif args.cmd == "template":
        name = args.analysis
        mod = get_analysis(name)
        console.print(f"""# {name}.yaml
        analysis: {name}
        paths:
        tf_root: /Applications/Thermal-FIST
        params:
        # Fill per-analysis params below (validated by its Config model)
        """)
        return 0

    elif args.cmd == "run":
        # run
        cfg_raw = load_config(args.config)
        mod = get_analysis(cfg_raw.analysis)

        # validate analysis params
        template_cfg = getattr(mod, "Config")
        working_cfg = template_cfg.model_validate(cfg_raw.params)

        # resolve TF paths
        tfp = cfg_raw.paths.resolved()
        tf_inc_build = (tfp.tf_root / "build-shared" / "include").resolve()

        # build work
        work = mod.make_work(working_cfg)
        t0 = time.perf_counter()

        # run pool
        results = run_pool(
            work=work,
            init_fn=mod.init_worker,
            init_args=(str(tf_inc_build), str(tfp.tf_inc), str(tfp.tf_lib), working_cfg),
            task_fn=mod.run_task,
            workers=cfg_raw.runtime.workers,
            chunksize=None,
        )

        out_path = mod.postprocess(results, working_cfg)
        dt = time.perf_counter() - t0
        log.info(f"Results written to {out_path}")
        log.debug(f"Total time: {dt:.3f}s  |  Avg/task: {dt/len(work):.6f}s")
        return 0

    else:
        # log.error("Unknown command: %s", args.cmd)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
