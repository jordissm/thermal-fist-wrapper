"""Command-line interface for thermal-fist-wrapper."""
import argparse
import time
from .config import load_config
from .pool import run_pool
from .analyses import list_analyses, get_analysis

from .logger import get_logger, get_console, setup_logging



def build_parser() -> argparse.ArgumentParser:
    """
    Command-line interface.
    """
    # Main parser
    parser = argparse.ArgumentParser(prog="thermal-fist-wrapper",
                                     description="Analysis wrapper for Thermal-FIST")

    # Subparsers
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Create the parsers for the "run" command
    parser_run = subparsers.add_parser("run", help="Run an analysis from a config file")
    parser_run.add_argument("--config", required=True, help="YAML config")
    parser_run.add_argument("--log-level", default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            help="Set the logging level")

    # Create the parser for the "list" command
    subparsers.add_parser("list", help="List available analyses")

    # Create the parsers for the "template" command
    parser_tmpl = subparsers.add_parser("template", help="Print a template config for an analysis")
    parser_tmpl.add_argument("analysis", help="Analysis name")


    return parser

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = build_parser()

    return parser.parse_args(argv)

