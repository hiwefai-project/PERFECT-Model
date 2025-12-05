\"\"\"Main entry point for PERFECT-M.

This script parses command-line arguments (notably the path to the
configuration JSON file), sets the appropriate environment variable,
and then calls the high-level workflow function.
\"\"\"

import argparse
import os

from perfectm.pipeline.workflow import run_full_pipeline


def parse_args() -> argparse.Namespace:
    \"\"\"Parse command-line arguments for the PERFECT-M runner.

    Returns
    -------
    args:
        Object containing parsed arguments as attributes.
    \"\"\"
    parser = argparse.ArgumentParser(description=\"Run the PERFECT-M hydrological model.\")

    # Option to specify a custom configuration JSON file.
    parser.add_argument(
        \"--config\",
        type=str,
        default=\"config.json\",
        help=\"Path to the configuration JSON file (default: config.json)\",
    )

    return parser.parse_args()


def main() -> None:
    \"\"\"Entry point called when running this script as a program.\"\"\"
    # Parse command-line arguments.
    args = parse_args()

    # Set environment variable so perfectm.config uses this configuration file.
    os.environ[\"PERFECTM_CONFIG\"] = args.config

    # Run the high-level workflow.
    run_full_pipeline()


if __name__ == \"__main__\":
    main()
