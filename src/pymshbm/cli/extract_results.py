"""CLI entrypoint for extracting individual parcellations.

Plug-in replacement for CBIG_IndCBM_extract_MSHBM_result_SUB.m.

Usage:
    pymshbm-extract-results PROJECT_DIR SUB1 SUB2 ...
"""

import argparse
import sys

from pymshbm.core.extract_results import extract_mshbm_result_sub


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract individual parcellations from MSHBM group priors.",
    )
    parser.add_argument(
        "project_dir",
        help="Project directory containing priors/Params_Final.mat.",
    )
    parser.add_argument(
        "--priors_file",
        help="Optional path to a custom priors file. If not provided, defaults to project_dir/priors/Params_Final.mat.",
    )
    parser.add_argument(
        "subject_ids",
        nargs="+",
        help="Subject ID strings (one per subject in s_lambda).",
    )

    
    args = parser.parse_args(argv)

    try:
        results = extract_mshbm_result_sub(args.project_dir, args.subject_ids, args.priors_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    for i, (lh, rh) in enumerate(results):
        n_nonzero = int((lh > 0).sum() + (rh > 0).sum())
        print(f"Subject {args.subject_ids[i]}: {n_nonzero} labeled vertices")


if __name__ == "__main__":
    main()
