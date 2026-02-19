"""CLI entrypoint for the MSHBM wrapper pipeline.

Usage:
    pymshbm-wrapper SUB_LIST OUTPUT_DIR [--seed-labels-lh PATH --seed-labels-rh PATH]
    pymshbm-wrapper SUB_LIST OUTPUT_DIR [--freesurfer-dir DIR]
"""

import argparse
import logging
import sys

import numpy as np

from pymshbm.pipeline.wrapper import run_wrapper


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate FC profiles and prepare training data.",
    )
    parser.add_argument("sub_list", help="CSV file with subject_id, data_dir columns.")
    parser.add_argument("output_dir", help="Base output directory.")
    parser.add_argument("--seed-labels-lh", default=None,
                        help="Path to .npy file with lh seed labels.")
    parser.add_argument("--seed-labels-rh", default=None,
                        help="Path to .npy file with rh seed labels.")
    parser.add_argument("--freesurfer-dir", default=None,
                        help="FreeSurfer subjects directory for auto-computing seed labels.")
    parser.add_argument("--file-pattern",
                        default="*nat_resid_bpss_fsaverage6_sm*.nii.gz",
                        help="Glob pattern for fMRI files.")
    parser.add_argument("--seed-mesh", default="fsaverage3",
                        help="Seed mesh identifier.")
    parser.add_argument("--targ-mesh", default="fsaverage6",
                        help="Target mesh identifier.")
    parser.add_argument("--num-clusters", type=int, default=None,
                        help="Number of parcellation clusters. Enables training + CIFTI output.")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Maximum EM iterations for group prior estimation.")
    parser.add_argument("--mrf-weight", type=float, default=50.0,
                        help="MRF smoothness weight for individual parcellation (default: 50).")
    parser.add_argument("--spatial-weight", type=float, default=200.0,
                        help="Spatial prior weight for individual parcellation (default: 200).")
    parser.add_argument("--overwrite-fc", action="store_true", default=False,
                        help="Overwrite existing FC profile files instead of reusing them.")
    parser.add_argument("--overwrite-kmeans", action="store_true", default=False,
                        help="Recompute initial k-means centroids instead of reusing cached.")

    args = parser.parse_args(argv)

    seed_labels_lh = None
    seed_labels_rh = None

    if args.seed_labels_lh and args.seed_labels_rh:
        try:
            seed_labels_lh = np.load(args.seed_labels_lh)
            seed_labels_rh = np.load(args.seed_labels_rh)
        except Exception as exc:
            print(f"Error loading seed labels: {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.seed_labels_lh or args.seed_labels_rh:
        print("Error: --seed-labels-lh and --seed-labels-rh must both be "
              "provided, or both omitted.", file=sys.stderr)
        sys.exit(1)

    try:
        result_dir = run_wrapper(
            sub_list=args.sub_list,
            output_dir=args.output_dir,
            seed_labels_lh=seed_labels_lh,
            seed_labels_rh=seed_labels_rh,
            file_pattern=args.file_pattern,
            seed_mesh=args.seed_mesh,
            targ_mesh=args.targ_mesh,
            freesurfer_dir=args.freesurfer_dir,
            num_clusters=args.num_clusters,
            max_iter=args.max_iter,
            mrf_weight=args.mrf_weight,
            spatial_weight=args.spatial_weight,
            overwrite_fc=args.overwrite_fc,
            overwrite_kmeans=args.overwrite_kmeans,
        )
        print(f"Wrapper complete. Output: {result_dir}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
