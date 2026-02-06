import multiprocessing as mp
import argparse


def main(run_baseline=True, run_docking=True, run_dta_atlas=True, num_cores=12, sequential=False) -> None:
    """
    Compute diffusion profiles for MSI networks.

    Args:
        run_baseline: Compute profiles for default MSI (no filtered edges)
        run_docking: Compute profiles for docking-filtered edge files
        run_dta_atlas: Compute profiles for DTA-Atlas filtered edge files
        num_cores: Number of CPU cores to use for parallel computation
    """
    from diff_prof import (
        compute_all_diffusion_profiles_for_msi,
        compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs,
    )

    print("="*80)
    print("DIFFUSION PROFILE COMPUTATION")
    print("="*80)
    print(f"\nPipelines to run:")
    print(f"  - Baseline MSI:  {'Yes' if run_baseline else 'No'}")
    print(f"  - Docking:       {'Yes' if run_docking else 'No'}")
    print(f"  - DTA-Atlas:     {'Yes' if run_dta_atlas else 'No'}")
    print(f"  - CPU cores:     {num_cores}")
    print(f"  - Sequential:    {'Yes (low memory)' if sequential else 'No (parallel)'}")

    if not run_baseline and not run_docking and not run_dta_atlas:
        print("\nError: At least one pipeline must be enabled!")
        return

    # ========================================================================
    # BASELINE: Default MSI (no filtered edges)
    # ========================================================================
    if run_baseline:
        print("\n" + "="*80)
        print("BASELINE: Default MSI diffusion profiles")
        print("="*80)
        _, _ = compute_all_diffusion_profiles_for_msi(
            save_load_file_path="results/",
            num_cores=num_cores,
        )
        print("Baseline profiles saved to: results/")

    # ========================================================================
    # DOCKING: Filtered edges from docking/ML scores
    # ========================================================================
    if run_docking:
        print("\n" + "="*80)
        print("DOCKING: Diffusion profiles for docking-filtered edges")
        print("="*80)
        runs_docking = compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs(
            save_root="results_filtered/docking/",
            drug2protein_glob="data/docking/1_drug_to_protein_*cutoff_*.tsv",
            recompute=True,
            on_error="skip",
            sequential=sequential,
            num_cores=num_cores,
        )
        print(f"\nDocking: Prepared {len(runs_docking)} diffusion-profile runs")
        print(f"Output: results_filtered/docking/")

    # ========================================================================
    # DTA-ATLAS: Filtered edges from DTA-Atlas predictions
    # ========================================================================
    if run_dta_atlas:
        print("\n" + "="*80)
        print("DTA-ATLAS: Diffusion profiles for DTA-Atlas filtered edges")
        print("="*80)
        runs_dta_atlas = compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs(
            save_root="results_filtered/dta_atlas/",
            drug2protein_glob="data/dta_atlas/1_drug_to_protein_*cutoff_*.tsv",
            recompute=True,
            on_error="skip",
            sequential=sequential,
            num_cores=num_cores,
        )
        print(f"\nDTA-Atlas: Prepared {len(runs_dta_atlas)} diffusion-profile runs")
        print(f"Output: results_filtered/dta_atlas/")

    print("\n" + "="*80)
    print("DIFFUSION PROFILE COMPUTATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # When the multiprocessing start method is 'spawn' (common on macOS/Windows,
    # sometimes in certain environments), Python re-imports this module in child
    # processes. Guarding main prevents recursive process spawning.
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        # Start method already set (or fork not available); safe to ignore.
        pass

    parser = argparse.ArgumentParser(
        description="Compute diffusion profiles for MSI networks with various filtered edge sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all pipelines (default)
  python 2_compute_diffusion_profiles.py

  # Run only DTA-Atlas pipeline
  python 2_compute_diffusion_profiles.py --dta-atlas-only

  # Run only docking pipeline
  python 2_compute_diffusion_profiles.py --docking-only

  # Run only baseline (no filtered edges)
  python 2_compute_diffusion_profiles.py --baseline-only

  # Skip baseline, run filtered edges only
  python 2_compute_diffusion_profiles.py --no-baseline

  # Use more CPU cores
  python 2_compute_diffusion_profiles.py --num-cores 24

  # Low memory mode (sequential processing, ~15GB vs ~60GB+)
  python 2_compute_diffusion_profiles.py --sequential

  # Combine options
  python 2_compute_diffusion_profiles.py --dta-atlas-only --sequential
        """
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline MSI (no filtered edges)"
    )

    group.add_argument(
        "--docking-only",
        action="store_true",
        help="Run only the docking pipeline (skip baseline and DTA-Atlas)"
    )

    group.add_argument(
        "--dta-atlas-only",
        action="store_true",
        help="Run only the DTA-Atlas pipeline (skip baseline and docking)"
    )

    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the baseline MSI computation"
    )

    parser.add_argument(
        "--num-cores",
        type=int,
        default=12,
        help="Number of CPU cores to use (default: 12)"
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process profiles sequentially (low memory mode, ~15GB vs ~60GB+)"
    )

    args = parser.parse_args()

    # Determine which pipelines to run
    if args.baseline_only:
        run_baseline = True
        run_docking = False
        run_dta_atlas = False
    elif args.docking_only:
        run_baseline = False
        run_docking = True
        run_dta_atlas = False
    elif args.dta_atlas_only:
        run_baseline = False
        run_docking = False
        run_dta_atlas = True
    else:
        # Default: run all
        run_baseline = not args.no_baseline
        run_docking = True
        run_dta_atlas = True

    main(
        run_baseline=run_baseline,
        run_docking=run_docking,
        run_dta_atlas=run_dta_atlas,
        num_cores=args.num_cores,
        sequential=args.sequential,
    )
