#!/usr/bin/env python3
"""Compute diffusion profiles for MSI networks with various filtered edge sets.

This script supports:
- Custom input/output directories
- Interactive subset selection
- Remote SSH upload with automatic retry
- Resume mode (checks both local and remote)
- Comprehensive logging
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time


def main(
    # Input/output paths
    input_dir: str | None = None,
    output_dir: str | None = None,
    output_prefix: str = "results_filtered",
    data_dir: str = "data",
    # Pipeline selection (legacy mode)
    run_baseline: bool = True,
    run_docking: bool = True,
    run_dta_atlas: bool = True,
    # Computation options
    num_cores: int = 12,
    sequential: bool = False,
    recompute: bool = True,
    interactive: bool = False,
    # Remote sync options
    remote_host: str | None = None,
    remote_port: int = 22,
    remote_user: str | None = None,
    remote_path: str | None = None,
    ssh_key: str | None = None,
    delete_after_upload: bool = False,
) -> None:
    """
    Compute diffusion profiles for MSI networks.

    Args:
        input_dir: Custom input directory with drug_to_protein TSV files
        output_dir: Custom output directory for results
        output_prefix: Prefix for filtered output directories (legacy mode)
        data_dir: Base directory for MSI data files (indication_to_protein, etc.)
        run_baseline: Compute profiles for default MSI (no filtered edges)
        run_docking: Compute profiles for docking-filtered edge files
        run_dta_atlas: Compute profiles for DTA-Atlas filtered edge files
        num_cores: Number of CPU cores to use for parallel computation
        sequential: Process profiles one at a time (low memory mode)
        recompute: If False, skip already-completed runs (resume mode)
        interactive: Interactively select which configurations to compute
        remote_host: Remote SSH host for result backup
        remote_port: Remote SSH port
        remote_user: Remote SSH username
        remote_path: Remote directory path for results
        ssh_key: Path to SSH private key
        delete_after_upload: Delete local files after successful upload
    """
    from diff_prof import compute_all_diffusion_profiles_for_msi
    from diff_prof.batch import compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs
    from utils.logging_setup import setup_logging
    from utils.interactive import select_runs_interactively
    from utils.remote_sync import RemoteSync, UploadError

    start_time = time.time()

    # Determine output directory for logging
    if output_dir:
        log_output_dir = output_dir
    elif input_dir:
        # Custom input mode - use input_dir name as output subdirectory
        pipeline_name = os.path.basename(input_dir.rstrip("/"))
        log_output_dir = os.path.join(output_prefix, pipeline_name)
    else:
        # Legacy mode - use output_prefix
        log_output_dir = output_prefix

    os.makedirs(log_output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_output_dir)

    logger.info("=" * 80)
    logger.info("DIFFUSION PROFILE COMPUTATION")
    logger.info("=" * 80)
    logger.info(f"Command: {' '.join(sys.argv)}")

    # Construct MSI data file paths
    indication2protein_path = os.path.join(data_dir, "2_indication_to_protein.tsv")
    protein2protein_path = os.path.join(data_dir, "3_protein_to_protein.tsv")
    protein2bio_func_path = os.path.join(data_dir, "4_protein_to_biological_function.tsv")
    bio_func2bio_func_path = os.path.join(data_dir, "5_biological_function_to_biological_function.tsv")

    # Log configuration
    logger.info("\nConfiguration:")
    if input_dir:
        logger.info(f"  Input directory:  {input_dir}")
    logger.info(f"  Data directory:   {data_dir}")
    if output_dir:
        logger.info(f"  Output directory: {output_dir}")
    else:
        logger.info(f"  Output prefix:    {output_prefix}")
    logger.info(f"  CPU cores:        {num_cores}")
    logger.info(f"  Sequential:       {'Yes (low memory)' if sequential else 'No (parallel)'}")
    logger.info(f"  Resume mode:      {'Yes (skip complete)' if not recompute else 'No (recompute all)'}")
    logger.info(f"  Interactive:      {'Yes' if interactive else 'No'}")

    # Setup remote sync if configured
    remote_sync = None

    if remote_host and remote_user and remote_path:
        logger.info(f"\nRemote sync:")
        logger.info(f"  Host:             {remote_host}:{remote_port}")
        logger.info(f"  User:             {remote_user}")
        logger.info(f"  Path:             {remote_path}")
        logger.info(f"  Delete after:     {'Yes' if delete_after_upload else 'No'}")

        try:
            remote_sync = RemoteSync(
                host=remote_host,
                port=remote_port,
                username=remote_user,
                remote_path=remote_path,
                key_path=ssh_key,
                logger=logger,
            )
            logger.info("\nConnecting to remote server...")
            remote_sync.connect()
            logger.info("Remote connection established.")
        except Exception as e:
            logger.error(f"Failed to connect to remote server: {e}")
            logger.error("Aborting. Please check your SSH configuration.")
            return

    try:
        # ======================================================================
        # CUSTOM INPUT MODE: Scan provided directory for TSV files
        # ======================================================================
        if input_dir:
            drug2protein_glob = os.path.join(input_dir, "1_drug_to_protein_*.tsv")
            drug2protein_files = sorted(glob.glob(drug2protein_glob))

            if not drug2protein_files:
                logger.error(f"No TSV files found matching: {drug2protein_glob}")
                return

            logger.info(f"\nFound {len(drug2protein_files)} TSV file(s) in {input_dir}")

            # Interactive selection
            if interactive:
                drug2protein_files = select_runs_interactively(
                    drug2protein_files,
                    pipeline_name=os.path.basename(input_dir.rstrip("/")),
                )
                if not drug2protein_files:
                    logger.info("No configurations selected. Exiting.")
                    return
                logger.info(f"Selected {len(drug2protein_files)} configuration(s).")

            # Determine output directory
            if output_dir:
                save_root = output_dir
            else:
                pipeline_name = os.path.basename(input_dir.rstrip("/"))
                save_root = os.path.join(output_prefix, pipeline_name)

            logger.info(f"\nOutput directory: {save_root}")

            logger.info("\n" + "=" * 80)
            logger.info(f"Computing diffusion profiles for custom input: {input_dir}")
            logger.info("=" * 80)

            runs = compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs(
                save_root=save_root,
                drug2protein_file_paths=drug2protein_files,
                recompute=recompute,
                on_error="skip",
                sequential=sequential,
                num_cores=num_cores,
                remote_sync=remote_sync,
                delete_after_upload=delete_after_upload,
                logger=logger,
                indication2protein_file_path=indication2protein_path,
                protein2protein_file_path=protein2protein_path,
                protein2biological_function_file_path=protein2bio_func_path,
                biological_function2biological_function_file_path=bio_func2bio_func_path,
            )

            logger.info(f"\nCompleted {len(runs)} diffusion-profile runs")
            logger.info(f"Output: {save_root}")

        # ======================================================================
        # LEGACY MODE: Run predefined pipelines
        # ======================================================================
        else:
            if not run_baseline and not run_docking and not run_dta_atlas:
                logger.error("At least one pipeline must be enabled!")
                return

            logger.info(f"\nPipelines to run:")
            logger.info(f"  - Baseline MSI:  {'Yes' if run_baseline else 'No'}")
            logger.info(f"  - Docking:       {'Yes' if run_docking else 'No'}")
            logger.info(f"  - DTA-Atlas:     {'Yes' if run_dta_atlas else 'No'}")

            # ==================================================================
            # BASELINE: Default MSI (no filtered edges)
            # ==================================================================
            if run_baseline:
                baseline_output = output_dir if output_dir else "results/"
                logger.info("\n" + "=" * 80)
                logger.info("BASELINE: Default MSI diffusion profiles")
                logger.info("=" * 80)
                _, _ = compute_all_diffusion_profiles_for_msi(
                    save_load_file_path=baseline_output,
                    num_cores=num_cores,
                )
                logger.info(f"Baseline profiles saved to: {baseline_output}")

            # ==================================================================
            # DOCKING: Filtered edges from docking/ML scores
            # ==================================================================
            if run_docking:
                docking_glob = "data/docking/1_drug_to_protein_*cutoff_*.tsv"
                docking_files = sorted(glob.glob(docking_glob))

                if docking_files:
                    if interactive:
                        docking_files = select_runs_interactively(docking_files, "Docking")
                        if not docking_files:
                            logger.info("No docking configurations selected. Skipping.")
                        else:
                            logger.info(f"Selected {len(docking_files)} docking configuration(s).")

                    if docking_files:
                        docking_output = os.path.join(output_prefix, "docking")
                        logger.info("\n" + "=" * 80)
                        logger.info("DOCKING: Diffusion profiles for docking-filtered edges")
                        logger.info("=" * 80)
                        runs_docking = compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs(
                            save_root=docking_output,
                            drug2protein_file_paths=docking_files,
                            recompute=recompute,
                            on_error="skip",
                            sequential=sequential,
                            num_cores=num_cores,
                            remote_sync=remote_sync,
                            delete_after_upload=delete_after_upload,
                            logger=logger,
                            indication2protein_file_path=indication2protein_path,
                            protein2protein_file_path=protein2protein_path,
                            protein2biological_function_file_path=protein2bio_func_path,
                            biological_function2biological_function_file_path=bio_func2bio_func_path,
                        )
                        logger.info(f"\nDocking: Completed {len(runs_docking)} diffusion-profile runs")
                        logger.info(f"Output: {docking_output}")
                else:
                    logger.warning(f"No docking TSV files found matching: {docking_glob}")

            # ==================================================================
            # DTA-ATLAS: Filtered edges from DTA-Atlas predictions
            # ==================================================================
            if run_dta_atlas:
                dta_glob = "data/dta_atlas/1_drug_to_protein_*cutoff_*.tsv"
                dta_files = sorted(glob.glob(dta_glob))

                if dta_files:
                    if interactive:
                        dta_files = select_runs_interactively(dta_files, "DTA-Atlas")
                        if not dta_files:
                            logger.info("No DTA-Atlas configurations selected. Skipping.")
                        else:
                            logger.info(f"Selected {len(dta_files)} DTA-Atlas configuration(s).")

                    if dta_files:
                        dta_output = os.path.join(output_prefix, "dta_atlas")
                        logger.info("\n" + "=" * 80)
                        logger.info("DTA-ATLAS: Diffusion profiles for DTA-Atlas filtered edges")
                        logger.info("=" * 80)
                        runs_dta_atlas = compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs(
                            save_root=dta_output,
                            drug2protein_file_paths=dta_files,
                            recompute=recompute,
                            on_error="skip",
                            sequential=sequential,
                            num_cores=num_cores,
                            remote_sync=remote_sync,
                            delete_after_upload=delete_after_upload,
                            logger=logger,
                            indication2protein_file_path=indication2protein_path,
                            protein2protein_file_path=protein2protein_path,
                            protein2biological_function_file_path=protein2bio_func_path,
                            biological_function2biological_function_file_path=bio_func2bio_func_path,
                        )
                        logger.info(f"\nDTA-Atlas: Completed {len(runs_dta_atlas)} diffusion-profile runs")
                        logger.info(f"Output: {dta_output}")
                else:
                    logger.warning(f"No DTA-Atlas TSV files found matching: {dta_glob}")

    except UploadError as e:
        logger.error(f"\nUpload failed: {e}")
        logger.error("Computation aborted due to upload failure.")
        raise
    except KeyboardInterrupt:
        logger.warning("\nComputation interrupted by user.")
        raise
    finally:
        # Close remote connection
        if remote_sync:
            remote_sync.close()
            logger.info("Remote connection closed.")

    # Summary
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("\n" + "=" * 80)
    logger.info("DIFFUSION PROFILE COMPUTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {hours}h {minutes}m {seconds}s")
    logger.info(f"Log file: {log_output_dir}/diffusion_profiles_*.log")


if __name__ == "__main__":
    # Note: The diffusion profile computation now uses explicit 'spawn' context
    # for memory-efficient parallel processing. Workers load data from disk
    # instead of inheriting parent memory via fork.

    parser = argparse.ArgumentParser(
        description="Compute diffusion profiles for MSI networks with various filtered edge sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all pipelines (default)
  python 2_compute_diffusion_profiles.py

  # Custom input directory (scans for TSV files)
  python 2_compute_diffusion_profiles.py --input-dir /path/to/edges/ --output-dir /tmp/results

  # Custom input with custom data directory (for MSI base files)
  python 2_compute_diffusion_profiles.py --input-dir /scratch/edges/dta_atlas/ \\
      --data-dir /scratch/edges/data/ --output-dir /tmp/results

  # Interactive selection
  python 2_compute_diffusion_profiles.py --input-dir data/dta_atlas/ --interactive

  # Run only DTA-Atlas pipeline (legacy mode)
  python 2_compute_diffusion_profiles.py --dta-atlas-only

  # Remote upload with delete-after-upload
  python 2_compute_diffusion_profiles.py --input-dir data/dta_atlas/ \\
      --data-dir data/ --remote-host myserver.edu --remote-user myuser \\
      --remote-path /data/results --delete-after-upload

  # Resume with remote checking
  python 2_compute_diffusion_profiles.py --input-dir data/dta_atlas/ --resume \\
      --remote-host myserver.edu --remote-user myuser --remote-path /data/results

  # Low memory mode
  python 2_compute_diffusion_profiles.py --sequential --num-cores 4

  # Combine options
  python 2_compute_diffusion_profiles.py --dta-atlas-only --sequential --resume --interactive
        """
    )

    # Input/output options
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Custom input directory containing drug_to_protein TSV files (from 1_prepare_filtered_edges.py)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for results"
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="results_filtered",
        help="Prefix for filtered output directories in legacy mode (default: results_filtered)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base directory for MSI data files (indication_to_protein, protein_to_protein, etc.). Default: data/"
    )

    # Pipeline selection (legacy mode)
    pipeline_group = parser.add_mutually_exclusive_group()

    pipeline_group.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline MSI (no filtered edges)"
    )

    pipeline_group.add_argument(
        "--docking-only",
        action="store_true",
        help="Run only the docking pipeline (skip baseline and DTA-Atlas)"
    )

    pipeline_group.add_argument(
        "--dta-atlas-only",
        action="store_true",
        help="Run only the DTA-Atlas pipeline (skip baseline and docking)"
    )

    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the baseline MSI computation"
    )

    # Computation options
    parser.add_argument(
        "--num-cores",
        type=int,
        default=12,
        help="Number of CPU cores to use (default: 12)"
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process profiles sequentially (~15GB, slowest but lowest memory)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed runs (checks local and remote if configured)"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactively select which configurations to compute"
    )

    # Remote sync options
    parser.add_argument(
        "--remote-host",
        type=str,
        help="Remote SSH host for result backup"
    )

    parser.add_argument(
        "--remote-port",
        type=int,
        default=22,
        help="Remote SSH port (default: 22)"
    )

    parser.add_argument(
        "--remote-user",
        type=str,
        help="Remote SSH username"
    )

    parser.add_argument(
        "--remote-path",
        type=str,
        help="Remote directory path for results"
    )

    parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key (optional, uses default keys if not specified)"
    )

    parser.add_argument(
        "--delete-after-upload",
        action="store_true",
        help="Delete local .npy files after successful upload to remote (saves disk space)"
    )

    args = parser.parse_args()

    # Determine which pipelines to run (legacy mode)
    if args.input_dir:
        # Custom input mode - ignore legacy pipeline flags
        run_baseline = False
        run_docking = False
        run_dta_atlas = False
    elif args.baseline_only:
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

    # Validate remote sync options
    remote_options = [args.remote_host, args.remote_user, args.remote_path]
    if any(remote_options) and not all(remote_options):
        parser.error("--remote-host, --remote-user, and --remote-path must all be specified together")

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        data_dir=args.data_dir,
        run_baseline=run_baseline,
        run_docking=run_docking,
        run_dta_atlas=run_dta_atlas,
        num_cores=args.num_cores,
        sequential=args.sequential,
        recompute=not args.resume,
        interactive=args.interactive,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_user=args.remote_user,
        remote_path=args.remote_path,
        ssh_key=args.ssh_key,
        delete_after_upload=args.delete_after_upload,
    )
