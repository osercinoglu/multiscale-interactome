from msi.msi import MSI
from diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx
import polars as pl
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
#import paramiko

from filter_logic import filter_drug_protein_edges
from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles

DRUG_COL = 'drugbank_id'
TGT_COL = 'uniprot_id'
LABEL_COL = 'sample_type'  # positive / negative_balanced
PAIR_COL = 'compound_target_pair'


def _pair_best_score(df_in, score_col: str, best: str, streaming: bool = False):
    """Extract best score per compound-target pair from a LazyFrame or DataFrame.

    Args:
        df_in: LazyFrame or DataFrame with scores
        score_col: Column name containing scores
        best: 'min' or 'max' - which score is better
        streaming: If True, use streaming mode for memory efficiency (recommended for large datasets)
    """
    if best not in {"min", "max"}:
        raise ValueError("best must be 'min' or 'max'")

    # Convert to lazy if needed
    df_lazy = df_in if isinstance(df_in, pl.LazyFrame) else df_in.lazy()

    # Check if column exists by trying to select it
    try:
        df_lazy.select(pl.col(score_col)).collect_schema()
    except Exception:
        print(f"Warning: column not found, skipping: {score_col}")
        return pl.DataFrame()

    agg_score = (pl.min(score_col) if best == "min" else pl.max(score_col)).alias(score_col)

    # Build aggregation columns based on what's available
    agg_cols = [agg_score]

    # Add optional columns if they exist in the schema
    schema = df_lazy.collect_schema()
    if DRUG_COL in schema:
        agg_cols.append(pl.first(DRUG_COL).alias(DRUG_COL))
    if TGT_COL in schema:
        agg_cols.append(pl.first(TGT_COL).alias(TGT_COL))
    if LABEL_COL in schema:
        agg_cols.append(pl.first(LABEL_COL).alias(LABEL_COL))

    result = (
        df_lazy
        .filter(pl.col(score_col).is_not_null() & pl.col(score_col).is_finite())
        .group_by(PAIR_COL)
        .agg(agg_cols)
    )

    return result.collect(engine="streaming")


def load_dta_atlas_scores(base_path: str, score_col: str | None = None):
    """
    Load DTA-Atlas parquet files and adapt schema to match expected format.

    Args:
        base_path: Path to directory containing parquet files
        score_col: If provided, only load this score column plus ID columns (memory-efficient)

    Returns:
        pl.LazyFrame with compound_target_pair column added
    """
    df_lazy = pl.scan_parquet(f"{base_path}/*.parquet")

    # MEMORY OPTIMIZATION: Only select columns we need
    # This reduces memory from ~50GB (all 28 cols) to ~3-5GB (3 cols)
    if score_col is not None:
        df_lazy = df_lazy.select([DRUG_COL, TGT_COL, score_col])

    # Add compound_target_pair to match existing schema
    df_lazy = df_lazy.with_columns(
        (pl.col(DRUG_COL) + "_" + pl.col(TGT_COL)).alias(PAIR_COL)
    )

    return df_lazy


def process_tool_scores(
    df_scores,
    tool_name,
    score_column,
    better,
    cutoff_range,
    df_edges,
    df_mapping,
    output_dir="data",
    output_prefix="1_drug_to_protein_filtered",
    edge_source='from_scores'
):
    """
    Generic function to process scores and generate filtered edges for one tool.

    Args:
        df_scores: DataFrame with scores (already processed by _pair_best_score)
        tool_name: Name of the tool (for output filename)
        score_column: Column name containing scores
        better: 'lower' or 'higher'
        cutoff_range: Array of cutoff values to iterate over
        df_edges: Original drug-protein edges
        df_mapping: Uniprot-gene mapping
        output_dir: Directory for output files
        output_prefix: Prefix for output filenames
        edge_source: 'filter_existing' or 'from_scores'
    """
    if df_scores.is_empty():
        print(f"Skipping filtering for {tool_name} due to missing scores.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for cutoff in cutoff_range:
        score_cutoff = cutoff
        filtered_edges = filter_drug_protein_edges(
            df_edges,
            df_mapping,
            df_scores,
            score_column=score_column,
            better=better,
            score_cutoff=score_cutoff,
            edge_source=edge_source,
            on_unmapped_protein='drop',
        )

        # Format cutoff for filename (handle negative values and floats)
        cutoff_str = f"{score_cutoff}".replace(".", "_").replace("-", "neg")
        output_path = f"{output_dir}/{output_prefix}_{tool_name.lower()}_cutoff_{cutoff_str}.tsv"

        filtered_edges.write_csv(output_path, separator="\t")
        print(f"Saved filtered edges for {tool_name} with cutoff {score_cutoff} to {output_path}")


def main(run_docking=True, run_dta_atlas=True):
    """
    Main execution function.

    Args:
        run_docking: Whether to run the docking pipeline (default: True)
        run_dta_atlas: Whether to run the DTA-Atlas pipeline (default: True)
    """
    print("="*80)
    print("DRUG-PROTEIN EDGE FILTERING PIPELINE")
    print("="*80)
    print(f"\nPipelines to run:")
    print(f"  - Docking + ML (original): {'Yes' if run_docking else 'No'}")
    print(f"  - DTA-Atlas (extended):    {'Yes' if run_dta_atlas else 'No'}")

    if not run_docking and not run_dta_atlas:
        print("\nError: At least one pipeline must be enabled!")
        return

    # Load base data once
    print("\nLoading base data...")
    df_edges = pl.read_csv("data/1_drug_to_protein.tsv", separator="\t")
    df_mapping = pl.read_csv("data/uniprot_gene_mapping.csv")
    print(f"  - Drug-protein edges: {len(df_edges)} rows")
    print(f"  - Uniprot-gene mapping: {len(df_mapping)} rows")

    # ========================================================================
    # PIPELINE 1: DOCKING + ML (Original Dataset)
    # ========================================================================
    if run_docking:
        print("\n" + "="*80)
        print("PIPELINE 1: DOCKING + ML SCORES (Original Dataset)")
        print("="*80)

        df_lazy_docking = pl.scan_parquet("/scratch/serci001/data/combined_filtered_annotated_docking_results.parquet")

        # Define tool configurations (without loading data yet)
        docking_tool_configs = [
            ("GNINA", "GNINA_Score", "min", np.arange(-7.0, -9.1, -0.5)),
            ("SMINA", "SMINA_Score", "min", np.arange(-7.0, -9.1, -0.5)),
            ("LEDOCK", "Ledock_Score", "min", np.arange(-5.0, -7.1, -0.5)),
            ("GOLD", "GOLD_Score", "max", np.arange(60, 91, 10)),
            ("DeepDTA_ic50", "dta_DeepDTA_ic50", "max", np.arange(5, 9, 1)),
            ("GAT_GCN_ic50", "dta_GAT_GCN_ic50", "max", np.arange(5, 9, 1)),
            ("GCNNet_ic50", "dta_GCNNet_ic50", "max", np.arange(5, 9, 1)),
            ("GINConvNet_ic50", "dta_GINConvNet_ic50", "max", np.arange(5, 9, 1)),
            ("MLTLE_GCN_ic50", "dta_MLTLE_GCN_ic50", "max", np.arange(5, 9, 1)),
            ("MLTLE_GIN_ic50", "dta_MLTLE_GIN_ic50", "max", np.arange(5, 9, 1)),
        ]

        total_tools = len(docking_tool_configs)
        print(f"\nProcessing {total_tools} tools (4 docking + 6 ML)")

        # Process each tool one at a time (memory-efficient)
        for idx, (tool_name, score_col, best, cutoffs) in enumerate(docking_tool_configs, 1):
            print(f"\n[{idx}/{total_tools}] Processing {tool_name}...")

            # Extract scores for this tool only
            print(f"  Extracting scores...")
            df_scores = _pair_best_score(df_lazy_docking, score_col, best=best)

            if df_scores.is_empty():
                print(f"  Skipping {tool_name} - no scores found")
                continue

            print(f"  Extracted {len(df_scores):,} unique pairs")

            # Determine better direction for filter_drug_protein_edges
            better = "lower" if best == "min" else "higher"

            # Process immediately
            process_tool_scores(
                df_scores=df_scores,
                tool_name=tool_name,
                score_column=score_col,
                better=better,
                cutoff_range=cutoffs,
                df_edges=df_edges,
                df_mapping=df_mapping,
                output_dir="data/docking",
                output_prefix="1_drug_to_protein",
                edge_source='from_scores'
            )

            # Free memory before next iteration
            del df_scores
            gc.collect()

    # ========================================================================
    # PIPELINE 2: DTA-ATLAS (Extended ML Predictions)
    # ========================================================================
    if run_dta_atlas:
        print("\n" + "="*80)
        print("PIPELINE 2: DTA-ATLAS DATASET (Extended ML Predictions)")
        print("="*80)

        dta_atlas_path = "/scratch/serci001/data/dta-atlas/home/madinasu/convertedData/dta_atlas_dataset_1_0"

        # Define all ML tools with all score types (kd, ki, ic50)
        ml_tools = ["DeepDTA", "GAT_GCN", "GCNNet", "GINConvNet", "MLTLE_GCN", "MLTLE_GIN"]
        score_types = ["kd", "ki", "ic50"]
        cutoff_range = np.arange(5, 9, 1)

        # Also process aggregate scores
        aggregate_scores = ["avg_kd", "avg_ki", "avg_ic50"]

        # Build list of all tool variants to process
        tool_variants = []
        for tool in ml_tools:
            for score_type in score_types:
                tool_variants.append((f"{tool}_{score_type}", f"{tool}_{score_type}"))
        for agg_score in aggregate_scores:
            tool_variants.append((agg_score, agg_score))

        total_variants = len(tool_variants)
        print(f"\nProcessing {len(ml_tools)} ML tools × {len(score_types)} score types = {len(ml_tools) * len(score_types)} variants")
        print(f"Plus {len(aggregate_scores)} aggregate scores")
        print(f"Total: {total_variants} variants")
        print(f"\nUsing memory-efficient mode: loading only required columns per tool")

        # Process each tool variant one at a time (memory-efficient)
        # For each tool: load ONLY needed columns -> process -> free memory
        for idx, (tool_variant, score_col) in enumerate(tool_variants, 1):
            print(f"\n[{idx}/{total_variants}] Processing {tool_variant}...")

            # Create a FRESH lazy frame with ONLY the columns we need (3 cols instead of 28)
            # This is the key memory optimization: ~3GB instead of ~50GB
            print(f"  Loading only required columns...")
            df_lazy_atlas = load_dta_atlas_scores(dta_atlas_path, score_col=score_col)

            # Extract scores using streaming mode for additional memory efficiency
            print(f"  Extracting scores (streaming mode)...")
            df_scores = _pair_best_score(df_lazy_atlas, score_col, best="max", streaming=True)

            # Free the lazy frame reference
            del df_lazy_atlas

            if df_scores.is_empty():
                print(f"  Skipping {tool_variant} - no scores found")
                gc.collect()
                continue

            print(f"  Extracted {len(df_scores):,} unique pairs")

            # Process immediately
            process_tool_scores(
                df_scores=df_scores,
                tool_name=tool_variant,
                score_column=score_col,
                better="higher",
                cutoff_range=cutoff_range,
                df_edges=df_edges,
                df_mapping=df_mapping,
                output_dir="data/dta_atlas",
                output_prefix="1_drug_to_protein",
                edge_source='from_scores'
            )

            # Free memory before next iteration
            del df_scores
            gc.collect()

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    if run_docking:
        print(f"  - Docking pipeline: data/docking/1_drug_to_protein_*.tsv")
    if run_dta_atlas:
        print(f"  - DTA-Atlas pipeline: data/dta_atlas/1_drug_to_protein_*.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate filtered drug-protein edge files from docking and/or ML prediction scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both pipelines (default)
  python 1_prepare_filtered_edges.py

  # Run only DTA-Atlas pipeline
  python 1_prepare_filtered_edges.py --dta-atlas-only

  # Run only docking pipeline
  python 1_prepare_filtered_edges.py --docking-only
        """
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--docking-only",
        action="store_true",
        help="Run only the docking pipeline (skip DTA-Atlas)"
    )

    group.add_argument(
        "--dta-atlas-only",
        action="store_true",
        help="Run only the DTA-Atlas pipeline (skip docking)"
    )

    args = parser.parse_args()

    # Determine which pipelines to run based on arguments
    if args.docking_only:
        run_docking = True
        run_dta_atlas = False
    elif args.dta_atlas_only:
        run_docking = False
        run_dta_atlas = True
    else:
        # Default: run both
        run_docking = True
        run_dta_atlas = True

    main(run_docking=run_docking, run_dta_atlas=run_dta_atlas)
