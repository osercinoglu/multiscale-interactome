#!/usr/bin/env python3
"""Assess drug-disease prediction performance from diffusion profile similarities.

Evaluates how effectively similarity scores rank approved drugs for each disease
using AUROC, Average Precision, and Recall@K. Supports optional k-fold
cross-validation on the drug set.

Input: CSV output from 3_compare_diffusion_profiles.py (within-model mode)
Reference: TSV/CSV of approved drug-indication pairs
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

from utils.logging_setup import setup_logging


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerDiseaseResult:
    """Metrics for a single indication (disease)."""

    indication_id: str
    n_drugs_total: int
    n_drugs_positive: int
    auroc: float | None
    avg_precision: float | None
    recall_at_k: float | None
    top_k: int
    fold: int | None = None  # None = full dataset, 0..n-1 = CV fold


@dataclass
class OverallResult:
    """Aggregated metrics across indications."""

    method: str | None
    run_id: str | None
    mode: str  # "full" or "cv"
    n_indications: int
    median_auroc: float | None
    mean_auroc: float | None
    std_auroc: float | None
    mean_avg_precision: float | None
    std_avg_precision: float | None
    mean_recall_at_k: float | None
    std_recall_at_k: float | None
    top_k: int
    n_folds: int | None = None
    fold_median_aurocs: list[float] = field(default_factory=list)
    fold_mean_aps: list[float] = field(default_factory=list)
    fold_mean_recalls: list[float] = field(default_factory=list)


@dataclass
class PairedComparison:
    """Metrics for one test run paired with a baseline, both on a common drug pool."""

    method: str
    run_id: str
    n_common_drugs: int
    mode: str   # "restricted" or "restricted_cv"
    top_k: int
    # Test-run metrics
    run_auroc: float | None
    run_ap: float | None
    run_recall: float | None
    # Baseline metrics (same common drug pool)
    baseline_auroc: float | None
    baseline_ap: float | None
    baseline_recall: float | None
    # Deltas: run - baseline (positive = run beats baseline)
    delta_auroc: float | None
    delta_ap: float | None
    delta_recall: float | None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _normalize_reference_columns(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Normalize drug-indication column names to 'drug' and 'indication'.

    Accepts:
      - 'drug' + 'indication'   (existing 6_drug_indication_df.tsv format)
      - 'drug_id' + 'ind_id'    (drug repurposing DB format)

    Raises ValueError with a clear message if neither is found.
    """
    if "drug" in df.columns and "indication" in df.columns:
        return df
    if "drug_id" in df.columns and "ind_id" in df.columns:
        return df.rename(columns={"drug_id": "drug", "ind_id": "indication"})
    raise ValueError(
        f"Reference file '{path}' must have columns ('drug', 'indication') "
        f"or ('drug_id', 'ind_id'). Found: {list(df.columns)}"
    )


def load_reference(
    path: str,
    logger: logging.Logger,
    status_filter: list[str] | None = None,
) -> dict[str, set[str]]:
    """Load approved drug-indication pairs from TSV/CSV.

    Auto-detects separator by inspecting the first line.
    Accepts column names 'drug'/'indication' (existing format) or
    'drug_id'/'ind_id' (drug repurposing DB format).

    Args:
        path: Path to reference TSV/CSV file.
        logger: Logger instance.
        status_filter: If provided and the file has a 'status' column,
            only rows with matching status values are kept
            (e.g. ['Approved']).

    Returns:
        dict mapping indication_id -> set of approved drug_ids
    """
    with open(path) as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(path, sep=sep, dtype=str)
    df = _normalize_reference_columns(df, path)

    if status_filter is not None:
        if "status" in df.columns:
            n_before = len(df)
            df = df[df["status"].isin(status_filter)]
            n_after = len(df)
            logger.info(
                f"  Status filter {status_filter}: kept {n_after:,} / {n_before:,} rows"
            )
        else:
            logger.warning(
                f"  --status-filter provided but reference has no 'status' column — "
                f"filter ignored."
            )

    df = df[["drug", "indication"]].dropna()

    approved_by_indication: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        approved_by_indication.setdefault(row["indication"], set()).add(row["drug"])

    n_pairs = len(df)
    n_drugs = df["drug"].nunique()
    n_indications = len(approved_by_indication)
    logger.info(
        f"Reference: {n_pairs:,} approved pairs, "
        f"{n_drugs:,} unique drugs, {n_indications:,} unique indications"
    )
    return approved_by_indication


def load_similarity_csv(paths: list[str], logger: logging.Logger) -> pd.DataFrame:
    """Load one or more similarity CSVs.

    Only loads columns needed for evaluation (run_id, drug_id, indication_id,
    similarity) to minimize memory usage.
    """
    frames = []
    use_cols = ["run_id", "drug_id", "indication_id", "similarity"]
    for p in paths:
        try:
            df = pd.read_csv(p, usecols=use_cols, dtype={"run_id": str, "drug_id": str, "indication_id": str})
            frames.append(df)
            logger.debug(f"  Loaded {len(df):,} rows from {p}")
        except Exception as e:
            logger.warning(f"  Failed to load {p}: {e}")

    if not frames:
        raise ValueError(f"No valid CSV data loaded from {paths}")

    combined = pd.concat(frames, ignore_index=True)
    n_before = len(combined)
    combined = combined.dropna(subset=["similarity"])
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        logger.warning(f"  Dropped {n_dropped:,} rows with NaN similarity")

    logger.info(
        f"  Loaded {len(combined):,} rows, "
        f"{combined['drug_id'].nunique():,} drugs x "
        f"{combined['indication_id'].nunique():,} indications"
    )
    return combined


def discover_input_csvs(input_dir: str, logger: logging.Logger) -> dict[str, list[str]]:
    """Auto-discover CSV files from step 3 output directory.

    Supports:
      input_dir/method_*/within/*.csv  -> dict keyed by method name
      input_dir/within/*.csv           -> dict with single key
      input_dir/*.csv                  -> dict with single key
    """
    result: dict[str, list[str]] = {}

    # Try method_*/within/*.csv
    method_dirs = sorted(glob.glob(os.path.join(input_dir, "method_*")))
    if method_dirs:
        for method_dir in method_dirs:
            method_name = os.path.basename(method_dir).replace("method_", "")
            csvs = sorted(glob.glob(os.path.join(method_dir, "within", "*.csv")))
            if csvs:
                result[method_name] = csvs
                logger.info(f"  Discovered method '{method_name}': {len(csvs)} CSV(s)")
        if result:
            return result

    # Try within/*.csv
    within_csvs = sorted(glob.glob(os.path.join(input_dir, "within", "*.csv")))
    if within_csvs:
        key = os.path.basename(input_dir)
        result[key] = within_csvs
        logger.info(f"  Discovered {len(within_csvs)} CSV(s) in within/")
        return result

    # Try *.csv directly
    flat_csvs = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if flat_csvs:
        key = os.path.basename(input_dir)
        result[key] = flat_csvs
        logger.info(f"  Discovered {len(flat_csvs)} CSV(s) in {input_dir}")
        return result

    logger.warning(f"  No CSV files found in {input_dir}")
    return result


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics_for_indication(args) -> PerDiseaseResult | None:
    """Compute AUROC, AP, Recall@K for one indication.

    Module-level function for ProcessPoolExecutor (must be picklable).

    Args:
        args: tuple of (indication_id, drug_ids, similarities,
                        approved_set, top_k, fold)
    """
    indication_id, drug_ids, similarities, approved_set, top_k, fold = args

    y_true = np.array([1 if d in approved_set else 0 for d in drug_ids])
    y_score = np.array(similarities, dtype=np.float64)

    n_pos = int(y_true.sum())
    n_total = len(y_true)

    if n_pos == 0:
        return None

    auroc = None
    avg_prec = None
    recall_k = None

    # AUROC and AP require both positive and negative examples
    if n_pos > 0 and n_pos < n_total:
        auroc = float(roc_auc_score(y_true, y_score))
        avg_prec = float(average_precision_score(y_true, y_score))

    # Recall@K: of top K drugs by similarity, how many are approved?
    if n_pos > 0:
        top_k_actual = min(top_k, n_total)
        top_indices = np.argsort(y_score)[::-1][:top_k_actual]
        n_true_in_top_k = int(y_true[top_indices].sum())
        recall_k = n_true_in_top_k / n_pos

    return PerDiseaseResult(
        indication_id=indication_id,
        n_drugs_total=n_total,
        n_drugs_positive=n_pos,
        auroc=auroc,
        avg_precision=avg_prec,
        recall_at_k=recall_k,
        top_k=top_k,
        fold=fold,
    )


def evaluate_run(
    df: pd.DataFrame,
    approved_by_indication: dict[str, set[str]],
    top_k: int = 50,
    min_positives: int = 1,
    max_workers: int = 4,
    drug_subset: set[str] | None = None,
    fold: int | None = None,
    logger: logging.Logger | None = None,
) -> list[PerDiseaseResult]:
    """Evaluate prediction performance for all indications in the dataframe.

    Args:
        df: Similarity dataframe with columns drug_id, indication_id, similarity.
        approved_by_indication: Reference mapping indication -> set of approved drugs.
        top_k: K for Recall@K.
        min_positives: Minimum approved drugs per indication to include.
        max_workers: Number of parallel workers.
        drug_subset: If provided, only consider these drugs (for CV).
        fold: Fold index (None for full dataset).
        logger: Logger instance.
    """
    if drug_subset is not None:
        df = df[df["drug_id"].isin(drug_subset)]

    grouped = df.groupby("indication_id")
    worker_args = []
    skipped_no_positives = 0

    for indication_id, group in grouped:
        approved = approved_by_indication.get(indication_id, set())
        if drug_subset is not None:
            approved = approved & drug_subset
        if len(approved) < min_positives:
            skipped_no_positives += 1
            continue

        drug_ids = group["drug_id"].values.tolist()
        similarities = group["similarity"].values.tolist()
        worker_args.append((indication_id, drug_ids, similarities, approved, top_k, fold))

    if logger:
        logger.debug(
            f"  Evaluating {len(worker_args)} indications "
            f"(skipped {skipped_no_positives} with <{min_positives} approved drugs)"
        )

    results: list[PerDiseaseResult] = []

    if max_workers > 1 and len(worker_args) > 10:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for result in ex.map(compute_metrics_for_indication, worker_args, chunksize=10):
                if result is not None:
                    results.append(result)
    else:
        for args in worker_args:
            result = compute_metrics_for_indication(args)
            if result is not None:
                results.append(result)

    return results


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    df: pd.DataFrame,
    approved_by_indication: dict[str, set[str]],
    n_folds: int = 5,
    top_k: int = 50,
    min_positives: int = 1,
    max_workers: int = 4,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> list[list[PerDiseaseResult]]:
    """Run k-fold cross-validation on the drug set.

    Splits drugs into folds, computes metrics on held-out drugs per fold.

    Returns:
        List of per-fold result lists.
    """
    all_drugs = sorted(df["drug_id"].unique())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    if logger:
        logger.info(f"\n  Cross-validation: {n_folds} folds, {len(all_drugs)} drugs, seed={seed}")

    all_fold_results: list[list[PerDiseaseResult]] = []

    for fold_idx, (_, test_indices) in enumerate(kf.split(all_drugs)):
        held_out_drugs = set(np.array(all_drugs)[test_indices])

        if logger:
            logger.info(f"  Fold {fold_idx + 1}/{n_folds}: {len(held_out_drugs)} held-out drugs")

        # Filter approved sets to held-out drugs only
        fold_approved = {}
        for ind, drugs in approved_by_indication.items():
            filtered = drugs & held_out_drugs
            if filtered:
                fold_approved[ind] = filtered

        fold_results = evaluate_run(
            df,
            fold_approved,
            top_k=top_k,
            min_positives=min_positives,
            max_workers=max_workers,
            drug_subset=held_out_drugs,
            fold=fold_idx,
            logger=logger,
        )
        all_fold_results.append(fold_results)

        if logger:
            n_valid = len([r for r in fold_results if r.auroc is not None])
            logger.info(f"    {n_valid} indications with valid AUROC")

    return all_fold_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    per_disease: list[PerDiseaseResult],
    method: str | None = None,
    run_id: str | None = None,
    mode: str = "full",
    top_k: int = 50,
) -> OverallResult:
    """Aggregate per-disease results into overall metrics."""
    aurocs = [r.auroc for r in per_disease if r.auroc is not None]
    aps = [r.avg_precision for r in per_disease if r.avg_precision is not None]
    recalls = [r.recall_at_k for r in per_disease if r.recall_at_k is not None]

    return OverallResult(
        method=method,
        run_id=run_id,
        mode=mode,
        n_indications=len(per_disease),
        median_auroc=float(np.median(aurocs)) if aurocs else None,
        mean_auroc=float(np.mean(aurocs)) if aurocs else None,
        std_auroc=float(np.std(aurocs)) if aurocs else None,
        mean_avg_precision=float(np.mean(aps)) if aps else None,
        std_avg_precision=float(np.std(aps)) if aps else None,
        mean_recall_at_k=float(np.mean(recalls)) if recalls else None,
        std_recall_at_k=float(np.std(recalls)) if recalls else None,
        top_k=top_k,
    )


def aggregate_cv_results(
    fold_results: list[list[PerDiseaseResult]],
    method: str | None = None,
    run_id: str | None = None,
    top_k: int = 50,
    mode: str = "cv",
) -> OverallResult:
    """Aggregate cross-validation results across folds."""
    fold_summaries = [aggregate_results(fr, method=method, run_id=run_id, top_k=top_k) for fr in fold_results]

    fold_median_aurocs = [s.median_auroc for s in fold_summaries if s.median_auroc is not None]
    fold_mean_aps = [s.mean_avg_precision for s in fold_summaries if s.mean_avg_precision is not None]
    fold_mean_recalls = [s.mean_recall_at_k for s in fold_summaries if s.mean_recall_at_k is not None]

    # Total indications evaluated across all folds
    all_results = [r for fr in fold_results for r in fr]

    return OverallResult(
        method=method,
        run_id=run_id,
        mode=mode,
        n_indications=len(all_results),
        median_auroc=float(np.mean(fold_median_aurocs)) if fold_median_aurocs else None,
        mean_auroc=None,
        std_auroc=float(np.std(fold_median_aurocs)) if fold_median_aurocs else None,
        mean_avg_precision=float(np.mean(fold_mean_aps)) if fold_mean_aps else None,
        std_avg_precision=float(np.std(fold_mean_aps)) if fold_mean_aps else None,
        mean_recall_at_k=float(np.mean(fold_mean_recalls)) if fold_mean_recalls else None,
        std_recall_at_k=float(np.std(fold_mean_recalls)) if fold_mean_recalls else None,
        top_k=top_k,
        n_folds=len(fold_results),
        fold_median_aurocs=fold_median_aurocs,
        fold_mean_aps=fold_mean_aps,
        fold_mean_recalls=fold_mean_recalls,
    )


# ---------------------------------------------------------------------------
# Comparative evaluation helpers
# ---------------------------------------------------------------------------

def compute_common_drug_pool(
    dfs: list[pd.DataFrame],
    logger: logging.Logger,
) -> set[str]:
    """Return drug IDs present in every provided dataframe."""
    sets = [set(df["drug_id"].unique()) for df in dfs]
    common = sets[0].intersection(*sets[1:])
    pool_sizes = [len(s) for s in sets]
    logger.info(
        f"  Common drug pool: {len(common)} drugs "
        f"(intersection across pools of sizes {pool_sizes})"
    )
    return common


def compare_against_baseline(
    method_run_dfs: dict[str, list[tuple[str, pd.DataFrame]]],
    baseline_method_dfs: dict[str, pd.DataFrame],
    approved_by_indication: dict[str, set[str]],
    top_k: int,
    min_positives: int,
    cv: int,
    seed: int,
    max_workers: int,
    logger: logging.Logger,
) -> list[PairedComparison]:
    """Compare each test run against the same-method baseline on a common drug pool.

    For each (method, run_id) pair in method_run_dfs:
      - Finds the matching baseline df from baseline_method_dfs
      - Restricts both to the common drug intersection
      - Evaluates both (full, and optionally CV)
      - Returns a PairedComparison with metrics and deltas

    Args:
        method_run_dfs:     method → list of (run_id, df) for test runs
        baseline_method_dfs: method → baseline df
        approved_by_indication: reference drug-indication pairs
        top_k, min_positives, cv, seed, max_workers: evaluation settings
        logger: logger instance

    Returns:
        List of PairedComparison, sorted by method then delta_auroc descending.
    """
    results: list[PairedComparison] = []

    for method, run_list in sorted(method_run_dfs.items()):
        if method not in baseline_method_dfs:
            logger.warning(
                f"  No baseline found for method '{method}' — skipping {len(run_list)} run(s)."
            )
            continue

        baseline_df = baseline_method_dfs[method]
        logger.info(f"\n  Method '{method}': comparing {len(run_list)} run(s) against baseline")

        for run_id, run_df in run_list:
            common_drugs = run_df["drug_id"].unique()
            # Intersect with baseline's drug pool
            baseline_drugs = set(baseline_df["drug_id"].unique())
            common_drugs = set(common_drugs) & baseline_drugs
            n_common = len(common_drugs)

            if n_common == 0:
                logger.warning(f"    Run '{run_id}': no common drugs with baseline — skipping.")
                continue

            restricted_run = run_df[run_df["drug_id"].isin(common_drugs)]
            restricted_base = baseline_df[baseline_df["drug_id"].isin(common_drugs)]

            if cv > 0:
                # CV mode: run cross-validation on both
                run_folds = cross_validate(
                    restricted_run, approved_by_indication,
                    n_folds=cv, top_k=top_k, min_positives=min_positives,
                    max_workers=max_workers, seed=seed, logger=logger,
                )
                run_overall = aggregate_cv_results(
                    run_folds, method=method, run_id=run_id,
                    top_k=top_k, mode="restricted_cv",
                )
                base_folds = cross_validate(
                    restricted_base, approved_by_indication,
                    n_folds=cv, top_k=top_k, min_positives=min_positives,
                    max_workers=max_workers, seed=seed, logger=logger,
                )
                base_overall = aggregate_cv_results(
                    base_folds, method=method, run_id="baseline",
                    top_k=top_k, mode="restricted_cv",
                )
                mode = "restricted_cv"
            else:
                # Full mode
                run_results = evaluate_run(
                    restricted_run, approved_by_indication,
                    top_k=top_k, min_positives=min_positives,
                    max_workers=max_workers, logger=logger,
                )
                run_overall = aggregate_results(
                    run_results, method=method, run_id=run_id,
                    mode="restricted", top_k=top_k,
                )
                base_results = evaluate_run(
                    restricted_base, approved_by_indication,
                    top_k=top_k, min_positives=min_positives,
                    max_workers=max_workers, logger=logger,
                )
                base_overall = aggregate_results(
                    base_results, method=method, run_id="baseline",
                    mode="restricted", top_k=top_k,
                )
                mode = "restricted"

            def _delta(a: float | None, b: float | None) -> float | None:
                return (a - b) if (a is not None and b is not None) else None

            results.append(PairedComparison(
                method=method,
                run_id=run_id,
                n_common_drugs=n_common,
                mode=mode,
                top_k=top_k,
                run_auroc=run_overall.median_auroc,
                run_ap=run_overall.mean_avg_precision,
                run_recall=run_overall.mean_recall_at_k,
                baseline_auroc=base_overall.median_auroc,
                baseline_ap=base_overall.mean_avg_precision,
                baseline_recall=base_overall.mean_recall_at_k,
                delta_auroc=_delta(run_overall.median_auroc, base_overall.median_auroc),
                delta_ap=_delta(run_overall.mean_avg_precision, base_overall.mean_avg_precision),
                delta_recall=_delta(run_overall.mean_recall_at_k, base_overall.mean_recall_at_k),
            ))

    # Sort: by method, then best AUROC delta first
    results.sort(key=lambda p: (
        p.method,
        -(p.delta_auroc if p.delta_auroc is not None else float("-inf")),
    ))
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_per_disease_csv(
    results: list[PerDiseaseResult],
    path: str,
    method: str | None = None,
    run_id: str | None = None,
) -> None:
    """Write per-disease results to CSV."""
    rows = []
    for r in results:
        rows.append({
            "method": method or "",
            "run_id": run_id or "",
            "fold": r.fold if r.fold is not None else "",
            "indication_id": r.indication_id,
            "n_drugs_total": r.n_drugs_total,
            "n_drugs_positive": r.n_drugs_positive,
            "auroc": r.auroc if r.auroc is not None else "",
            "avg_precision": r.avg_precision if r.avg_precision is not None else "",
            f"recall_at_{r.top_k}": r.recall_at_k if r.recall_at_k is not None else "",
        })
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def write_summary_csv(summaries: list[OverallResult], path: str) -> None:
    """Write summary results to CSV."""
    rows = []
    for s in summaries:
        rows.append({
            "method": s.method or "",
            "run_id": s.run_id or "",
            "mode": s.mode,
            "n_indications": s.n_indications,
            "median_auroc": s.median_auroc if s.median_auroc is not None else "",
            "mean_auroc": s.mean_auroc if s.mean_auroc is not None else "",
            "std_auroc": s.std_auroc if s.std_auroc is not None else "",
            "mean_ap": s.mean_avg_precision if s.mean_avg_precision is not None else "",
            "std_ap": s.std_avg_precision if s.std_avg_precision is not None else "",
            f"mean_recall_at_{s.top_k}": s.mean_recall_at_k if s.mean_recall_at_k is not None else "",
            f"std_recall_at_{s.top_k}": s.std_recall_at_k if s.std_recall_at_k is not None else "",
            "n_folds": s.n_folds if s.n_folds is not None else "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _print_one_summary(s: OverallResult, logger: logging.Logger) -> None:
    """Print a single OverallResult entry."""
    label = []
    if s.method:
        label.append(f"Method: {s.method}")
    if s.run_id:
        label.append(f"Run: {s.run_id}")
    label.append(f"Mode: {s.mode}")
    logger.info(f"\n{' | '.join(label)}")
    logger.info("-" * 60)

    if s.mode in ("full", "restricted"):
        logger.info(f"  Indications evaluated:   {s.n_indications}")
        if s.median_auroc is not None:
            logger.info(f"  Median AUROC:            {s.median_auroc:.4f}")
        if s.mean_avg_precision is not None:
            logger.info(
                f"  Mean Avg Precision:      {s.mean_avg_precision:.4f}"
                f" +/- {s.std_avg_precision:.4f}"
            )
        if s.mean_recall_at_k is not None:
            logger.info(
                f"  Mean Recall@{s.top_k}:          {s.mean_recall_at_k:.4f}"
                f" +/- {s.std_recall_at_k:.4f}"
            )

    elif s.mode in ("cv", "restricted_cv"):
        logger.info(f"  {s.n_folds}-Fold Cross-Validation")
        logger.info(f"  Total indication evaluations: {s.n_indications}")
        if s.fold_median_aurocs:
            for i, (a, ap, r) in enumerate(
                zip(s.fold_median_aurocs, s.fold_mean_aps, s.fold_mean_recalls)
            ):
                logger.info(
                    f"    Fold {i + 1}: AUROC={a:.4f}  AP={ap:.4f}  R@{s.top_k}={r:.4f}"
                )
        if s.median_auroc is not None:
            logger.info(
                f"  Mean across folds:"
            )
            logger.info(
                f"    Median AUROC:      {s.median_auroc:.4f}"
                f" +/- {s.std_auroc:.4f}"
            )
        if s.mean_avg_precision is not None:
            logger.info(
                f"    Mean AP:           {s.mean_avg_precision:.4f}"
                f" +/- {s.std_avg_precision:.4f}"
            )
        if s.mean_recall_at_k is not None:
            logger.info(
                f"    Mean Recall@{s.top_k}:    {s.mean_recall_at_k:.4f}"
                f" +/- {s.std_recall_at_k:.4f}"
            )


def print_summary(summaries: list[OverallResult], logger: logging.Logger) -> None:
    """Print formatted summary to console/log."""
    main_summaries = [s for s in summaries if s.mode in ("full", "cv")]
    restricted_summaries = [s for s in summaries if s.mode in ("restricted", "restricted_cv")]

    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE ASSESSMENT RESULTS")
    logger.info("=" * 80)
    for s in main_summaries:
        _print_one_summary(s, logger)

    if restricted_summaries:
        logger.info("\n" + "=" * 80)
        logger.info("COMPARATIVE EVALUATION (common drug pool — pool-size-invariant)")
        logger.info("=" * 80)
        for s in restricted_summaries:
            _print_one_summary(s, logger)

    logger.info("\n" + "=" * 80)


def print_comparison_table(
    comparisons: list[PairedComparison],
    logger: logging.Logger,
    top_k: int = 50,
) -> None:
    """Print a compact comparison table (one section per method) to the log."""
    if not comparisons:
        return

    methods = sorted({p.method for p in comparisons})
    id_width = 44

    for method in methods:
        method_pairs = [p for p in comparisons if p.method == method]
        mode = method_pairs[0].mode if method_pairs else "restricted"

        logger.info("\n" + "=" * 80)
        logger.info(
            f"COMPARISON TABLE — method: {method} | mode: {mode} | common drug pool per run"
        )
        logger.info("=" * 80)

        header = (
            f"{'Run':<{id_width}}  {'Drugs':>5}"
            f"  {'AUROC':>6}  {'Base':>6}  {'ΔAUROC':>7}"
            f"  {'AP':>6}  {'Base':>6}  {'ΔAP':>7}"
            f"  {'R@'+str(top_k):>6}  {'Base':>6}  {'ΔR':>7}"
        )
        logger.info(header)
        logger.info("─" * len(header))

        for p in method_pairs:
            run_label = p.run_id
            if len(run_label) > id_width:
                run_label = run_label[:id_width - 1] + "…"

            def _fmt(v: float | None) -> str:
                return f"{v:.4f}" if v is not None else "   N/A"

            def _fmt_delta(v: float | None) -> str:
                if v is None:
                    return "    N/A"
                return f"{v:+.4f}"

            row = (
                f"{run_label:<{id_width}}  {p.n_common_drugs:>5}"
                f"  {_fmt(p.run_auroc):>6}  {_fmt(p.baseline_auroc):>6}  {_fmt_delta(p.delta_auroc):>7}"
                f"  {_fmt(p.run_ap):>6}  {_fmt(p.baseline_ap):>6}  {_fmt_delta(p.delta_ap):>7}"
                f"  {_fmt(p.run_recall):>6}  {_fmt(p.baseline_recall):>6}  {_fmt_delta(p.delta_recall):>7}"
            )
            logger.info(row)


def write_comparison_csv(comparisons: list[PairedComparison], path: str) -> None:
    """Write comparison table to CSV."""
    if not comparisons:
        return
    rows = []
    for p in comparisons:
        rows.append({
            "method": p.method,
            "run_id": p.run_id,
            "n_common_drugs": p.n_common_drugs,
            "mode": p.mode,
            "top_k": p.top_k,
            "run_auroc": p.run_auroc if p.run_auroc is not None else "",
            "baseline_auroc": p.baseline_auroc if p.baseline_auroc is not None else "",
            "delta_auroc": p.delta_auroc if p.delta_auroc is not None else "",
            "run_ap": p.run_ap if p.run_ap is not None else "",
            "baseline_ap": p.baseline_ap if p.baseline_ap is not None else "",
            "delta_ap": p.delta_ap if p.delta_ap is not None else "",
            "run_recall": p.run_recall if p.run_recall is not None else "",
            "baseline_recall": p.baseline_recall if p.baseline_recall is not None else "",
            "delta_recall": p.delta_recall if p.delta_recall is not None else "",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    input_csv: list[str] | None = None,
    input_dir: str | None = None,
    reference: str = "",
    top_k: int = 50,
    min_positives: int = 1,
    cv: int = 0,
    seed: int = 42,
    output_dir: str = "results_assessment",
    max_workers: int | None = None,
    compare: bool = False,
    baseline_dir: str | None = None,
    status_filter: list[str] | None = None,
) -> None:
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 1)

    logger = setup_logging(
        output_dir,
        log_name="assess_performance",
        logger_name="assess_performance",
    )

    logger.info("=" * 80)
    logger.info("DRUG-DISEASE PREDICTION PERFORMANCE ASSESSMENT")
    logger.info("=" * 80)
    logger.info(f"Command: {' '.join(sys.argv)}")

    logger.info(f"\nConfiguration:")
    logger.info(f"  Reference:        {reference}")
    logger.info(f"  Top-K:            {top_k}")
    logger.info(f"  Min positives:    {min_positives}")
    logger.info(f"  Cross-validation: {cv} folds" if cv > 0 else "  Cross-validation: disabled")
    logger.info(f"  Seed:             {seed}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Max workers:      {max_workers}")
    logger.info(f"  Compare mode:     {'enabled' if compare else 'disabled'}")
    if baseline_dir:
        logger.info(f"  Baseline dir:     {baseline_dir}")
    if status_filter:
        logger.info(f"  Status filter:    {status_filter}")

    # 1. Load reference
    logger.info(f"\nLoading reference data...")
    approved_by_indication = load_reference(reference, logger, status_filter=status_filter)

    # 2. Discover/load CSVs
    logger.info(f"\nDiscovering input CSVs...")
    if input_dir:
        method_csvs = discover_input_csvs(input_dir, logger)
    else:
        method_csvs = {"": input_csv}

    if not method_csvs:
        logger.error("No input CSVs found. Provide --input-csv or --input-dir.")
        return

    # 3. Evaluate each method/CSV set
    all_summaries: list[OverallResult] = []
    all_per_disease: list[tuple[str | None, str | None, list[PerDiseaseResult]]] = []

    summary_path = os.path.join(output_dir, "summary.csv")
    per_disease_path = os.path.join(output_dir, "per_disease_results.csv")

    # Remove existing output files (we append per-disease results)
    for p in [summary_path, per_disease_path]:
        if os.path.exists(p):
            os.remove(p)

    # Accumulate (method, run_id, df) for comparative evaluation
    all_loaded_dfs: list[tuple[str, str, pd.DataFrame]] = []

    total_methods = len(method_csvs)
    for method_idx, (method_name, csv_paths) in enumerate(method_csvs.items(), 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"METHOD {method_idx}/{total_methods}: {method_name or '(unspecified)'}")
        logger.info(f"{'=' * 80}")

        try:
            df = load_similarity_csv(csv_paths, logger)
        except ValueError as e:
            logger.error(f"  {e}")
            continue

        run_ids = df["run_id"].unique()
        for run_id in run_ids:
            run_df = df[df["run_id"] == run_id]
            logger.info(f"\n  Run: {run_id} ({len(run_df):,} rows)")
            if compare or baseline_dir:
                all_loaded_dfs.append((method_name, run_id, run_df))

            # Full evaluation
            logger.info(f"  Full dataset evaluation...")
            full_results = evaluate_run(
                run_df, approved_by_indication,
                top_k=top_k, min_positives=min_positives,
                max_workers=max_workers, logger=logger,
            )
            full_overall = aggregate_results(
                full_results, method=method_name, run_id=run_id,
                mode="full", top_k=top_k,
            )
            all_summaries.append(full_overall)
            write_per_disease_csv(full_results, per_disease_path, method=method_name, run_id=run_id)

            n_valid = len([r for r in full_results if r.auroc is not None])
            logger.info(f"  Full: {n_valid} indications with valid AUROC")
            if full_overall.median_auroc is not None:
                logger.info(
                    f"  Full: Median AUROC={full_overall.median_auroc:.4f}, "
                    f"Mean AP={full_overall.mean_avg_precision:.4f}, "
                    f"Mean R@{top_k}={full_overall.mean_recall_at_k:.4f}"
                )

            # Cross-validation
            if cv > 0:
                logger.info(f"\n  {cv}-fold cross-validation...")
                fold_results = cross_validate(
                    run_df, approved_by_indication,
                    n_folds=cv, top_k=top_k, min_positives=min_positives,
                    max_workers=max_workers, seed=seed, logger=logger,
                )
                cv_overall = aggregate_cv_results(
                    fold_results, method=method_name, run_id=run_id, top_k=top_k,
                )
                all_summaries.append(cv_overall)

                # Write per-disease CV results
                for fold_result_list in fold_results:
                    write_per_disease_csv(
                        fold_result_list, per_disease_path,
                        method=method_name, run_id=run_id,
                    )

                if cv_overall.median_auroc is not None:
                    logger.info(
                        f"  CV: Median AUROC={cv_overall.median_auroc:.4f}, "
                        f"Mean AP={cv_overall.mean_avg_precision:.4f}, "
                        f"Mean R@{top_k}={cv_overall.mean_recall_at_k:.4f}"
                    )

    # 4. Comparative evaluation on common drug pool
    if compare:
        if len(all_loaded_dfs) < 2:
            logger.warning(
                "\n--compare requires at least 2 runs across all methods. "
                f"Found {len(all_loaded_dfs)}. Skipping comparative evaluation."
            )
        else:
            logger.info(f"\n{'=' * 80}")
            logger.info("COMPUTING COMPARATIVE EVALUATION (common drug pool)")
            logger.info(f"{'=' * 80}")
            common_drugs = compute_common_drug_pool(
                [df for _, _, df in all_loaded_dfs], logger
            )
            if not common_drugs:
                logger.warning("  No common drugs found across runs. Skipping.")
            else:
                for method_name, run_id, run_df in all_loaded_dfs:
                    restricted_df = run_df[run_df["drug_id"].isin(common_drugs)]
                    logger.info(
                        f"\n  Restricted evaluation: {method_name or run_id} "
                        f"({len(restricted_df):,} rows, "
                        f"{restricted_df['drug_id'].nunique()} drugs)"
                    )

                    # Full restricted
                    restricted_results = evaluate_run(
                        restricted_df, approved_by_indication,
                        top_k=top_k, min_positives=min_positives,
                        max_workers=max_workers, logger=logger,
                    )
                    restricted_overall = aggregate_results(
                        restricted_results, method=method_name, run_id=run_id,
                        mode="restricted", top_k=top_k,
                    )
                    all_summaries.append(restricted_overall)
                    write_per_disease_csv(
                        restricted_results, per_disease_path,
                        method=method_name, run_id=run_id,
                    )

                    n_valid = len([r for r in restricted_results if r.auroc is not None])
                    if restricted_overall.median_auroc is not None:
                        logger.info(
                            f"  Restricted full: {n_valid} indications, "
                            f"Median AUROC={restricted_overall.median_auroc:.4f}, "
                            f"Mean AP={restricted_overall.mean_avg_precision:.4f}, "
                            f"Mean R@{top_k}={restricted_overall.mean_recall_at_k:.4f}"
                        )

                    # CV restricted
                    if cv > 0:
                        logger.info(f"  {cv}-fold CV on restricted pool...")
                        fold_results = cross_validate(
                            restricted_df, approved_by_indication,
                            n_folds=cv, top_k=top_k, min_positives=min_positives,
                            max_workers=max_workers, seed=seed, logger=logger,
                        )
                        cv_restricted_overall = aggregate_cv_results(
                            fold_results, method=method_name, run_id=run_id,
                            top_k=top_k, mode="restricted_cv",
                        )
                        all_summaries.append(cv_restricted_overall)
                        for fold_result_list in fold_results:
                            write_per_disease_csv(
                                fold_result_list, per_disease_path,
                                method=method_name, run_id=run_id,
                            )
                        if cv_restricted_overall.median_auroc is not None:
                            logger.info(
                                f"  Restricted CV: "
                                f"Median AUROC={cv_restricted_overall.median_auroc:.4f}, "
                                f"Mean AP={cv_restricted_overall.mean_avg_precision:.4f}, "
                                f"Mean R@{top_k}={cv_restricted_overall.mean_recall_at_k:.4f}"
                            )

    # 5. Baseline comparison
    if baseline_dir:
        logger.info(f"\n{'=' * 80}")
        logger.info("LOADING BASELINE")
        logger.info(f"{'=' * 80}")
        logger.info(f"\nDiscovering baseline CSVs from: {baseline_dir}")
        baseline_method_csvs = discover_input_csvs(baseline_dir, logger)

        if not baseline_method_csvs:
            logger.warning("No baseline CSVs found. Skipping baseline comparison.")
        else:
            baseline_dfs: dict[str, pd.DataFrame] = {}
            for bmethod, bpaths in baseline_method_csvs.items():
                try:
                    bdf = load_similarity_csv(bpaths, logger)
                    # Use only the first run_id found in the baseline file
                    baseline_run_ids = bdf["run_id"].unique()
                    if len(baseline_run_ids) > 1:
                        logger.warning(
                            f"  Baseline method '{bmethod}' has {len(baseline_run_ids)} run_ids; "
                            f"using first: '{baseline_run_ids[0]}'"
                        )
                    baseline_dfs[bmethod] = bdf[bdf["run_id"] == baseline_run_ids[0]]
                except ValueError as e:
                    logger.warning(f"  Could not load baseline for method '{bmethod}': {e}")

            # Build method → [(run_id, df)] mapping from accumulated dfs
            method_run_dfs: dict[str, list[tuple[str, pd.DataFrame]]] = {}
            for mname, rid, rdf in all_loaded_dfs:
                method_run_dfs.setdefault(mname, []).append((rid, rdf))

            if not method_run_dfs:
                logger.warning(
                    "No test runs available for baseline comparison. "
                    "Use --input-dir or --input-csv to specify test runs."
                )
            else:
                logger.info(f"\n{'=' * 80}")
                logger.info("RUNNING BASELINE COMPARISONS")
                logger.info(f"{'=' * 80}")
                paired = compare_against_baseline(
                    method_run_dfs, baseline_dfs, approved_by_indication,
                    top_k=top_k, min_positives=min_positives,
                    cv=cv, seed=seed, max_workers=max_workers, logger=logger,
                )
                comparison_csv_path = os.path.join(output_dir, "comparison_table.csv")
                print_comparison_table(paired, logger, top_k=top_k)
                write_comparison_csv(paired, comparison_csv_path)
                logger.info(f"\n  Comparison table written to: {comparison_csv_path}")

    # 6. Write summary
    write_summary_csv(all_summaries, summary_path)
    print_summary(all_summaries, logger)

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"\nTotal time: {hours}h {minutes}m {seconds}s")
    logger.info(f"Output: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assess drug-disease prediction performance from diffusion profile similarities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single CSV, full evaluation
  python 4_assess_performance.py \\
      --input-csv results_comparison/within/run_abc.csv \\
      --reference data/6_drug_indication_df.tsv

  # Auto-discover all methods from --method all output
  python 4_assess_performance.py \\
      --input-dir results_comparison/ \\
      --reference data/6_drug_indication_df.tsv

  # With 5-fold cross-validation
  python 4_assess_performance.py \\
      --input-csv results_comparison/within/run_abc.csv \\
      --reference data/6_drug_indication_df.tsv \\
      --cv 5

  # Custom Recall@K threshold
  python 4_assess_performance.py \\
      --input-csv results.csv --reference ref.tsv \\
      --top-k 100 --min-positives 2
        """,
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-csv",
        nargs="+",
        help="Path(s) to similarity CSV file(s) from step 3",
    )
    input_group.add_argument(
        "--input-dir",
        help="Directory to auto-discover method_*/within/*.csv structure",
    )

    # Reference
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to approved drug-indication pairs (TSV/CSV, auto-detected separator)",
    )
    parser.add_argument(
        "--status-filter",
        nargs="+",
        default=None,
        metavar="STATUS",
        help=(
            "Only include reference pairs with these status values "
            "(e.g. --status-filter Approved). Only applied when the reference "
            "file has a 'status' column. Default: include all rows."
        ),
    )

    # Evaluation options
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="K for Recall@K metric (default: 50)",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=1,
        help="Minimum approved drugs per indication to include in evaluation (default: 1)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Number of cross-validation folds on drug set. 0 = disabled (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for cross-validation splits (default: 42)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="results_assessment",
        help="Output directory for results (default: results_assessment/)",
    )

    # Baseline comparison
    parser.add_argument(
        "--baseline-dir",
        default=None,
        metavar="DIR",
        help=(
            "Parent directory containing baseline comparison results "
            "(method_*/within/ structure, same as --input-dir). "
            "Each run from --input-dir is paired with the matching-method baseline "
            "and evaluated on their common drug pool. "
            "Generates a comparison table at the end of the report and writes "
            "comparison_table.csv to --output-dir."
        ),
    )

    # Comparative evaluation
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help=(
            "After per-method evaluation, re-evaluate all methods/runs on the "
            "common drug intersection for pool-size-invariant Recall@K comparison. "
            "Requires at least 2 runs (across methods or within one method)."
        ),
    )

    # Performance
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Process pool size for parallel metric computation (default: min(32, cpu_count))",
    )

    args = parser.parse_args()

    main(
        input_csv=args.input_csv,
        input_dir=args.input_dir,
        reference=args.reference,
        top_k=args.top_k,
        min_positives=args.min_positives,
        cv=args.cv,
        seed=args.seed,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        compare=args.compare,
        baseline_dir=args.baseline_dir,
        status_filter=args.status_filter,
    )
