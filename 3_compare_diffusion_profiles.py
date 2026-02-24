#!/usr/bin/env python3
"""Compare diffusion profiles across or within MSI runs.

Modes:
- within:  drug × indication similarity within a single run
- between: same entity's profile across different runs

Supports local and remote (SSH) profile directories.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import logging
import os

# Limit BLAS thread pool to 1 thread per worker — parallelism is managed
# by ThreadPoolExecutor via --max-workers, not by internal BLAS threading.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import typing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

import pickle

from diff_prof.compare import Method, diffusion_profile_similarity
from diff_prof.diffusion_profiles import DiffusionProfiles
from utils.interactive import select_runs_interactively
from utils.logging_setup import setup_logging
from utils.remote_sync import DownloadError, RemoteSync


# ---------------------------------------------------------------------------
# Data classes & helpers
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    """Metadata for a discovered diffusion-profile run."""

    run_id: str
    run_dir: str  # local path (may be a cache dir for remote runs)
    is_remote: bool = False
    remote_prefix: str = ""  # subdir under remote_path; "" if remote_path IS the run


class OutputWriter:
    """Buffered CSV writer with optional row-count splitting."""

    def __init__(
        self,
        output_dir: str,
        file_prefix: str,
        columns: list[str],
        max_rows_per_file: int = 0,
        logger: logging.Logger | None = None,
    ):
        self._output_dir = output_dir
        self._file_prefix = file_prefix.replace("/", "__")
        self._columns = columns
        self._max_rows = max_rows_per_file
        self._log = logger or logging.getLogger("compare_profiles")

        self._buffer: list[dict] = []
        self._total_written = 0
        self._file_index = 0
        self._flush_threshold = 10_000

        os.makedirs(output_dir, exist_ok=True)

    def add_row(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_threshold:
            self._write_buffer()

    def flush(self) -> None:
        if self._buffer:
            self._write_buffer()
        self._log.info(
            f"  Output: {self._total_written} rows -> "
            f"{self._file_index + (1 if self._total_written > 0 else 0)} file(s) "
            f"in {self._output_dir}"
        )

    # ------------------------------------------------------------------

    def _write_buffer(self) -> None:
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer, columns=self._columns)
        self._buffer.clear()

        # Check if we need to start a new part file
        if (
            self._max_rows > 0
            and self._total_written > 0
            and self._total_written // self._max_rows
            != (self._total_written + len(df)) // self._max_rows
        ):
            self._file_index += 1

        file_path = self._current_file_path()
        header = not os.path.exists(file_path)
        df.to_csv(file_path, mode="a", header=header, index=False)
        self._total_written += len(df)
        self._log.debug(f"  Flushed {len(df)} rows (total: {self._total_written:,})")

    def _current_file_path(self) -> str:
        if self._max_rows > 0:
            return os.path.join(
                self._output_dir,
                f"{self._file_prefix}_part{self._file_index:04d}.csv",
            )
        return os.path.join(self._output_dir, f"{self._file_prefix}.csv")


def _chunks(lst: list, size: int):
    """Yield successive chunks from list."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _clean_file_name(file_name: str) -> str:
    """Clean filename for profile loading (matches DiffusionProfiles.clean_file_name)."""
    return "".join(
        c for c in file_name if c.isalpha() or c.isdigit() or c == " " or c == "_"
    ).rstrip()


_PROFILE_SUFFIX = "_p_visit_array.npy"


def _infer_entity_lists(filenames: typing.Iterable[str]) -> tuple[list[str], list[str]]:
    """Infer drug/indication lists from ``.npy`` profile filenames.

    DrugBank IDs start with ``DB``; everything else is treated as an
    indication (UMLS CUI, typically ``C\\d+``).

    Args:
        filenames: Iterable of filenames (e.g. from ``os.listdir()``).

    Returns:
        (drugs, indications) — sorted lists of entity IDs.
    """
    drugs: list[str] = []
    indications: list[str] = []
    for fname in filenames:
        if not fname.endswith(_PROFILE_SUFFIX):
            continue
        entity_id = fname[: -len(_PROFILE_SUFFIX)]
        if entity_id.startswith("DB"):
            drugs.append(entity_id)
        else:
            indications.append(entity_id)
    return sorted(drugs), sorted(indications)


def _has_profiles(directory: str) -> bool:
    """Check whether a local directory contains any .npy profile files."""
    try:
        return any(f.endswith(".npy") for f in os.listdir(directory))
    except OSError:
        return False


def _load_run_metadata(run_dir: str) -> dict:
    """Load node2idx and drug/indication lists from a run directory.

    Falls back gracefully when metadata pkl files are missing:
    - ``node2idx.pkl`` missing → ``node2idx``/``nodelist`` set to ``None``
    - ``drugs_indications_lists.pkl`` missing → inferred from ``.npy`` filenames

    Returns dict with keys: node2idx, nodelist, drugs_in_graph,
    indications_in_graph.
    """
    node2idx = None
    nodelist = None
    node2idx_path = os.path.join(run_dir, "node2idx.pkl")
    if os.path.exists(node2idx_path):
        with open(node2idx_path, "rb") as f:
            node2idx = pickle.load(f)
        idx2node = {v: k for k, v in node2idx.items()}
        nodelist = [idx2node[i] for i in range(len(idx2node))]

    drugs = None
    indications = None
    lists_path = os.path.join(run_dir, "drugs_indications_lists.pkl")
    if os.path.exists(lists_path):
        with open(lists_path, "rb") as f:
            lists_data = pickle.load(f)
        drugs = lists_data["drugs_in_graph"]
        indications = lists_data["indications_in_graph"]

    # Fallback: infer entity lists from .npy filenames
    if drugs is None or indications is None:
        drugs, indications = _infer_entity_lists(os.listdir(run_dir))

    return {
        "node2idx": node2idx,
        "nodelist": nodelist,
        "drugs_in_graph": drugs,
        "indications_in_graph": indications,
    }


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_local_runs(
    runs_dir: list[str] | None,
    runs_root: str | None,
    logger: logging.Logger,
) -> list[RunInfo]:
    """Discover valid run directories from local paths.

    A directory is considered a run if it contains ``.npy`` profile files
    (``node2idx.pkl`` is no longer required).
    """
    runs: list[RunInfo] = []

    if runs_dir:
        for d in runs_dir:
            d = d.rstrip("/")
            run_id = os.path.basename(d)
            if _has_profiles(d):
                runs.append(RunInfo(run_id=run_id, run_dir=d))
            else:
                logger.warning(f"Skipping {d}: no .npy profiles found")

    if runs_root:
        runs_root = runs_root.rstrip("/")
        if not os.path.isdir(runs_root):
            logger.warning(f"Runs root does not exist: {runs_root}")
            return runs
        for entry in sorted(os.listdir(runs_root)):
            candidate = os.path.join(runs_root, entry)
            if not os.path.isdir(candidate):
                continue
            if _has_profiles(candidate):
                runs.append(RunInfo(run_id=entry, run_dir=candidate))
            else:
                # Maybe a category folder — scan one level deeper
                for sub_entry in sorted(os.listdir(candidate)):
                    sub_candidate = os.path.join(candidate, sub_entry)
                    if os.path.isdir(sub_candidate) and _has_profiles(sub_candidate):
                        runs.append(RunInfo(
                            run_id=f"{entry}/{sub_entry}",
                            run_dir=sub_candidate,
                        ))

    return runs


def discover_remote_runs(
    remote_sync: RemoteSync,
    cache_dir: str,
    logger: logging.Logger,
) -> list[RunInfo]:
    """Discover runs on remote and download metadata to local cache.

    Supports three remote layouts:
    - remote_path IS a run directory (contains profiles directly)
    - remote_path is a root with run subdirectories
    - remote_path has category folders (e.g. docking/, dta_atlas/) each
      containing run subdirectories

    Metadata pkl files are downloaded when available but not required.
    When ``drugs_indications_lists.pkl`` is missing, entity lists are
    inferred from remote ``.npy`` filenames.
    """
    # Scan remote for run directories (may be slow over SSH)
    logger.info("Scanning remote for run directories...")
    run_ids = remote_sync.list_remote_subdirs(recursive=True)

    if not run_ids:
        # Maybe remote_path IS a single run (profiles live at root level)
        remote_files = remote_sync.list_remote_dir()
        if any(f.endswith(".npy") for f in remote_files):
            run_id = os.path.basename(remote_sync.remote_path.rstrip("/"))
            run_ids = [run_id]
            run_id_to_prefix = {run_id: ""}
            logger.info(f"Remote path is itself a run directory: {run_id}")
        else:
            logger.info("Found 0 run(s) on remote")
            return []
    else:
        run_id_to_prefix = {rid: rid for rid in run_ids}
        logger.info(f"Found {len(run_ids)} run directory(ies) on remote")

    runs: list[RunInfo] = []
    for run_idx, run_id in enumerate(run_ids, 1):
        logger.info(f"  Caching metadata for {run_id} ({run_idx}/{len(run_ids)})...")
        local_run_dir = os.path.join(cache_dir, run_id)
        os.makedirs(local_run_dir, exist_ok=True)

        prefix = run_id_to_prefix[run_id]

        # Try to download metadata pkl files (optional — check existence first
        # to avoid costly 3-retry cycles for files that simply don't exist)
        for artifact in ("node2idx.pkl", "drugs_indications_lists.pkl"):
            local_path = os.path.join(local_run_dir, artifact)
            if not os.path.exists(local_path):
                remote_subpath = os.path.join(prefix, artifact) if prefix else artifact
                remote_full = os.path.join(remote_sync.remote_path, remote_subpath)
                if not remote_sync.remote_file_exists(remote_full):
                    continue
                try:
                    remote_sync.download_file(remote_subpath, local_path)
                except DownloadError:
                    pass

        # If entity lists are still missing, infer from remote .npy filenames
        lists_path = os.path.join(local_run_dir, "drugs_indications_lists.pkl")
        if not os.path.exists(lists_path):
            try:
                remote_files = remote_sync.list_remote_dir(prefix)
            except Exception:
                logger.warning(f"    Cannot list remote files for {run_id}, skipping")
                continue
            drugs, indications = _infer_entity_lists(remote_files)
            if not drugs and not indications:
                logger.warning(
                    f"    Skipping {run_id}: no profiles and no metadata found"
                )
                continue
            logger.info(
                f"    Inferred {len(drugs)} drugs, {len(indications)} indications "
                f"from filenames"
            )
            with open(lists_path, "wb") as f:
                pickle.dump(
                    {"drugs_in_graph": drugs, "indications_in_graph": indications}, f
                )

        has_node2idx = os.path.exists(os.path.join(local_run_dir, "node2idx.pkl"))
        if not has_node2idx:
            logger.info(f"    Note: no node2idx.pkl (between-model comparison unavailable)")

        runs.append(RunInfo(
            run_id=run_id, run_dir=local_run_dir,
            is_remote=True, remote_prefix=prefix,
        ))

    logger.info(f"{len(runs)} of {len(run_ids)} run(s) usable")
    return runs


def ensure_profiles_local(
    remote_sync: RemoteSync | None,
    run: RunInfo,
    entity_ids: list[str],
    logger: logging.Logger,
) -> None:
    """Download profile .npy files for given entities if not already cached."""
    if remote_sync is None or not run.is_remote:
        return

    # Pre-scan to find which profiles need downloading
    missing: list[tuple[str, str, str]] = []  # (entity_id, filename, local_path)
    for entity_id in entity_ids:
        clean_name = _clean_file_name(entity_id)
        filename = f"{clean_name}_p_visit_array.npy"
        local_path = os.path.join(run.run_dir, filename)
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            missing.append((entity_id, filename, local_path))

    if not missing:
        return

    downloaded = 0
    for entity_id, filename, local_path in tqdm(
        missing,
        desc=f"    Downloading ({run.run_id})",
        unit="file",
        leave=False,
    ):
        try:
            remote_subpath = os.path.join(run.remote_prefix, filename) if run.remote_prefix else filename
            remote_sync.download_file(remote_subpath, local_path)
            downloaded += 1
        except DownloadError:
            logger.debug(f"Profile not available for {entity_id} in {run.run_id}")
    logger.info(f"    Downloaded {downloaded}/{len(missing)} profiles for {run.run_id}")


# ---------------------------------------------------------------------------
# Reference / approved-pairs helpers
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


def load_approved_pairs(
    path: str,
    status_filter: list[str] | None,
    logger: logging.Logger,
) -> set[tuple[str, str]]:
    """Load a set of (drug_id, indication_id) tuples from a reference file.

    Accepts TSV or CSV; auto-detects separator. Column names 'drug'/'indication'
    or 'drug_id'/'ind_id' are both accepted. When *status_filter* is provided
    and the file has a 'status' column, only rows with matching values are kept.
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
                f"  Approved pairs status filter {status_filter}: "
                f"kept {n_after:,} / {n_before:,} rows"
            )
        else:
            logger.warning(
                "  --status-filter provided but approved-pairs file has no "
                "'status' column — filter ignored."
            )

    df = df[["drug", "indication"]].dropna()
    pairs = set(zip(df["drug"], df["indication"]))
    logger.info(
        f"  Loaded {len(pairs):,} approved (drug, indication) pairs from {path}"
    )
    return pairs


# ---------------------------------------------------------------------------
# Comparison: within-model
# ---------------------------------------------------------------------------

WITHIN_COLUMNS = [
    "run_id", "drug_id", "indication_id",
    "similarity", "l2_distance", "n_nodes",
]


def _compare_pair_worker(args):
    """Worker function for parallel comparison (picklable for ProcessPoolExecutor).

    Args:
        args: tuple of (drug_id, indication_id, drug_profile, indication_profile,
                        method, normalization, run_id, n_nodes)
    """
    drug_id, ind_id, drug_profile, ind_profile, method, norm, run_id, n_nodes = args

    sim, diag = diffusion_profile_similarity(
        drug_profile,
        ind_profile,
        method=method,
        align="position",
        normalization=norm,
        return_diagnostics=True,
    )
    return {
        "run_id": run_id,
        "drug_id": drug_id,
        "indication_id": ind_id,
        "similarity": sim,
        "l2_distance": diag.get("l2_distance"),
        "n_nodes": n_nodes,
    }


def compare_within(
    runs: list[RunInfo],
    method: str,
    normalization: str | None,
    chunk_size: int,
    max_workers: int,
    output_dir: str,
    max_rows_per_file: int,
    remote_sync: RemoteSync | None,
    logger: logging.Logger,
    approved_pairs: set[tuple[str, str]] | None = None,
) -> None:
    """Drug × indication similarity within each run."""
    within_dir = os.path.join(output_dir, "within")

    for run_idx, run in enumerate(runs, 1):
        logger.info(f"\n[{run_idx}/{len(runs)}] Within-model: {run.run_id}")

        msi_data = _load_run_metadata(run.run_dir)
        drugs = msi_data["drugs_in_graph"]
        indications = msi_data["indications_in_graph"]
        n_nodes = len(msi_data["nodelist"]) if msi_data["nodelist"] is not None else None

        logger.info(
            f"  {len(drugs)} drugs x {len(indications)} indications "
            f"= {len(drugs) * len(indications)} pairs"
        )

        dp = DiffusionProfiles(
            alpha=None, max_iter=None, tol=None, weights=None,
            num_cores=None, save_load_file_path=run.run_dir,
        )

        # Load ALL indications once (reused across drug chunks)
        ensure_profiles_local(remote_sync, run, indications, logger)
        dp.load_diffusion_profiles(indications)
        indication_profiles = dict(dp.drug_or_indication2diffusion_profile)

        available_indications = [i for i in indications if i in indication_profiles]
        if not available_indications:
            logger.warning(f"  No indication profiles available. Skipping.")
            continue

        # Infer n_nodes from profile length if metadata was missing
        if n_nodes is None and indication_profiles:
            first_profile = next(iter(indication_profiles.values()))
            n_nodes = len(first_profile)

        writer = OutputWriter(
            within_dir, run.run_id, WITHIN_COLUMNS,
            max_rows_per_file=max_rows_per_file, logger=logger,
        )

        norm = None if normalization == "none" else normalization
        total_chunks = (len(drugs) + chunk_size - 1) // chunk_size
        pairs_done = 0

        for chunk_idx, drug_chunk in enumerate(_chunks(drugs, chunk_size), 1):
            logger.info(f"  Chunk {chunk_idx}/{total_chunks}: {len(drug_chunk)} drugs...")
            ensure_profiles_local(remote_sync, run, drug_chunk, logger)
            dp.load_diffusion_profiles(drug_chunk)
            drug_profiles = {
                k: v for k, v in dp.drug_or_indication2diffusion_profile.items()
                if k in set(drug_chunk)
            }

            available_drugs = [d for d in drug_chunk if d in drug_profiles]
            if not available_drugs:
                continue

            # Prepare arguments for parallel processing
            # Format: (drug_id, ind_id, drug_profile, ind_profile, method, norm, run_id, n_nodes)
            worker_args = [
                (d, i, drug_profiles[d], indication_profiles[i], method, norm, run.run_id, n_nodes)
                for d in available_drugs
                for i in available_indications
                if approved_pairs is None or (d, i) in approved_pairs
            ]

            # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for result in ex.map(_compare_pair_worker, worker_args, chunksize=100):
                    writer.add_row(result)

            pairs_done += len(worker_args)
            total_pairs = len(drugs) * len(available_indications)
            logger.info(
                f"  Processed {pairs_done:,}/{total_pairs:,} pairs "
                f"({100 * pairs_done // total_pairs}%)"
            )

            dp.drug_or_indication2diffusion_profile = {}
            gc.collect()

        writer.flush()
        del dp, indication_profiles
        gc.collect()


# ---------------------------------------------------------------------------
# Comparison: between-model
# ---------------------------------------------------------------------------

BETWEEN_COLUMNS = [
    "entity_id", "entity_type", "run_a", "run_b",
    "similarity", "l2_distance", "n_common_nodes",
]


def _compare_between_worker(args):
    """Worker function for between-model parallel comparison (picklable for ProcessPoolExecutor).

    Args:
        args: tuple of (entity_id, entity_type, profile_a_aligned, profile_b_aligned,
                        method, normalization, run_a_id, run_b_id, n_common)
    """
    eid, etype, profile_a, profile_b, method, norm, run_a_id, run_b_id, n_common = args

    sim, diag = diffusion_profile_similarity(
        profile_a, profile_b,
        method=method,
        align="position",
        normalization=norm,
        return_diagnostics=True,
    )
    return {
        "entity_id": eid,
        "entity_type": etype,
        "run_a": run_a_id,
        "run_b": run_b_id,
        "similarity": sim,
        "l2_distance": diag.get("l2_distance"),
        "n_common_nodes": n_common,
    }


def compare_between(
    runs: list[RunInfo],
    reference_dir: str | None,
    entity_type: str,
    method: str,
    normalization: str | None,
    chunk_size: int,
    max_workers: int,
    output_dir: str,
    max_rows_per_file: int,
    remote_sync: RemoteSync | None,
    logger: logging.Logger,
) -> None:
    """Same entity across different runs."""
    between_dir = os.path.join(output_dir, "between")

    # Build run pairs
    if reference_dir:
        ref_dir_norm = os.path.normpath(reference_dir)
        ref_run = None
        other_runs = []
        for r in runs:
            if os.path.normpath(r.run_dir) == ref_dir_norm:
                ref_run = r
            else:
                other_runs.append(r)
        if ref_run is None:
            logger.error(f"Reference run not found among discovered runs: {reference_dir}")
            return
        run_pairs = [(ref_run, other) for other in other_runs]
        logger.info(f"Reference: {ref_run.run_id} vs {len(other_runs)} other run(s)")
    else:
        run_pairs = list(itertools.combinations(runs, 2))
        logger.info(f"Pairwise: {len(run_pairs)} pair(s) from {len(runs)} runs")

    if not run_pairs:
        logger.warning("No run pairs to compare.")
        return

    norm = None if normalization == "none" else normalization

    for pair_idx, (run_a, run_b) in enumerate(run_pairs, 1):
        logger.info(f"\n[{pair_idx}/{len(run_pairs)}] {run_a.run_id} vs {run_b.run_id}")

        msi_data_a = _load_run_metadata(run_a.run_dir)
        msi_data_b = _load_run_metadata(run_b.run_dir)

        # Between-model comparison requires node2idx for alignment
        node2idx_a = msi_data_a["node2idx"]
        node2idx_b = msi_data_b["node2idx"]
        if node2idx_a is None or node2idx_b is None:
            missing = []
            if node2idx_a is None:
                missing.append(run_a.run_id)
            if node2idx_b is None:
                missing.append(run_b.run_id)
            logger.warning(
                f"  Skipping pair: node2idx.pkl missing for {', '.join(missing)} "
                f"(required for between-model alignment)"
            )
            continue

        # Precompute intersection indices ONCE per pair
        common_nodes = [n for n in msi_data_a["nodelist"] if n in node2idx_b]
        n_common = len(common_nodes)

        if n_common == 0:
            logger.warning(f"  No common nodes. Skipping pair.")
            continue

        idx_a = np.fromiter((node2idx_a[n] for n in common_nodes), dtype=np.int64)
        idx_b = np.fromiter((node2idx_b[n] for n in common_nodes), dtype=np.int64)

        # Find common entities
        entities: list[tuple[str, str]] = []
        if entity_type in ("drug", "both"):
            common_drugs = sorted(
                set(msi_data_a["drugs_in_graph"]) & set(msi_data_b["drugs_in_graph"])
            )
            entities.extend((d, "drug") for d in common_drugs)
        if entity_type in ("indication", "both"):
            common_indications = sorted(
                set(msi_data_a["indications_in_graph"])
                & set(msi_data_b["indications_in_graph"])
            )
            entities.extend((i, "indication") for i in common_indications)

        logger.info(
            f"  common_nodes={n_common}, entities_to_compare={len(entities)}"
        )

        if not entities:
            logger.warning(f"  No common entities. Skipping pair.")
            continue

        dp_a = DiffusionProfiles(
            alpha=None, max_iter=None, tol=None, weights=None,
            num_cores=None, save_load_file_path=run_a.run_dir,
        )
        dp_b = DiffusionProfiles(
            alpha=None, max_iter=None, tol=None, weights=None,
            num_cores=None, save_load_file_path=run_b.run_dir,
        )

        writer = OutputWriter(
            between_dir,
            f"{run_a.run_id}_vs_{run_b.run_id}",
            BETWEEN_COLUMNS,
            max_rows_per_file=max_rows_per_file,
            logger=logger,
        )

        entity_ids = [e[0] for e in entities]
        entity_type_map = {e[0]: e[1] for e in entities}
        total_chunks = (len(entity_ids) + chunk_size - 1) // chunk_size
        entities_done = 0

        for chunk_idx, chunk_entities in enumerate(_chunks(entity_ids, chunk_size), 1):
            logger.info(f"  Chunk {chunk_idx}/{total_chunks}: {len(chunk_entities)} entities...")
            ensure_profiles_local(remote_sync, run_a, chunk_entities, logger)
            ensure_profiles_local(remote_sync, run_b, chunk_entities, logger)
            dp_a.load_diffusion_profiles(chunk_entities)
            dp_b.load_diffusion_profiles(chunk_entities)

            profiles_a = dp_a.drug_or_indication2diffusion_profile
            profiles_b = dp_b.drug_or_indication2diffusion_profile

            # Prepare arguments for parallel processing
            # Filter and align profiles, then format for worker
            worker_args = []
            for eid in chunk_entities:
                if eid in profiles_a and eid in profiles_b:
                    a_aligned = profiles_a[eid][idx_a]
                    b_aligned = profiles_b[eid][idx_b]
                    worker_args.append((
                        eid, entity_type_map[eid], a_aligned, b_aligned,
                        method, norm, run_a.run_id, run_b.run_id, n_common
                    ))

            # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for result in ex.map(_compare_between_worker, worker_args, chunksize=50):
                    writer.add_row(result)

            entities_done += len(chunk_entities)
            logger.info(
                f"  Processed {entities_done:,}/{len(entity_ids):,} entities "
                f"({100 * entities_done // len(entity_ids)}%)"
            )

            dp_a.drug_or_indication2diffusion_profile = {}
            dp_b.drug_or_indication2diffusion_profile = {}
            gc.collect()

        writer.flush()
        del dp_a, dp_b
        gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    mode: str,
    runs_dir: list[str] | None = None,
    runs_root: str | None = None,
    reference: str | None = None,
    entity_type: str = "both",
    method: str = "l2",
    normalization: str = "none",
    output_dir: str = "results_comparison",
    max_rows_per_file: int = 0,
    chunk_size: int = 300,
    max_workers: int | None = None,
    interactive: bool = False,
    remote_host: str | None = None,
    remote_port: int = 22,
    remote_user: str | None = None,
    remote_path: str | None = None,
    ssh_key: str | None = None,
    cache_dir: str | None = None,
    ssh_timeout: int = 30,
    clear_cache: bool = False,
    approved_pairs_path: str | None = None,
    status_filter: list[str] | None = None,
) -> None:
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 1)

    if cache_dir is None:
        cache_dir = os.path.join(output_dir, ".cache")

    if clear_cache and os.path.isdir(cache_dir):
        import glob as _glob

        stale = _glob.glob(os.path.join(cache_dir, "**", "*.npy"), recursive=True)
        for f in stale:
            os.remove(f)
        if stale:
            # Use print since logger isn't set up yet
            print(f"Cleared {len(stale)} cached .npy file(s) from {cache_dir}")

    logger = setup_logging(
        output_dir,
        log_name="compare_profiles",
        logger_name="compare_profiles",
    )

    logger.info("=" * 80)
    logger.info("DIFFUSION PROFILE COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Command: {' '.join(sys.argv)}")

    logger.info(f"\nConfiguration:")
    logger.info(f"  Mode:             {mode}")
    logger.info(f"  Method:           {method}")
    logger.info(f"  Normalization:    {normalization}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Chunk size:       {chunk_size}")
    logger.info(f"  Max workers:      {max_workers}")
    if mode == "between":
        logger.info(f"  Entity type:      {entity_type}")
        if reference:
            logger.info(f"  Reference:        {reference}")
        else:
            logger.info(f"  Reference:        (none, pairwise)")
    if max_rows_per_file > 0:
        logger.info(f"  Max rows/file:    {max_rows_per_file}")
    if approved_pairs_path:
        logger.info(f"  Approved pairs:   {approved_pairs_path}")
        if status_filter:
            logger.info(f"  Status filter:    {status_filter}")

    # Load approved pairs filter (within-mode only)
    approved_pairs: set[tuple[str, str]] | None = None
    if approved_pairs_path:
        logger.info(f"\nLoading approved pairs filter...")
        approved_pairs = load_approved_pairs(approved_pairs_path, status_filter, logger)

    # Setup remote if configured
    remote_sync: RemoteSync | None = None

    if remote_host and remote_user and remote_path:
        logger.info(f"\nRemote input:")
        logger.info(f"  Host:             {remote_host}:{remote_port}")
        logger.info(f"  User:             {remote_user}")
        logger.info(f"  Path:             {remote_path}")
        logger.info(f"  Cache:            {cache_dir}")
        logger.info(f"  SSH timeout:      {ssh_timeout}s")

        try:
            remote_sync = RemoteSync(
                host=remote_host,
                port=remote_port,
                username=remote_user,
                remote_path=remote_path,
                key_path=ssh_key,
                logger=logger,
                timeout=ssh_timeout,
            )
            logger.info("\nConnecting to remote server...")
            remote_sync.connect()
            logger.info("Remote connection established.")
        except Exception as e:
            logger.error(f"Failed to connect to remote server: {e}")
            logger.error("Aborting. Check SSH configuration.")
            return

    try:
        # Discover runs
        runs: list[RunInfo] = []

        local_runs = discover_local_runs(runs_dir, runs_root, logger)
        if local_runs:
            logger.info(f"\nFound {len(local_runs)} local run(s)")
        runs.extend(local_runs)

        if remote_sync:
            remote_runs = discover_remote_runs(remote_sync, cache_dir, logger)
            if remote_runs:
                logger.info(f"Found {len(remote_runs)} remote run(s)")
            # Merge, avoiding duplicates by run_id
            existing_ids = {r.run_id for r in runs}
            for rr in remote_runs:
                if rr.run_id not in existing_ids:
                    runs.append(rr)

        if not runs:
            logger.error("No valid runs discovered. Provide --runs-dir, --runs-root, or remote options.")
            return

        logger.info(f"\nTotal runs available: {len(runs)}")
        for r in runs:
            tag = " (remote)" if r.is_remote else ""
            logger.info(f"  - {r.run_id}{tag}")

        # Interactive selection
        if interactive and len(runs) > 1:
            run_labels = [
                f"{r.run_id}{' (remote)' if r.is_remote else ''}" for r in runs
            ]
            selected = select_runs_interactively(run_labels, "runs for comparison")
            if not selected:
                logger.info("No runs selected. Exiting.")
                return
            selected_set = set(selected)
            runs = [r for r, label in zip(runs, run_labels) if label in selected_set]
            logger.info(f"Selected {len(runs)} run(s)")

        # Validate
        if mode == "within" and len(runs) < 1:
            logger.error("Within-model comparison requires at least 1 run.")
            return
        if mode == "between" and len(runs) < 2:
            logger.error("Between-model comparison requires at least 2 runs.")
            return

        # Determine which methods to run
        if method == "all":
            methods_to_run = list(typing.get_args(Method))
            logger.info(f"\nRunning all comparison methods: {methods_to_run}")
        else:
            methods_to_run = [method]

        # Dispatch - loop through all requested methods
        for method_idx, current_method in enumerate(methods_to_run, 1):
            if len(methods_to_run) > 1:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"METHOD {method_idx}/{len(methods_to_run)}: {current_method.upper()}")
                logger.info(f"{'=' * 80}")

            # Create method-specific output directory when running all methods
            if method == "all":
                method_output_dir = os.path.join(output_dir, f"method_{current_method}")
            else:
                method_output_dir = output_dir

            if mode == "within":
                compare_within(
                    runs=runs,
                    method=current_method,
                    normalization=normalization,
                    chunk_size=chunk_size,
                    max_workers=max_workers,
                    output_dir=method_output_dir,
                    max_rows_per_file=max_rows_per_file,
                    remote_sync=remote_sync,
                    logger=logger,
                    approved_pairs=approved_pairs,
                )
            elif mode == "between":
                compare_between(
                    runs=runs,
                    reference_dir=reference,
                    entity_type=entity_type,
                    method=current_method,
                    normalization=normalization,
                    chunk_size=chunk_size,
                    max_workers=max_workers,
                    output_dir=method_output_dir,
                    max_rows_per_file=max_rows_per_file,
                    remote_sync=remote_sync,
                    logger=logger,
                )

    except DownloadError as e:
        logger.error(f"\nDownload failed: {e}")
        raise
    except KeyboardInterrupt:
        logger.warning("\nComparison interrupted by user.")
        raise
    finally:
        if remote_sync:
            remote_sync.close()
            logger.info("Remote connection closed.")

    # Summary
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {hours}h {minutes}m {seconds}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    method_choices = ["all"] + list(typing.get_args(Method))

    parser = argparse.ArgumentParser(
        description="Compare diffusion profiles across or within MSI runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Within-model: drug x indication similarity
  python 3_compare_diffusion_profiles.py --mode within \\
      --runs-dir results_filtered/dta_atlas/1_drug_to_protein_gat_gcn_ic50_cutoff_5

  # Run all comparison methods at once
  python 3_compare_diffusion_profiles.py --mode within \\
      --runs-dir results/ --method all --output-dir comparisons/

  # Specific method (cosine similarity)
  python 3_compare_diffusion_profiles.py --mode within \\
      --runs-dir results/ --method cosine

  # Between-model: pairwise across all runs in a directory
  python 3_compare_diffusion_profiles.py --mode between \\
      --runs-root results_filtered/dta_atlas/

  # Between-model with reference run
  python 3_compare_diffusion_profiles.py --mode between \\
      --reference results/ --runs-root results_filtered/dta_atlas/

  # Remote runs with interactive selection
  python 3_compare_diffusion_profiles.py --mode between \\
      --remote-host myserver.edu --remote-port 50002 \\
      --remote-user onur --remote-path /data/mi_results \\
      --interactive --chunk-size 100

  # Within-model on remote with custom output
  python 3_compare_diffusion_profiles.py --mode within \\
      --remote-host myserver.edu --remote-user onur \\
      --remote-path /data/mi_results --output-dir /tmp/comparisons
        """,
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["within", "between"],
        required=True,
        help="Comparison mode: 'within' (drug x indication per run) or 'between' (same entity across runs)",
    )

    # Run selection
    parser.add_argument(
        "--runs-dir",
        nargs="+",
        help="Explicit local run directories to compare",
    )
    parser.add_argument(
        "--runs-root",
        help="Root directory to scan for run subdirectories containing .npy profiles",
    )

    # Between-mode options
    parser.add_argument(
        "--reference",
        help="Reference run directory for between-mode (one-vs-many). Without this, all pairs are compared.",
    )
    parser.add_argument(
        "--entity-type",
        choices=["drug", "indication", "both"],
        default="both",
        help="Which entity types to compare in between-mode (default: both)",
    )

    # Within-mode filtering
    parser.add_argument(
        "--approved-pairs",
        default=None,
        metavar="FILE",
        help=(
            "Optional CSV/TSV of known drug-indication pairs. "
            "When provided, only (drug, indication) pairs present in this file "
            "are written to the within-mode output. "
            "Accepts 'drug'+'indication' or 'drug_id'+'ind_id' column names."
        ),
    )
    parser.add_argument(
        "--status-filter",
        nargs="+",
        default=None,
        metavar="STATUS",
        help=(
            "Only include pairs with these status values from --approved-pairs "
            "(e.g. --status-filter Approved). Ignored if --approved-pairs is not set. "
            "Only applied when the file has a 'status' column."
        ),
    )

    # Comparison options
    parser.add_argument(
        "--method",
        choices=method_choices,
        default="l2",
        help=f"Similarity method: 'all' runs all methods, or choose one (default: l2, available: {method_choices})",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "l1", "l2"],
        default="none",
        help="Normalization before comparison (default: none)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="results_comparison",
        help="Output directory for CSV results (default: results_comparison/)",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Split output CSVs at this row count (default: 0 = no split)",
    )

    # Performance
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Number of profiles loaded per chunk (default: 300)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Thread pool size (default: min(32, cpu_count))",
    )

    # Interactive
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactively select which runs to compare",
    )

    # Remote input
    parser.add_argument("--remote-host", help="SSH host where profiles are stored")
    parser.add_argument("--remote-port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--remote-user", help="SSH username")
    parser.add_argument("--remote-path", help="Remote base path containing run subdirectories")
    parser.add_argument("--ssh-key", help="Path to SSH private key")
    parser.add_argument(
        "--ssh-timeout",
        type=int,
        default=30,
        help="SSH/SFTP operation timeout in seconds (default: 30). 0 = no timeout.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Local cache directory for downloaded remote profiles (default: {output-dir}/.cache/)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all cached .npy files before starting (keeps .pkl metadata). "
             "Use to recover from corrupt downloads.",
    )

    args = parser.parse_args()

    # Validate remote options
    remote_options = [args.remote_host, args.remote_user, args.remote_path]
    if any(remote_options) and not all(remote_options):
        parser.error("--remote-host, --remote-user, and --remote-path must all be specified together")

    # Validate at least one source of runs
    if not args.runs_dir and not args.runs_root and not args.remote_host:
        parser.error("Provide at least one of: --runs-dir, --runs-root, or remote options")

    main(
        mode=args.mode,
        runs_dir=args.runs_dir,
        runs_root=args.runs_root,
        reference=args.reference,
        entity_type=args.entity_type,
        method=args.method,
        normalization=args.normalization,
        output_dir=args.output_dir,
        max_rows_per_file=args.max_rows_per_file,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        interactive=args.interactive,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_user=args.remote_user,
        remote_path=args.remote_path,
        ssh_key=args.ssh_key,
        cache_dir=args.cache_dir,
        ssh_timeout=args.ssh_timeout,
        clear_cache=args.clear_cache,
        approved_pairs_path=args.approved_pairs,
        status_filter=args.status_filter,
    )
