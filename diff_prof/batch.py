from __future__ import annotations

import gc
import glob
import logging
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, TYPE_CHECKING

from msi.msi import MSI

if TYPE_CHECKING:
    from utils.remote_sync import RemoteSync

from .defaults import (
	DEFAULT_ALPHA,
	DEFAULT_MAX_ITER,
	DEFAULT_TOL,
	DEFAULT_WEIGHTS,
	default_num_cores,
)
from .diffusion_profiles import DiffusionProfiles


@dataclass(frozen=True)
class DiffusionRunMeta:
	"""Lightweight metadata for a diffusion-profile run.

	Diffusion profiles live on disk under `save_load_file_path`.
	Use the original MSI developers' pattern to load selectively:

	- dp = DiffusionProfiles(..., save_load_file_path=run.save_load_file_path)
	- msi = MSI(drug2protein_file_path=run.drug2protein_file_path, ...); msi.load()
	- msi.load_saved_node_idx_mapping_and_nodelist(dp.save_load_file_path)
	- dp.load_diffusion_profiles(selected_nodes)
	"""

	run_id: str
	drug2protein_file_path: str
	save_load_file_path: str
	computed: bool


def _safe_run_id_from_path(path: str) -> str:
	base = os.path.basename(path)
	stem = os.path.splitext(base)[0]
	# keep reasonably filesystem-safe
	return "".join(c for c in stem if (c.isalnum() or c in {"-", "_"}))


def _run_has_saved_artifacts(run_dir: str) -> bool:
	return (
		os.path.exists(os.path.join(run_dir, "node2idx.pkl"))
		and os.path.exists(os.path.join(run_dir, "graph.pkl"))
	)


def _get_expected_profile_count(run_dir: str) -> int | None:
	"""Determine expected profile count from saved artifacts.

	Works with both new format (worker data files) and old format (graph.pkl only).
	"""
	# Try new format first (faster, doesn't load full graph)
	try:
		msi_data = MSI.load_worker_data(run_dir)
		return len(msi_data['drugs_in_graph']) + len(msi_data['indications_in_graph'])
	except FileNotFoundError:
		pass

	# Fall back to old format: load graph and count drug/indication nodes
	graph_path = os.path.join(run_dir, "graph.pkl")
	if not os.path.exists(graph_path):
		return None

	try:
		import pickle
		with open(graph_path, "rb") as f:
			graph = pickle.load(f)

		# Count nodes by type
		count = 0
		for node in graph.nodes():
			node_type = graph.nodes[node].get("type")
			if node_type in ("drug", "indication"):
				count += 1
		return count
	except Exception:
		return None


def _count_existing_profiles(run_dir: str) -> int:
	"""Count existing profile .npy files in run directory."""
	profiles = glob.glob(os.path.join(run_dir, "*_p_visit_array.npy"))
	return len(profiles)


def _run_is_complete(
	run_dir: str,
	remote_sync: "RemoteSync | None" = None,
	run_id: str | None = None,
) -> bool:
	"""Check if run has all expected profiles (complete computation).

	Checks local first, then remote if available.

	Args:
		run_dir: Local directory for this run
		remote_sync: Optional RemoteSync instance for checking remote
		run_id: Run ID for remote path construction (required if remote_sync provided)

	Returns:
		True if run is complete (locally or remotely)
	"""
	# Try to get expected count from local artifacts
	expected = None
	if _run_has_saved_artifacts(run_dir):
		expected = _get_expected_profile_count(run_dir)

	# If no local artifacts, we can't determine expected count
	# (Remote-only completion check would require remote artifacts)
	if expected is None:
		return False

	# Check local first
	local_count = _count_existing_profiles(run_dir)
	if local_count == expected:
		return True

	# Check remote if available
	if remote_sync and run_id:
		try:
			remote_count = remote_sync.count_remote_profiles(run_id)
			if remote_count == expected:
				return True
		except Exception:
			pass  # Fall back to local-only checking

	return False


def compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs(
	*,
	save_root: str,
	drug2protein_glob: str = "data/1_drug_to_protein_filtered_*.tsv",
	drug2protein_file_paths: Sequence[str] | None = None,
	recompute: bool = True,
	on_error: str = "raise",
	sequential: bool = False,
	# Remote sync options
	remote_sync: "RemoteSync | None" = None,
	on_profile_complete: Callable[[str], None] | None = None,
	delete_after_upload: bool = False,
	# Logging
	logger: logging.Logger | None = None,
	# Forwarded MSI args
	nodes=None,
	edges=None,
	drug2protein_directed: bool = False,
	indication2protein_file_path: str = "data/2_indication_to_protein.tsv",
	indication2protein_directed: bool = False,
	protein2protein_file_path: str = "data/3_protein_to_protein.tsv",
	protein2protein_directed: bool = False,
	protein2biological_function_file_path: str = "data/4_protein_to_biological_function.tsv",
	protein2biological_function_directed: bool = False,
	biological_function2biological_function_file_path: str = "data/5_biological_function_to_biological_function.tsv",
	biological_function2biological_function_directed: bool = True,
	# Diffusion hyperparams
	alpha: float | None = None,
	max_iter: int | None = None,
	tol: float | None = None,
	weights: Mapping[str, float] | None = None,
	num_cores: int | None = None,
) -> dict[str, DiffusionRunMeta]:
	"""Compute diffusion profiles for *each* filtered drug→protein TSV.

	Creates one output directory per TSV under `save_root` to avoid file collisions.

	Returns a dict mapping `run_id` -> DiffusionRunMeta.

	This function is intentionally low-RAM: it computes and saves diffusion profiles
	to disk per run, but does not load/return diffusion vectors in memory.

	Args:
		save_root: Root directory for output
		drug2protein_glob: Glob pattern for finding TSV files
		drug2protein_file_paths: Explicit list of TSV paths (overrides glob)
		recompute: If True, recompute all; if False, skip complete runs
		on_error: 'raise' or 'skip'
		sequential: If True, process profiles one at a time (low memory mode)
		remote_sync: Optional RemoteSync instance for remote completion checking
		on_profile_complete: Optional callback for each completed profile (overrides auto-upload)
		delete_after_upload: If True and remote_sync is set, delete local .npy after upload
		logger: Optional logger instance
		... (MSI and diffusion hyperparameters)
	"""
	log = logger or logging.getLogger("diffusion_profiles")
	if on_error not in {"raise", "skip"}:
		raise ValueError("on_error must be 'raise' or 'skip'")

	os.makedirs(save_root, exist_ok=True)

	if drug2protein_file_paths is None:
		drug2protein_file_paths = sorted(glob.glob(drug2protein_glob))
	else:
		drug2protein_file_paths = list(drug2protein_file_paths)

	if not drug2protein_file_paths:
		raise ValueError(f"No drug2protein TSV files found (glob={drug2protein_glob!r})")

	# Keep defaults aligned with compute_all_diffusion_profiles_for_msi.
	from msi.msi import DRUG, INDICATION, PROTEIN, BIOLOGICAL_FUNCTION
	from msi.msi import (
		DRUG_PROTEIN,
		INDICATION_PROTEIN,
		PROTEIN_PROTEIN,
		PROTEIN_BIOLOGICAL_FUNCTION,
		BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION,
	)

	resolved_nodes = [DRUG, INDICATION, PROTEIN, BIOLOGICAL_FUNCTION] if nodes is None else nodes
	resolved_edges = (
		[
			DRUG_PROTEIN,
			INDICATION_PROTEIN,
			PROTEIN_PROTEIN,
			PROTEIN_BIOLOGICAL_FUNCTION,
			BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION,
		]
		if edges is None
		else edges
	)

	required_weight_keys = set(resolved_nodes)
	if (BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION in resolved_edges) and (BIOLOGICAL_FUNCTION in resolved_nodes):
		from msi.msi import UP_BIOLOGICAL_FUNCTION, DOWN_BIOLOGICAL_FUNCTION
		required_weight_keys |= {UP_BIOLOGICAL_FUNCTION, DOWN_BIOLOGICAL_FUNCTION}

	resolved_alpha = DEFAULT_ALPHA if alpha is None else alpha
	resolved_max_iter = DEFAULT_MAX_ITER if max_iter is None else max_iter
	resolved_tol = DEFAULT_TOL if tol is None else tol
	resolved_weights = dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)

	missing = required_weight_keys - set(resolved_weights.keys())
	if missing:
		raise ValueError(f"weights missing required keys: {sorted(missing)}")

	resolved_num_cores = default_num_cores() if num_cores is None else int(num_cores)
	resolved_num_cores = max(1, resolved_num_cores)

	runs: dict[str, DiffusionRunMeta] = {}

	total_files = len(drug2protein_file_paths)
	for file_idx, path in enumerate(drug2protein_file_paths, 1):
		run_id = _safe_run_id_from_path(path)
		out_dir = os.path.join(save_root, run_id)
		filename = os.path.basename(path)

		log.info(f"[{file_idx}/{total_files}] Processing: {filename}")

		# Skip complete runs when not recomputing (check local and remote)
		if not recompute and _run_is_complete(out_dir, remote_sync=remote_sync, run_id=run_id):
			expected = _get_expected_profile_count(out_dir)
			local_count = _count_existing_profiles(out_dir)
			if local_count == expected:
				log.info(f"  Already complete locally ({expected} profiles). Skipping.")
			else:
				log.info(f"  Already complete on remote ({expected} profiles). Skipping.")
			runs[run_id] = DiffusionRunMeta(
				run_id=run_id,
				drug2protein_file_path=path,
				save_load_file_path=out_dir,
				computed=False,
			)
			continue

		os.makedirs(out_dir, exist_ok=True)

		try:
			# Determine if we need to compute
			# - recompute=True: always compute
			# - recompute=False but incomplete: compute (fix partial state)
			# - recompute=False and complete: already handled above (skip)
			needs_compute = recompute or not _run_is_complete(
				out_dir, remote_sync=remote_sync, run_id=run_id
			)

			if needs_compute:
				if not recompute:
					# Resume mode but incomplete - explain why we're recomputing
					actual = _count_existing_profiles(out_dir)
					expected = _get_expected_profile_count(out_dir)
					if expected is not None:
						log.info(f"  Incomplete ({actual}/{expected} profiles). Recomputing...")
					else:
						log.info(f"  No completion data found. Computing...")

				msi = MSI(
					nodes=resolved_nodes,
					edges=resolved_edges,
					drug2protein_file_path=path,
					drug2protein_directed=drug2protein_directed,
					indication2protein_file_path=indication2protein_file_path,
					indication2protein_directed=indication2protein_directed,
					protein2protein_file_path=protein2protein_file_path,
					protein2protein_directed=protein2protein_directed,
					protein2biological_function_file_path=protein2biological_function_file_path,
					protein2biological_function_directed=protein2biological_function_directed,
					biological_function2biological_function_file_path=biological_function2biological_function_file_path,
					biological_function2biological_function_directed=biological_function2biological_function_directed,
				)
				msi.load()

				dp = DiffusionProfiles(
					alpha=resolved_alpha,
					max_iter=resolved_max_iter,
					tol=resolved_tol,
					weights=resolved_weights,
					num_cores=resolved_num_cores,
					save_load_file_path=out_dir,
				)
				# Build per-run upload callback if remote_sync is configured
				run_callback = on_profile_complete
				if run_callback is None and remote_sync is not None:
					from utils.remote_sync import make_upload_callback
					run_callback = make_upload_callback(
						remote_sync=remote_sync,
						run_dir=run_id,
						local_dir=out_dir,
						delete_after_upload=delete_after_upload,
						logger=log,
					)

				dp.calculate_diffusion_profiles(
					msi,
					sequential=sequential,
					on_profile_complete=run_callback,
					logger=log,
				)
				computed = True

				# Upload metadata to remote if configured
				if remote_sync is not None:
					for artifact in ("node2idx.pkl", "drugs_indications_lists.pkl"):
						local_artifact = os.path.join(out_dir, artifact)
						if os.path.exists(local_artifact):
							try:
								remote_sync.upload_file(
									local_artifact, os.path.join(run_id, artifact)
								)
							except Exception as e:
								log.warning(f"Failed to upload {artifact}: {e}")

				# CRITICAL: Free memory after each run to prevent accumulation
				del dp
				del msi
				gc.collect()
				log.info(f"  Completed and freed memory for: {run_id}")
			else:
				# This branch should not be reached due to the skip logic above
				# but keep for safety
				computed = True
				log.info(f"  Using existing artifacts for: {run_id}")

			runs[run_id] = DiffusionRunMeta(
				run_id=run_id,
				drug2protein_file_path=path,
				save_load_file_path=out_dir,
				computed=computed,
			)
		except Exception as e:
			log.error(f"  ERROR: {e}")
			log.debug(f"  Exception details:", exc_info=True)
			if on_error == "skip":
				gc.collect()  # Still try to free memory on error
				continue
			raise

	return runs
