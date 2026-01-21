from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from msi.msi import MSI

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


def compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs(
	*,
	save_root: str,
	drug2protein_glob: str = "data/1_drug_to_protein_filtered_*.tsv",
	drug2protein_file_paths: Sequence[str] | None = None,
	recompute: bool = True,
	on_error: str = "raise",
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
	"""
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

	for path in drug2protein_file_paths:
		run_id = _safe_run_id_from_path(path)
		out_dir = os.path.join(save_root, run_id)
		os.makedirs(out_dir, exist_ok=True)

		try:
			if recompute:
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
				dp.calculate_diffusion_profiles(msi)
				computed = True
			else:
				computed = _run_has_saved_artifacts(out_dir)
				if not computed:
					raise FileNotFoundError(
						f"Missing saved diffusion artifacts in {out_dir!r}; rerun with recompute=True"
					)

			runs[run_id] = DiffusionRunMeta(
				run_id=run_id,
				drug2protein_file_path=path,
				save_load_file_path=out_dir,
				computed=computed,
			)
		except Exception:
			if on_error == "skip":
				continue
			raise

	return runs
