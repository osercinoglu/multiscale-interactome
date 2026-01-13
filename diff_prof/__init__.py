from __future__ import annotations

import os
from typing import Mapping

import numpy as np

from msi.msi import (
	DOWN_BIOLOGICAL_FUNCTION,
	BIOLOGICAL_FUNCTION,
	BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION,
	DRUG,
	DRUG_PROTEIN,
	INDICATION,
	INDICATION_PROTEIN,
	PROTEIN,
	PROTEIN_BIOLOGICAL_FUNCTION,
	PROTEIN_PROTEIN,
	UP_BIOLOGICAL_FUNCTION,
	MSI,
)

from .defaults import (
	DEFAULT_ALPHA,
	DEFAULT_MAX_ITER,
	DEFAULT_TOL,
	DEFAULT_WEIGHTS,
	default_num_cores,
)
from .diffusion_profiles import DiffusionProfiles
from .compare import diffusion_profile_similarity
from .batch import compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs, DiffusionRun


def compute_all_diffusion_profiles_for_msi(
	nodes=[DRUG, INDICATION, PROTEIN, BIOLOGICAL_FUNCTION],
	edges=[
		DRUG_PROTEIN,
		INDICATION_PROTEIN,
		PROTEIN_PROTEIN,
		PROTEIN_BIOLOGICAL_FUNCTION,
		BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION,
	],
	drug2protein_file_path: str = "data/1_drug_to_protein.tsv",
	drug2protein_directed: bool = False,
	indication2protein_file_path: str = "data/2_indication_to_protein.tsv",
	indication2protein_directed: bool = False,
	protein2protein_file_path: str = "data/3_protein_to_protein.tsv",
	protein2protein_directed: bool = False,
	protein2biological_function_file_path: str = "data/4_protein_to_biological_function.tsv",
	protein2biological_function_directed: bool = False,
	biological_function2biological_function_file_path: str = "data/5_biological_function_to_biological_function.tsv",
	biological_function2biological_function_directed: bool = True,
	*,
	save_load_file_path: str,
	alpha: float | None = None,
	max_iter: int | None = None,
	tol: float | None = None,
	weights: Mapping[str, float] | None = None,
	num_cores: int | None = None,
) -> tuple[dict[str, np.ndarray], MSI]:
	"""Build MSI from the given MSI args and recompute all diffusion profiles.

	Accepts the same parameters as `msi.msi.MSI.__init__` plus diffusion hyperparameters.
	Always recomputes diffusion profiles (overwriting/refreshing on-disk `.npy` outputs).

	Returns:
		(profiles, msi):
			- profiles: dict mapping drug/indication node id -> diffusion profile vector
			- msi: the loaded MSI instance (for node ordering via `nodelist`/`node2idx`)
	"""

	os.makedirs(save_load_file_path, exist_ok=True)

	msi = MSI(
		nodes=nodes,
		edges=edges,
		drug2protein_file_path=drug2protein_file_path,
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

	resolved_alpha = DEFAULT_ALPHA if alpha is None else alpha
	resolved_max_iter = DEFAULT_MAX_ITER if max_iter is None else max_iter
	resolved_tol = DEFAULT_TOL if tol is None else tol
	resolved_weights = dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)

	required_weight_keys = set(nodes)
	if (BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION in edges) and (BIOLOGICAL_FUNCTION in nodes):
		required_weight_keys |= {UP_BIOLOGICAL_FUNCTION, DOWN_BIOLOGICAL_FUNCTION}
	missing = required_weight_keys - set(resolved_weights.keys())
	if missing:
		raise ValueError(f"weights missing required keys: {sorted(missing)}")

	resolved_num_cores = default_num_cores() if num_cores is None else int(num_cores)
	resolved_num_cores = max(1, resolved_num_cores)

	dp = DiffusionProfiles(
		alpha=resolved_alpha,
		max_iter=resolved_max_iter,
		tol=resolved_tol,
		weights=resolved_weights,
		num_cores=resolved_num_cores,
		save_load_file_path=save_load_file_path,
	)

	dp.calculate_diffusion_profiles(msi)
	dp.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

	return dp.drug_or_indication2diffusion_profile, msi

