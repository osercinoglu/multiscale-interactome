import logging
import os
import pickle
import multiprocessing
import networkx as nx
import math
import time
import copy
import scipy
import scipy.sparse
import numpy as np
from typing import Callable

WEIGHT = 'weight'

# =============================================================================
# Module-level worker state and functions for memory-efficient parallel processing
# =============================================================================

# Global state for worker processes (set by _worker_init)
_worker_data = None
_worker_alpha = None
_worker_max_iter = None
_worker_tol = None
_worker_save_path = None


def _worker_init(save_load_file_path, alpha, max_iter, tol):
	"""Initializer for Pool workers. Loads data from disk once per worker process."""
	global _worker_data, _worker_alpha, _worker_max_iter, _worker_tol, _worker_save_path

	from msi.msi import MSI

	# Load worker data from disk
	msi_data = MSI.load_worker_data(save_load_file_path)
	initial_M = scipy.sparse.load_npz(os.path.join(save_load_file_path, "initial_M.npz"))

	_worker_data = {
		'node2idx': msi_data['node2idx'],
		'nodelist': msi_data['nodelist'],
		'drug_or_indication2proteins': msi_data['drug_or_indication2proteins'],
		'drugs_in_graph': msi_data['drugs_in_graph'],
		'indications_in_graph': msi_data['indications_in_graph'],
		'initial_M': initial_M,
	}
	_worker_alpha = alpha
	_worker_max_iter = max_iter
	_worker_tol = tol
	_worker_save_path = save_load_file_path


def _clean_file_name(file_name):
	"""Clean filename for saving (standalone version for workers)."""
	return "".join([c for c in file_name if c.isalpha() or c.isdigit() or c==' ' or c =="_"]).rstrip()


def _compute_single_profile(node_id):
	"""Compute a single diffusion profile. Called by Pool workers.

	Uses module-level globals set by _worker_init().
	Returns the node ID on success for progress tracking.
	"""
	global _worker_data, _worker_alpha, _worker_max_iter, _worker_tol, _worker_save_path

	selected_list = [node_id]

	# Create modified M matrix (same logic as convert_M_to_make_all_drugs_indications_sinks_except_selected)
	reconstructed_M = copy.deepcopy(_worker_data['initial_M'])

	# Delete edges INTO selected drugs and indications
	for drug_or_indication in selected_list:
		proteins = _worker_data['drug_or_indication2proteins'][drug_or_indication]
		idxs_to_remove_in_edges = [_worker_data['node2idx'][p] for p in proteins]
		reconstructed_M[idxs_to_remove_in_edges, _worker_data['node2idx'][drug_or_indication]] = 0

	# Delete edges OUT of non-selected drugs and indications
	idxs_to_remove_out_edges = []
	for d_or_i in _worker_data['drugs_in_graph'] + _worker_data['indications_in_graph']:
		if d_or_i not in selected_list:
			proteins = _worker_data['drug_or_indication2proteins'][d_or_i]
			for protein in proteins:
				idxs_to_remove_out_edges.append(
					(_worker_data['node2idx'][d_or_i], _worker_data['node2idx'][protein])
				)

	if idxs_to_remove_out_edges:
		i, j = zip(*idxs_to_remove_out_edges)
		reconstructed_M[i, j] = 0.0

	# Refine M, S
	S = np.array(reconstructed_M.sum(axis=1)).flatten()
	S[S != 0] = 1.0 / S[S != 0]
	Q = scipy.sparse.spdiags(S.T, 0, *reconstructed_M.shape, format='csr')
	M = Q * reconstructed_M

	# Build personalization dict
	nodelist = _worker_data['nodelist']
	N_nodes = len(nodelist)
	per_dict = dict.fromkeys(nodelist, 0)
	per_dict[node_id] = 1.0

	# Power iteration
	p = np.array([per_dict[n] for n in nodelist], dtype=float)
	p = p / p.sum()
	dangling_weights = p
	is_dangling = np.where(S == 0)[0]

	x = np.repeat(1.0 / N_nodes, N_nodes)
	for _ in range(_worker_max_iter):
		xlast = x
		x = _worker_alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - _worker_alpha) * p
		err = np.absolute(x - xlast).sum()
		if err < N_nodes * _worker_tol:
			break

	# Save result
	clean_name = _clean_file_name(node_id)
	filepath = os.path.join(_worker_save_path, clean_name + "_p_visit_array.npy")
	np.save(filepath, x)

	return node_id


# =============================================================================
# DiffusionProfiles class
# =============================================================================

class DiffusionProfiles():
	def __init__(self, alpha, max_iter, tol, weights, num_cores, save_load_file_path):
		self.alpha = alpha
		self.max_iter = max_iter
		self.tol = tol
		self.weights = weights
		self.num_cores = num_cores
		self.save_load_file_path = save_load_file_path

	def get_initial_M(self, msi):
		# This function is adapted from the NetworkX implementation of Personalized PageRank #
		# Use to_scipy_sparse_array and convert to matrix for compatibility
		M = nx.to_scipy_sparse_array(msi.graph, nodelist = msi.nodelist, weight = WEIGHT, dtype = float)
		M = scipy.sparse.csr_matrix(M)
		N = len(msi.graph)
		if (N == 0):
			assert(False)
		self.initial_M = M

	def save_initial_M(self):
		"""Save the initial_M sparse matrix to disk for worker processes."""
		path = os.path.join(self.save_load_file_path, "initial_M.npz")
		scipy.sparse.save_npz(path, self.initial_M)

	def convert_M_to_make_all_drugs_indications_sinks_except_selected(self, msi, selected_drugs_and_indications):
		reconstructed_M = copy.deepcopy(self.initial_M)

		# Delete edges INTO selected drugs and indications
		for drug_or_indication in selected_drugs_and_indications:
			proteins_pointing_to_selected_drug_or_indication = msi.drug_or_indication2proteins[drug_or_indication]
			idxs_to_remove_in_edges = [msi.node2idx[protein] for protein in proteins_pointing_to_selected_drug_or_indication]
			reconstructed_M[idxs_to_remove_in_edges, msi.node2idx[drug_or_indication]] = 0

		# Delete edges OUT of nonselected drugs and indications
		idxs_to_remove_out_edges = []
		for drug_or_indication in msi.drugs_in_graph + msi.indications_in_graph:
			if not(drug_or_indication in selected_drugs_and_indications):
				proteins_drug_or_indication_points_to = msi.drug_or_indication2proteins[drug_or_indication]
				for protein in proteins_drug_or_indication_points_to:
					idxs_to_remove_out_edges.append((msi.node2idx[drug_or_indication], msi.node2idx[protein]))
		i, j = zip(*idxs_to_remove_out_edges)
		reconstructed_M[i, j] = 0.0
		return reconstructed_M

	def refine_M_S(self, M):
		# This function is adapted from the NetworkX implementation of Personalized PageRank #
		S = np.array(M.sum(axis=1)).flatten()
		S[S != 0] = 1.0 / S[S != 0]
		Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
		M = Q * M

		return M, S

	def get_personalization_dictionary(self, nodes_to_start_from, nodelist):
		personalization_dict = dict.fromkeys(nodelist, 0)
		N = len(nodes_to_start_from)
		for node in nodes_to_start_from:
			personalization_dict[node] = 1./N
		return personalization_dict

	def power_iteration(self, M, S, nodelist, per_dict):
		# This function is adapted from the NetworkX implementation of Personalized PageRank #
		# Size of nodelist
		N = len(nodelist)

		# Personalization vector
		missing = set(nodelist) - set(per_dict)
		if missing:
			raise nx.NetworkXError('Personalization dictionary must have a value for every node. Missing nodes %s' % missing)
		p = np.array([per_dict[n] for n in nodelist], dtype=float)
		p = p / p.sum()

		# Dangling nodes
		dangling_weights = p
		is_dangling = np.where(S == 0)[0]

		# power iteration: make up to max_iter iterations
		x = np.repeat(1.0 / N, N) # Initialize; alternatively x = p
		for _ in range(self.max_iter):
			xlast = x
			x = self.alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - self.alpha) * p
			# check convergence, l1 norm
			err = np.absolute(x - xlast).sum()
			if err < N * self.tol:
				return x
		raise nx.NetworkXError('pagerank_scipy: power iteration failed to converge in %d iterations.' % self.max_iter)

	def clean_file_name(self, file_name):
		return "".join([c for c in file_name if c.isalpha() or c.isdigit() or c==' ' or c =="_"]).rstrip()

	def save_diffusion_profile(self, diffusion_profile, selected_drug_or_indication):
		f = os.path.join(self.save_load_file_path, self.clean_file_name(selected_drug_or_indication) + "_p_visit_array.npy")
		np.save(f, diffusion_profile)

	def calculate_diffusion_profile_batch(self, msi, selected_drugs_and_indications_batch, batch_id=None, total_batches=None):
		batch_size = len(selected_drugs_and_indications_batch)
		for idx, selected_drugs_and_indications in enumerate(selected_drugs_and_indications_batch, 1):
			self.calculate_diffusion_profile(msi, selected_drugs_and_indications)
			# Progress every 50 items or at milestones within batch
			if idx % 50 == 0 or idx == batch_size or idx in [1, 10, 25]:
				if batch_id is not None:
					print(f"    [Batch {batch_id}/{total_batches}] {idx}/{batch_size} ({100*idx//batch_size}%)", flush=True)    		

	def calculate_diffusion_profile(self, msi, selected_drugs_and_indications):
		assert(len(selected_drugs_and_indications) == 1) # Not enabling functionality to run from multiple
		selected_drug_or_indication = selected_drugs_and_indications[0]
		M = self.convert_M_to_make_all_drugs_indications_sinks_except_selected(msi, selected_drugs_and_indications)
		M, S = self.refine_M_S(M)
		per_dict = self.get_personalization_dictionary(selected_drugs_and_indications, msi.nodelist)
		diffusion_profile = self.power_iteration(M, S, msi.nodelist, per_dict)
		self.save_diffusion_profile(diffusion_profile, selected_drug_or_indication)

	def batch_list(self, list_, batch_size = None, num_cores = None):
		if batch_size == float('inf'):
			batched_list = [list_]
		else:
			if batch_size is None:
				batch_size = math.ceil(len(list_) / (float(num_cores)))
			batched_list = []
			for i in range(0, len(list_), batch_size):
				batched_list.append(list_[i:i + batch_size])
		return batched_list

	def calculate_diffusion_profiles(
		self,
		msi,
		sequential: bool = False,
		on_profile_complete: Callable[[str], None] | None = None,
		logger: logging.Logger | None = None,
	):
		"""Calculate diffusion profiles for all drugs and indications.

		Uses multiprocessing.Pool with worker initializer for memory-efficient
		parallel computation. Workers load data from disk rather than inheriting
		parent memory via fork.

		Args:
			msi: MSI instance with loaded graph
			sequential: If True, process profiles one at a time (low memory mode)
			on_profile_complete: Optional callback called after each profile is saved.
				Receives the node_id as argument. Can be used for remote upload.
				If callback raises an exception, computation is aborted.
			logger: Optional logger instance. If not provided, uses print().
		"""
		log = logger or logging.getLogger("diffusion_profiles")

		# Save MSI data required by workers
		msi.save_graph(self.save_load_file_path)
		msi.save_node2idx(self.save_load_file_path)
		msi.save_worker_data(self.save_load_file_path)

		# Weight graph and create initial_M
		msi.weight_graph(self.weights)
		self.get_initial_M(msi)
		self.save_initial_M()

		# Build computation list (flat list of node IDs for Pool)
		computation_list = msi.drugs_in_graph + msi.indications_in_graph
		total_items = len(computation_list)

		# Sequential mode: process one at a time (lowest memory usage)
		if sequential or self.num_cores == 1:
			log.info(f"Computing {total_items} diffusion profiles sequentially (low memory mode)...")
			# Initialize worker state in main process
			_worker_init(self.save_load_file_path, self.alpha, self.max_iter, self.tol)

			for idx, node_id in enumerate(computation_list, 1):
				_compute_single_profile(node_id)
				# Call callback if provided (may raise to abort)
				if on_profile_complete:
					on_profile_complete(node_id)
				# Progress every 100 items or at milestones
				if idx % 100 == 0 or idx == total_items or idx in [1, 10, 50]:
					log.info(f"  Progress: {idx}/{total_items} ({100*idx//total_items}%)")
			log.info(f"All {total_items} profiles computed.")
			return

		# Parallel mode: use Pool with spawn context (memory-efficient)
		log.info(f"Computing {total_items} diffusion profiles with {self.num_cores} workers...")
		log.debug(f"Workers will load data from: {self.save_load_file_path}")

		# Use 'spawn' context to ensure clean worker processes (no inherited memory)
		ctx = multiprocessing.get_context('spawn')

		completed = 0
		with ctx.Pool(
			processes=self.num_cores,
			initializer=_worker_init,
			initargs=(self.save_load_file_path, self.alpha, self.max_iter, self.tol)
		) as pool:
			# Use imap_unordered for progress tracking
			for result in pool.imap_unordered(_compute_single_profile, computation_list, chunksize=10):
				completed += 1
				# Call callback if provided (may raise to abort)
				if on_profile_complete:
					on_profile_complete(result)
				if completed % 100 == 0 or completed == total_items or completed in [1, 10, 50]:
					log.info(f"  Progress: {completed}/{total_items} ({100*completed//total_items}%)")

		log.info(f"All {total_items} profiles computed.")

	def load_diffusion_profiles(self, drugs_and_indications):
		assert(not(self.save_load_file_path is None))

		# Load diffusion profiles
		drug_or_indication2diffusion_profile = dict()
		for drug_or_indication in drugs_and_indications:
			file_path = os.path.join(self.save_load_file_path, self.clean_file_name(drug_or_indication) + "_p_visit_array.npy")
			if (os.path.exists(file_path)):
				diffusion_profile = np.load(file_path)
				drug_or_indication2diffusion_profile[drug_or_indication] = diffusion_profile
			else:
				print("Loading failed at " + str(drug_or_indication) + " | " + str(file_path))
		self.drug_or_indication2diffusion_profile = drug_or_indication2diffusion_profile