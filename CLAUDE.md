# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental fork of the [Stanford multiscale-interactome](https://www.github.com/snap-stanford/multiscale-interactome) project. It computes drug-disease relationship scores using diffusion-based algorithms on heterogeneous biological networks.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (no test runner configured - run individual test files)
python -c "from tests.msi import test_msi; test_msi()"
python -c "from tests.diff_prof import test_diffusion_profiles; test_diffusion_profiles()"
```

## Architecture

### Core Modules

**MSI (Multiscale Interactome)** - `msi/msi.py`
- Constructs a heterogeneous biological network from 5 TSV edge files in `data/`
- Node types: drugs, indications, proteins, biological functions (GO terms)
- Edge types: drug-protein, indication-protein, protein-protein, protein-biological_function, biological_function-biological_function
- Key data structures: `node2idx`, `idx2node`, `node2type`, `type2nodes`, `nodelist`
- `weight_graph(weights)` assigns edge weights by successor node type for diffusion

**Diffusion Profiles** - `diff_prof/`
- `compute_all_diffusion_profiles_for_msi()` - main entry point
- Implements personalized PageRank via power iteration (`diffusion_profiles.py`)
- Default hyperparameters in `defaults.py`: alpha=0.859, max_iter=1000, tol=1e-6
- Parallel computation using multiprocessing (batched by core count)
- Saves profiles as `{node_id}_p_visit_array.npy` files

**Filter Logic** - `filter_logic.py`
- `filter_drug_protein_edges()` - filters/generates drug-protein edges using docking scores
- Uses Polars for lazy query optimization on large datasets
- Two modes: `filter_existing` (keep scored edges) or `from_docking` (generate new edges)

### Data Flow

```
TSV files (data/*.tsv) → MSI.load() → NetworkX DiGraph
                                           ↓
                                    MSI.weight_graph(weights)
                                           ↓
                    DiffusionProfiles.calculate_diffusion_profiles()
                                           ↓
                    Per-drug/indication .npy profile files (results/)
```

### Key Patterns

- **Component architecture**: Each edge type (e.g., `DrugToProtein`) inherits from `NodeToNode` base class
- **Class-specific adjacency**: `cs_adj_dict` groups successors by type for weighted diffusion
- **Directed biological functions**: GO term hierarchy is directed; edges have up/down semantics

## Usage

```python
# Compute diffusion profiles
from diff_prof import compute_all_diffusion_profiles_for_msi
profiles, msi = compute_all_diffusion_profiles_for_msi(save_load_file_path="results/")

# Load MSI only
from msi.msi import MSI
msi = MSI()
msi.load()

# Compare diffusion profiles
from diff_prof import diffusion_profile_similarity
similarity = diffusion_profile_similarity(profile1, profile2, msi1, msi2)
```
