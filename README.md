This repository contains experimental code.

See the repository at https://www.github.com/snap-stanford/multiscale-interactome for the original multiscale interactome model.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate multiscale_interactome
```

### Required data files

The MSI network requires 5 TSV edge files in a `data/` directory (or a custom path via `--data-dir`):

```
data/
  1_drug_to_protein.tsv
  2_indication_to_protein.tsv
  3_protein_to_protein.tsv
  4_protein_to_biological_function.tsv
  5_biological_function_to_biological_function.tsv
```

These are from the [original Stanford repository](https://github.com/snap-stanford/multiscale-interactome).

## Pipeline overview

```
1_prepare_filtered_edges.py    Generate filtered drug-protein edge files
         |
         v
2_compute_diffusion_profiles.py    Compute diffusion profiles for each edge set
         |
         v
3_compare_diffusion_profiles.py    Compare profiles (within-model or between-model)
```

## Step 1: Prepare filtered edges

Generate filtered drug-protein edge TSV files from docking scores or DTA-Atlas predictions. See [USAGE.md](USAGE.md) for details.

```bash
# Run both pipelines
python 1_prepare_filtered_edges.py

# Or only one
python 1_prepare_filtered_edges.py --dta-atlas-only
python 1_prepare_filtered_edges.py --docking-only
```

Output: `data/docking/*.tsv` and `data/dta_atlas/*.tsv`

## Step 2: Compute diffusion profiles

Compute personalized PageRank diffusion profiles for each filtered edge set.

```bash
# Custom input directory with custom data path
python 2_compute_diffusion_profiles.py \
    --input-dir /path/to/edges/dta_atlas/ \
    --data-dir /path/to/data/ \
    --output-dir /path/to/results/ \
    --num-cores 6

# Baseline MSI (no filtered edges)
python 2_compute_diffusion_profiles.py --baseline-only \
    --data-dir /path/to/data/ \
    --output-dir /path/to/results/

# With remote upload (SSH/SFTP)
python 2_compute_diffusion_profiles.py \
    --input-dir data/dta_atlas/ --data-dir data/ \
    --remote-host myserver.edu --remote-user myuser \
    --remote-path /data/results --delete-after-upload

# Resume interrupted computation
python 2_compute_diffusion_profiles.py \
    --input-dir data/dta_atlas/ --data-dir data/ \
    --output-dir results/ --resume

# Interactive selection of which edge files to process
python 2_compute_diffusion_profiles.py \
    --input-dir data/dta_atlas/ --interactive
```

Key options:
- `--input-dir`: Directory with `1_drug_to_protein_*.tsv` files
- `--data-dir`: Directory with the 5 base MSI edge files (default: `data/`)
- `--output-dir`: Where to save results (one subdirectory per edge file)
- `--num-cores`: CPU cores for parallel computation (default: 12)
- `--sequential`: Low-memory mode (process one profile at a time)
- `--resume`: Skip already-completed runs
- `--delete-after-upload`: Delete local `.npy` files after uploading to remote

Output structure:
```
output-dir/
  run_name_1/
    node2idx.pkl
    graph.pkl
    drugs_indications_lists.pkl
    DB00001_p_visit_array.npy
    C0002395_p_visit_array.npy
    ...
  run_name_2/
    ...
```

## Step 3: Compare diffusion profiles

Compare diffusion profiles across or within runs. Supports local and remote (SSH) profile directories.

### Within-model comparison

Compute drug x indication similarity within each run:

```bash
# Local runs
python 3_compare_diffusion_profiles.py --mode within \
    --runs-root results_filtered/dta_atlas/ \
    --output-dir comparisons/

# Remote runs
python 3_compare_diffusion_profiles.py --mode within \
    --remote-host myserver.edu --remote-port 50002 \
    --remote-user myuser --remote-path /data/results/ \
    --output-dir comparisons/ --max-workers 4

# Interactive selection
python 3_compare_diffusion_profiles.py --mode within \
    --remote-host myserver.edu --remote-user myuser \
    --remote-path /data/results/ \
    --output-dir comparisons/ --interactive
```

### Between-model comparison

Compare the same entity's profile across different runs (requires `node2idx.pkl` in each run):

```bash
# Pairwise comparison of all runs
python 3_compare_diffusion_profiles.py --mode between \
    --runs-root results_filtered/dta_atlas/ \
    --output-dir comparisons/

# One-vs-many with a reference run
python 3_compare_diffusion_profiles.py --mode between \
    --reference results/ \
    --runs-root results_filtered/dta_atlas/ \
    --output-dir comparisons/
```

Key options:
- `--method`: Similarity metric (`l2`, `l1`, `cosine`, `correlation`, `canberra`)
- `--normalization`: Normalize profiles before comparison (`none`, `l1`, `l2`)
- `--chunk-size`: Profiles loaded per chunk to control memory (default: 300)
- `--max-workers`: Thread pool size for parallel comparison
- `--clear-cache`: Wipe cached remote downloads to recover from corrupt files

Output: CSV files in `output-dir/within/` or `output-dir/between/`.

## Jupyter notebooks

The original interactive analysis notebook is also available:

```bash
jupyter notebook main.ipynb
```
