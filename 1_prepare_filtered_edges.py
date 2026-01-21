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
#import paramiko

from filter_logic import filter_drug_protein_edges
from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles

DRUG_COL = 'drugbank_id'
TGT_COL = 'uniprot_id'
LABEL_COL = 'sample_type'  # positive / negative_balanced
PAIR_COL = 'compound_target_pair'

df_lazy = pl.scan_parquet("data/combined_filtered_annotated_docking_results.parquet")

def _pair_best_score(df_in: pl.DataFrame, score_col: str, best: str) -> pl.DataFrame:
    if score_col not in df_in.columns:
        print(f"Warning: column not found, skipping: {score_col}")
        return pl.DataFrame()

    if best not in {"min", "max"}:
        raise ValueError("best must be 'min' or 'max'")

    agg_score = (pl.min(score_col) if best == "min" else pl.max(score_col)).alias(score_col)

    return (
        df_in
        .filter(pl.col(score_col).is_not_null() & pl.col(score_col).is_finite())
        .group_by(PAIR_COL)
        .agg([
            pl.first(DRUG_COL).alias(DRUG_COL),
            pl.first(TGT_COL).alias(TGT_COL),
            pl.first(LABEL_COL).alias(LABEL_COL),
            agg_score,
        ])
    )

df_gnina  = _pair_best_score(df_lazy, "GNINA_Score",  best="min").collect()
df_smina  = _pair_best_score(df_lazy, "SMINA_Score",  best="min").collect()
df_ledock = _pair_best_score(df_lazy, "Ledock_Score", best="min").collect()
df_gold   = _pair_best_score(df_lazy, "GOLD_Score",   best="max").collect()

df_DeepDTA = _pair_best_score(df_lazy, "dta_DeepDTA_ic50", best="max").collect()
df_GAT_GCN = _pair_best_score(df_lazy, "dta_GAT_GCN_ic50", best="max").collect()
df_GCNNet = _pair_best_score(df_lazy, "dta_GCNNet_ic50", best="max").collect()
df_GINConvNet = _pair_best_score(df_lazy, "dta_GINConvNet_ic50", best="max").collect()
df_MLTLE_GCN = _pair_best_score(df_lazy, "dta_MLTLE_GCN_ic50", best="max").collect()
df_MLTLE_GIN = _pair_best_score(df_lazy, "dta_MLTLE_GIN_ic50", best="max").collect()

tools_scores_dfs = {"GNINA": df_gnina, "SMINA": df_smina, "LEDOCK": df_ledock, "GOLD": df_gold,
                    "DeepDTA": df_DeepDTA, "GAT_GCN": df_GAT_GCN, "GCNNet": df_GCNNet,
                    "GINConvNet": df_GINConvNet, "MLTLE_GCN": df_MLTLE_GCN, "MLTLE_GIN": df_MLTLE_GIN}

# Load your dataframes efficiently
df_edges = pl.read_csv("data/1_drug_to_protein.tsv", separator="\t")
df_mapping = pl.read_csv("data/uniprot_gene_mapping.csv")

# Run the filter on all docking-based scores as well as ML-based scores
# by iterating over the tools_scores_dfs dictionary
cutoff_ranges = {}

# Use a range between -7.0 and -9.0 for docking scores (SMINA, GNINA)
# Use a range between -5.0 and -7.0 for LEDOCK
# Use a range between 60 and 90 for GOLD
# Use a range between 5 and 8 for ML-based scores
cutoff_ranges['GNINA'] = np.arange(-7.0, -9.1, -0.5)
cutoff_ranges['SMINA'] = np.arange(-7.0, -9.1, -0.5)
cutoff_ranges['LEDOCK'] = np.arange(-5.0, -7.1, -0.5)
cutoff_ranges['GOLD'] = np.arange(60, 91, 10)
cutoff_ranges['DeepDTA'] = np.arange(5, 9, 1)
cutoff_ranges['GAT_GCN'] = np.arange(5, 9, 1)
cutoff_ranges['GCNNet'] = np.arange(5, 9, 1)
cutoff_ranges['GINConvNet'] = np.arange(5, 9, 1)
cutoff_ranges['MLTLE_GCN'] = np.arange(5, 9, 1)
cutoff_ranges['MLTLE_GIN'] = np.arange(5, 9, 1) 

tool_score_columns = {
    "GNINA": "GNINA_Score",
    "SMINA": "SMINA_Score",
    "LEDOCK": "Ledock_Score",
    "GOLD": "GOLD_Score",
    "DeepDTA": "dta_DeepDTA_ic50",
    "GAT_GCN": "dta_GAT_GCN_ic50",
    "GCNNet": "dta_GCNNet_ic50",
    "GINConvNet": "dta_GINConvNet_ic50",
    "MLTLE_GCN": "dta_MLTLE_GCN_ic50",
    "MLTLE_GIN": "dta_MLTLE_GIN_ic50"
}

for tool_name, df_scores in tools_scores_dfs.items():
    if df_scores.is_empty():
        print(f"Skipping filtering for {tool_name} due to missing scores.")
        continue

    better = 'lower' if tool_name in {"GNINA", "SMINA", "LEDOCK"} else 'higher'
    for cutoff in cutoff_ranges[tool_name]:
        score_cutoff = cutoff
        filtered_edges = filter_drug_protein_edges(
            df_edges, 
            df_mapping, 
            df_scores, 
            score_column=tool_score_columns[tool_name], 
            better=better, 
            score_cutoff=score_cutoff,
            edge_source='from_scores',
            on_unmapped_protein='drop',
        )
        output_path = f"data/1_drug_to_protein_filtered_{tool_name.lower()}_cutoff_{score_cutoff}.tsv"
        filtered_edges.write_csv(output_path, separator="\t")
        print(f"Saved filtered edges for {tool_name} with cutoff {score_cutoff} to {output_path}")