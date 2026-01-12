import polars as pl

def filter_drug_protein_edges(
    drug_protein_df,
    gene_mapping_df,
    docking_scores_df,
    score_column,
    better='lower',
    score_cutoff=None,
    *,
    edge_source='filter_existing',
    output_score_column='score',
    on_unmapped_protein='drop',
    on_ambiguous_protein='first',
):
    """
    Filters (or generates) drug-protein edges based on docking scores.
    
    Args:
        drug_protein_df (pl.DataFrame): The original drug-protein edges (from 1_drug_to_protein.tsv).
        gene_mapping_df (pl.DataFrame): Mapping between Uniprot and Gene Names (from uniprot_gene_mapping.csv).
        docking_scores_df (pl.DataFrame): Docking scores.
        score_column (str): The column in docking_scores_df to use for filtering.
        better (str): 'lower' if lower scores are better (e.g., binding energy), 'higher' otherwise.
        score_cutoff (float): The threshold value.
        edge_source (str):
            - 'filter_existing' (default): keep only edges that already exist in drug_protein_df and pass the docking cutoff.
            - 'from_docking': generate edges solely from docking pairs passing the cutoff (may include edges absent in drug_protein_df).
        output_score_column (str): Name of the output docking score column (default: 'score').
        on_unmapped_protein (str): What to do if a docked gene cannot be mapped to a protein node id (Entrez) using drug_protein_df.
            - 'drop' (default): drop those docking-derived edges
            - 'raise': raise a ValueError
        on_ambiguous_protein (str): What to do if a gene name maps to multiple protein node ids in drug_protein_df.
            - 'first' (default): pick the minimum node id (deterministic)
            - 'drop': drop ambiguous genes
            - 'raise': raise a ValueError
        
    Returns:
        pl.DataFrame: Filtered (or generated) edges with an extra docking score column.
    """
    
    if edge_source not in {'filter_existing', 'from_docking'}:
        raise ValueError("edge_source must be 'filter_existing' or 'from_docking'")

    if better not in {'lower', 'higher'}:
        raise ValueError("better must be 'lower' or 'higher'")

    if on_unmapped_protein not in {'drop', 'raise'}:
        raise ValueError("on_unmapped_protein must be 'drop' or 'raise'")

    if on_ambiguous_protein not in {'first', 'drop', 'raise'}:
        raise ValueError("on_ambiguous_protein must be one of: 'first', 'drop', 'raise'")

    # 1. Prepare Docking Data
    # Extract Drug ID from compound_target_pair (assuming format DBxxxxx_Uniprot)
    # We use lazy execution to optimize memory
    q_docking = docking_scores_df.lazy() if isinstance(docking_scores_df, pl.DataFrame) else docking_scores_df
    
    # Filter by score first to reduce size immediately
    if score_cutoff is not None:
        if better == 'lower':
            q_docking = q_docking.filter(pl.col(score_column) < score_cutoff)
        elif better == 'higher':
            q_docking = q_docking.filter(pl.col(score_column) > score_cutoff)
            
    # Extract Drug ID and rename Uniprot column for join
    q_docking = q_docking.with_columns([
        pl.col("compound_target_pair").str.split("_").list.get(0).alias("drug_id"),
        pl.col("uniprot_id").alias("Entry")
    ])
    
    # 2. Prepare Mapping Data
    q_mapping = (gene_mapping_df.lazy() if isinstance(gene_mapping_df, pl.DataFrame) else gene_mapping_df).select(
        ["Entry", "Mapped_Gene_Name"]
    )
    
    # 3. Join Docking with Mapping to get Gene Names
    # We want to link the high-scoring Uniprot IDs to their Gene Names
    q_scored_genes = q_docking.join(q_mapping, on="Entry", how="inner")

    score_agg = pl.col(score_column).min() if better == 'lower' else pl.col(score_column).max()

    if edge_source == 'filter_existing':
        # 4. Filter Drug-Protein Edges
        # drug_to_protein.tsv has: node_1 (Drug), node_2_name (Gene Name)
        q_edges = drug_protein_df.lazy() if isinstance(drug_protein_df, pl.DataFrame) else drug_protein_df

        # Join edges with the valid (drug, gene) pairs from docking
        # We join on node_1 == drug_id AND node_2_name == Mapped_Gene_Name
        q_filtered_edges = q_edges.join(
            q_scored_genes,
            left_on=["node_1", "node_2_name"],
            right_on=["drug_id", "Mapped_Gene_Name"],
            how="inner",
        )

        # Select original columns and attach the best passing docking score per edge
        # (if multiple docking rows pass for the same drug-gene edge)
        original_cols = drug_protein_df.columns if isinstance(drug_protein_df, pl.DataFrame) else q_edges.collect_schema().names()
        q_final = q_filtered_edges.group_by(original_cols).agg(score_agg.alias(output_score_column))
        return q_final.collect()

    # edge_source == 'from_docking'
    # Build docking-derived (drug, gene) pairs with best passing score
    raw_score_agg = pl.col("_raw_score").min() if better == 'lower' else pl.col("_raw_score").max()
    q_pairs = (
        q_scored_genes
        .select(
            [
                pl.col("drug_id").alias("node_1"),
                pl.col("Mapped_Gene_Name").alias("node_2_name"),
                pl.col(score_column).alias("_raw_score"),
            ]
        )
        .group_by(["node_1", "node_2_name"])
        .agg(raw_score_agg.alias(output_score_column))
    )

    # Map gene symbol -> protein node id (Entrez) using the existing drug_protein_df
    q_edges = drug_protein_df.lazy() if isinstance(drug_protein_df, pl.DataFrame) else drug_protein_df
    q_gene_to_protein = (
        q_edges
        .select(["node_2_name", "node_2"])
        .drop_nulls(["node_2_name", "node_2"])
        .group_by("node_2_name")
        .agg(
            [
                pl.col("node_2").min().alias("node_2"),
                pl.col("node_2").n_unique().alias("_n_unique_node_2"),
            ]
        )
    )

    if on_ambiguous_protein == 'drop':
        q_gene_to_protein = q_gene_to_protein.filter(pl.col("_n_unique_node_2") == 1)
    elif on_ambiguous_protein == 'raise':
        ambiguous_ct = (
            q_gene_to_protein.filter(pl.col("_n_unique_node_2") > 1)
            .select(pl.len())
            .collect()
            .item()
        )
        if ambiguous_ct:
            raise ValueError(
                f"{ambiguous_ct} gene names map to multiple protein node ids in drug_protein_df; "
                "set on_ambiguous_protein='first' or 'drop' to continue"
            )

    q_gene_to_protein = q_gene_to_protein.drop("_n_unique_node_2")

    q_pairs = q_pairs.join(q_gene_to_protein, on="node_2_name", how="left")

    if on_unmapped_protein == 'drop':
        q_pairs = q_pairs.filter(pl.col("node_2").is_not_null())
    else:  # 'raise'
        unmapped_ct = q_pairs.filter(pl.col("node_2").is_null()).select(pl.len()).collect().item()
        if unmapped_ct:
            raise ValueError(
                f"{unmapped_ct} docking-derived edges could not be mapped to a protein node id (node_2); "
                "set on_unmapped_protein='drop' to ignore them"
            )

    # Bring over drug metadata when available, otherwise fill with simple defaults
    q_drug_attrs = (
        q_edges
        .select(["node_1", "node_1_type", "node_1_name"])
        .unique(subset=["node_1"])
    )
    q_pairs = q_pairs.join(q_drug_attrs, on="node_1", how="left")

    q_final = (
        q_pairs
        .with_columns(
            [
                pl.coalesce([pl.col("node_1_type"), pl.lit("drug")]).alias("node_1_type"),
                pl.lit("protein").alias("node_2_type"),
                pl.coalesce([pl.col("node_1_name"), pl.col("node_1")]).alias("node_1_name"),
            ]
        )
        .select(
            [
                "node_1",
                "node_2",
                "node_1_type",
                "node_2_type",
                "node_1_name",
                "node_2_name",
                output_score_column,
            ]
        )
    )

    return q_final.collect()
