import polars as pl

from filter_logic import filter_drug_protein_edges


def test_filter_drug_protein_edges_from_docking_generates_new_edges():
    # Existing edges: only one drug->protein edge present
    drug_protein_df = pl.DataFrame(
        {
            "node_1": ["DB00001"],
            "node_2": [101],
            "node_1_type": ["drug"],
            "node_2_type": ["protein"],
            "node_1_name": ["drug_a"],
            "node_2_name": ["GENE1"],
        }
    )

    # UniProt -> gene mapping
    gene_mapping_df = pl.DataFrame(
        {
            "Entry": ["P00001", "P00002"],
            "Mapped_Gene_Name": ["GENE1", "GENE2"],
        }
    )

    # Docking includes a second gene that is NOT present as an edge in drug_protein_df
    docking_scores_df = pl.DataFrame(
        {
            "compound_target_pair": ["DB00001_P00001", "DB00001_P00002"],
            "uniprot_id": ["P00001", "P00002"],
            "GNINA_Score": [-9.0, -9.5],
        }
    )

    # Add a protein id mapping for GENE2 via an additional row in drug_protein_df
    # (this reflects how the real interactome provides gene symbol -> Entrez id mapping)
    drug_protein_df_with_mapping = pl.concat(
        [
            drug_protein_df,
            pl.DataFrame(
                {
                    "node_1": ["DB99999"],
                    "node_2": [202],
                    "node_1_type": ["drug"],
                    "node_2_type": ["protein"],
                    "node_1_name": ["drug_z"],
                    "node_2_name": ["GENE2"],
                }
            ),
        ],
        how="vertical",
    )

    out = filter_drug_protein_edges(
        drug_protein_df_with_mapping,
        gene_mapping_df,
        docking_scores_df,
        score_column="GNINA_Score",
        better="lower",
        score_cutoff=-8.0,
        edge_source="from_docking",
        output_score_column="score",
        on_unmapped_protein="raise",
    )

    assert set(out.columns) == {
        "node_1",
        "node_2",
        "node_1_type",
        "node_2_type",
        "node_1_name",
        "node_2_name",
        "score",
    }

    # Should include both docking-derived edges for DB00001
    out_pairs = set(out.select(["node_1", "node_2_name"]).iter_rows())
    assert ("DB00001", "GENE1") in out_pairs
    assert ("DB00001", "GENE2") in out_pairs

    # And have mapped Entrez ids
    gene2_to_id = dict(out.select(["node_2_name", "node_2"]).iter_rows())
    assert gene2_to_id["GENE1"] == 101
    assert gene2_to_id["GENE2"] == 202


def test_filter_drug_protein_edges_filter_existing_default_unchanged():
    drug_protein_df = pl.DataFrame(
        {
            "node_1": ["DB00001"],
            "node_2": [101],
            "node_1_type": ["drug"],
            "node_2_type": ["protein"],
            "node_1_name": ["drug_a"],
            "node_2_name": ["GENE1"],
        }
    )

    gene_mapping_df = pl.DataFrame({"Entry": ["P00001"], "Mapped_Gene_Name": ["GENE1"]})
    docking_scores_df = pl.DataFrame(
        {
            "compound_target_pair": ["DB00001_P00001"],
            "uniprot_id": ["P00001"],
            "GNINA_Score": [-9.0],
        }
    )

    out = filter_drug_protein_edges(
        drug_protein_df,
        gene_mapping_df,
        docking_scores_df,
        score_column="GNINA_Score",
        better="lower",
        score_cutoff=-8.0,
        # default edge_source='filter_existing'
    )

    assert out.height == 1
    assert out["node_1"][0] == "DB00001"
    assert out["node_2_name"][0] == "GENE1"
    assert "score" in out.columns
