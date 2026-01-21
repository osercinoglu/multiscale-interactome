import multiprocessing as mp


def main() -> None:
    # Batch wrapper usage: build MSI + recompute ALL diffusion profiles
    # Note: this can take a while and writes many .npy files.
    from diff_prof import (
        compute_all_diffusion_profiles_for_msi,
        compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs,
    )

    # Example A: default MSI files (same as MSI())
    # (This returns a big in-memory dict; if you only need on-disk artifacts,
    # consider using DiffusionProfiles.calculate_diffusion_profiles directly.)
    _, _ = compute_all_diffusion_profiles_for_msi(
        save_load_file_path="results/",
        num_cores=12,
    )

    # Example B: compute diffusion profiles for EVERY filtered drug→protein TSV we generated
    # One output directory per TSV under results_filtered/
    runs_filtered = compute_all_diffusion_profiles_for_msi_across_filtered_drug2protein_tsvs(
        save_root="results_filtered/",
        drug2protein_glob="data/1_drug_to_protein_filtered_*cutoff_*.tsv",
        recompute=True,  # set to False to only validate presence of saved artifacts
        on_error="skip",
        num_cores=12,
    )

    print(
        f"Prepared/validated {len(runs_filtered)} diffusion-profile runs under results_filtered/ "
        "(profiles not loaded into RAM)."
    )


if __name__ == "__main__":
    # When the multiprocessing start method is 'spawn' (common on macOS/Windows,
    # sometimes in certain environments), Python re-imports this module in child
    # processes. Guarding main prevents recursive process spawning.
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        # Start method already set (or fork not available); safe to ignore.
        pass
    main()