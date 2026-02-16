from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np


Normalization = Literal[None, "l1", "l2"]
AlignStrategy = Literal["auto", "position", "intersection"]
Method = Literal["l1", "l2", "canberra", "cosine", "correlation"]


def _to_1d_array(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    return arr.ravel()


def _normalize(vec: np.ndarray, normalization: Normalization, eps: float) -> np.ndarray:
    if normalization is None:
        return vec

    if normalization == "l1":
        s = float(np.sum(vec))
        return vec / (s + eps)

    if normalization == "l2":
        n = float(np.linalg.norm(vec, ord=2))
        return vec / (n + eps)

    raise ValueError(f"Unknown normalization: {normalization}")


def diffusion_profile_similarity(
    a: Any,
    b: Any,
    *,
    method: Method = "l2",
    align: AlignStrategy = "auto",
    msi_a: Any | None = None,
    msi_b: Any | None = None,
    normalization: Normalization = None,
    eps: float = 1e-12,
    return_diagnostics: bool = True,
) -> float | tuple[float, dict[str, Any]]:
    """Compute similarity between two diffusion profiles.

    Supports profiles computed on different MSI graphs by aligning on the intersection
    of nodes (using `msi_a.node2idx`/`msi_b.node2idx`).

    Similarity conventions:
    - Higher is more similar
    - 1.0 means identical after alignment/normalization

    Args:
        a, b: Diffusion profiles (array-like). Typically 1D numpy arrays.
        method: Similarity method. Currently supports: "l2".
        align: "position" (require same length), "intersection" (require MSI mappings),
               or "auto" (position if same length else intersection).
        msi_a, msi_b: MSI objects corresponding to `a` and `b` when using intersection.
        normalization: Optional normalization applied after alignment: None, "l1", "l2".
        eps: Small constant for numerical stability.
        return_diagnostics: If True, returns (similarity, diagnostics).

    Returns:
        similarity or (similarity, diagnostics)
    """

    vec_a_full = _to_1d_array(a)
    vec_b_full = _to_1d_array(b)

    diagnostics: dict[str, Any] = {
        "method": method,
        "align": align,
        "normalization": normalization,
        "n_a": int(vec_a_full.shape[0]),
        "n_b": int(vec_b_full.shape[0]),
    }

    resolved_align: str
    if align == "auto":
        resolved_align = "position" if vec_a_full.shape == vec_b_full.shape else "intersection"
    else:
        resolved_align = align

    if resolved_align == "position":
        if vec_a_full.shape != vec_b_full.shape:
            raise ValueError(f"Shape mismatch for position alignment: {vec_a_full.shape} vs {vec_b_full.shape}")
        vec_a = vec_a_full
        vec_b = vec_b_full
        diagnostics["aligned_by"] = "position"
        diagnostics["n_common"] = int(vec_a.shape[0])

    elif resolved_align == "intersection":
        if msi_a is None or msi_b is None:
            raise ValueError("msi_a and msi_b are required for intersection alignment")
        if not (hasattr(msi_a, "node2idx") and hasattr(msi_b, "node2idx")):
            raise ValueError("msi_a/msi_b must have node2idx for intersection alignment")

        node2idx_a: Mapping[Any, int] = msi_a.node2idx  # type: ignore[assignment]
        node2idx_b: Mapping[Any, int] = msi_b.node2idx  # type: ignore[assignment]

        # Prefer preserving the reference MSI order to avoid sorting issues and to
        # keep semantics aligned with how the diffusion vector was constructed.
        if hasattr(msi_a, "nodelist"):
            common_nodes = [n for n in msi_a.nodelist if n in node2idx_b]
        else:
            common_nodes = [n for n in node2idx_a.keys() if n in node2idx_b]

        if len(common_nodes) == 0:
            raise ValueError("No common nodes between the two MSI graphs; cannot compare.")

        idx_a = np.fromiter((node2idx_a[n] for n in common_nodes), dtype=int)
        idx_b = np.fromiter((node2idx_b[n] for n in common_nodes), dtype=int)

        vec_a = vec_a_full[idx_a]
        vec_b = vec_b_full[idx_b]

        diagnostics["aligned_by"] = "intersection"
        diagnostics["n_common"] = int(len(common_nodes))
        diagnostics["sum_a_aligned"] = float(np.sum(vec_a))
        diagnostics["sum_b_aligned"] = float(np.sum(vec_b))

    else:
        raise ValueError(f"Unknown align strategy: {align}")

    vec_a = _normalize(vec_a.astype(float, copy=False), normalization, eps)
    vec_b = _normalize(vec_b.astype(float, copy=False), normalization, eps)

    diagnostics["sum_a"] = float(np.sum(vec_a))
    diagnostics["sum_b"] = float(np.sum(vec_b))

    if method == "l2":
        dist = float(np.linalg.norm(vec_a - vec_b, ord=2))
        norm_a = float(np.linalg.norm(vec_a, ord=2))
        norm_b = float(np.linalg.norm(vec_b, ord=2))
        rel_dist = dist / (norm_a + eps)

        # Convert distance -> similarity (bounded in (0, 1])
        sim = 1.0 / (1.0 + dist)

        diagnostics.update(
            {
                "l2_distance": dist,
                "l2_norm_a": norm_a,
                "l2_norm_b": norm_b,
                "relative_l2_distance": rel_dist,
                "similarity_transform": "1/(1+dist)",
                "direction": "higher_is_more_similar",
            }
        )

    elif method == "l1":
        dist = float(np.linalg.norm(vec_a - vec_b, ord=1))
        norm_a = float(np.linalg.norm(vec_a, ord=1))
        norm_b = float(np.linalg.norm(vec_b, ord=1))
        rel_dist = dist / (norm_a + eps)

        # Convert distance -> similarity (bounded in (0, 1])
        sim = 1.0 / (1.0 + dist)

        diagnostics.update(
            {
                "l1_distance": dist,
                "l1_norm_a": norm_a,
                "l1_norm_b": norm_b,
                "relative_l1_distance": rel_dist,
                "similarity_transform": "1/(1+dist)",
                "direction": "higher_is_more_similar",
            }
        )

    elif method == "canberra":
        # Canberra distance: Σ(|a_i - b_i| / (|a_i| + |b_i|))
        # Handle division by zero with epsilon
        numerator = np.abs(vec_a - vec_b)
        denominator = np.abs(vec_a) + np.abs(vec_b) + eps
        dist = float(np.sum(numerator / denominator))

        # Maximum possible Canberra distance is n (number of elements)
        max_dist = len(vec_a)

        # Convert distance -> similarity (bounded in (0, 1])
        sim = 1.0 / (1.0 + dist)

        diagnostics.update(
            {
                "canberra_distance": dist,
                "max_possible_canberra": max_dist,
                "relative_canberra_distance": dist / max_dist,
                "similarity_transform": "1/(1+dist)",
                "direction": "higher_is_more_similar",
            }
        )

    elif method == "cosine":
        # Cosine similarity: (a · b) / (||a|| × ||b||)
        dot_product = float(np.dot(vec_a, vec_b))
        norm_a = float(np.linalg.norm(vec_a, ord=2))
        norm_b = float(np.linalg.norm(vec_b, ord=2))

        # Handle zero vectors
        if norm_a < eps or norm_b < eps:
            cosine_sim = 0.0
        else:
            cosine_sim = dot_product / (norm_a * norm_b)

        # Cosine similarity is already in [-1, 1], typically [0, 1] for non-negative vectors
        # Map to [0, 1] range: (cosine + 1) / 2
        sim = (cosine_sim + 1.0) / 2.0

        diagnostics.update(
            {
                "cosine_similarity": cosine_sim,
                "dot_product": dot_product,
                "l2_norm_a": norm_a,
                "l2_norm_b": norm_b,
                "similarity_transform": "(cosine+1)/2",
                "direction": "higher_is_more_similar",
            }
        )

    elif method == "correlation":
        # Pearson correlation: equivalent to cosine similarity on centered vectors
        mean_a = float(np.mean(vec_a))
        mean_b = float(np.mean(vec_b))
        std_a = float(np.std(vec_a))
        std_b = float(np.std(vec_b))

        # Center the vectors
        centered_a = vec_a - mean_a
        centered_b = vec_b - mean_b

        # Handle constant vectors (std = 0)
        if std_a < eps or std_b < eps:
            # If both are constant and equal, perfect correlation
            if abs(mean_a - mean_b) < eps:
                correlation = 1.0
            else:
                correlation = 0.0
        else:
            # Compute correlation via centered cosine similarity
            dot_product = float(np.dot(centered_a, centered_b))
            norm_a = float(np.linalg.norm(centered_a, ord=2))
            norm_b = float(np.linalg.norm(centered_b, ord=2))
            correlation = dot_product / (norm_a * norm_b)

        # Correlation distance is 1 - correlation
        corr_dist = 1.0 - correlation

        # Convert to similarity: correlation is already in [-1, 1]
        # Map to [0, 1]: (correlation + 1) / 2
        sim = (correlation + 1.0) / 2.0

        diagnostics.update(
            {
                "correlation": correlation,
                "correlation_distance": corr_dist,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "std_a": std_a,
                "std_b": std_b,
                "similarity_transform": "(correlation+1)/2",
                "direction": "higher_is_more_similar",
            }
        )

    else:
        raise ValueError(f"Unsupported method: {method}")

    if return_diagnostics:
        return sim, diagnostics
    return sim
