from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


INVALID_ASSOCIATION_SCORE = -10_000_000.0


@dataclass
class PrototypeILPAssignmentResult:
    # (C, K) indices of selected prototypes grouped contiguously by class
    selected_indices_by_class: np.ndarray
    # (P, C) binary matrix; assignment_matrix[k, c] == 1 iff prototype k assigned to class c
    assignment_matrix: np.ndarray
    # (P, C) robust effect-size association matrix
    association_matrix: np.ndarray
    # objective value on original floating-point scale
    objective_value: float
    # (C,) whether each class had enough positive examples to be eligible
    valid_class_mask: np.ndarray


def winsorize(
    x: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo = np.quantile(x, lower)
    hi = np.quantile(x, upper)
    return np.clip(x, lo, hi)


def trimmed_mean(
    x: np.ndarray,
    trim: float = 0.10,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0

    k = int(trim * n)
    x_sorted = np.sort(x)

    # avoid empty slice for tiny sample sizes
    if 2 * k >= n:
        return float(np.mean(x_sorted))
    return float(np.mean(x_sorted[k : n - k]))


def robust_std(
    x: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return 0.0
    x_w = winsorize(x, lower=lower, upper=upper)
    return float(np.std(x_w, ddof=1))


def build_association_matrix(
    A: np.ndarray,  # (N, P) prototype activations
    Y: np.ndarray,  # (N, C) binary labels
    *,
    n_min: int = 10,
    trim: float = 0.10,
    eps: float = 1e-6,
    n_neg_repeats: int = 1,
    balanced_negative_sampling: bool = True,
    random_seed: int = 0,
    invalid_score: float = INVALID_ASSOCIATION_SCORE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute robust effect-size association matrix M where:

        M[k, c] = (mu_pos - mu_neg) / (sigma + eps)

    using:
      - trimmed means for positive / negative class-conditional activations
      - pooled robust standard deviation from the positive and negative groups

    Returns:
        M: (P, C)
        valid_class_mask: (C,) bool
    """
    A = np.asarray(A, dtype=np.float64)
    Y = np.asarray(Y)

    if A.ndim != 2:
        raise ValueError(f"A must have shape (N, P), got {A.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must have shape (N, C), got {Y.shape}")
    if A.shape[0] != Y.shape[0]:
        raise ValueError(
            f"A and Y must have same number of rows, got {A.shape[0]} and {Y.shape[0]}"
        )
    if n_neg_repeats < 1:
        raise ValueError("n_neg_repeats must be >= 1")

    N, P = A.shape
    _, C = Y.shape

    rng = np.random.default_rng(random_seed)

    M = np.full((P, C), invalid_score, dtype=np.float64)
    valid_class_mask = np.zeros(C, dtype=bool)

    for c in range(C):
        pos_idx = np.where(Y[:, c] == 1)[0]
        neg_idx_all = np.where(Y[:, c] == 0)[0]

        if pos_idx.size < n_min or neg_idx_all.size == 0:
            continue

        valid_class_mask[c] = True

        if balanced_negative_sampling:
            neg_sample_size = min(pos_idx.size, neg_idx_all.size)
        else:
            neg_sample_size = neg_idx_all.size

        for k in range(P):
            pos_vals = A[pos_idx, k]
            mu_pos = trimmed_mean(pos_vals, trim=trim)
            s_pos = robust_std(pos_vals)

            mu_neg_acc = 0.0
            s_neg_acc = 0.0

            for _ in range(n_neg_repeats):
                if balanced_negative_sampling and neg_idx_all.size > neg_sample_size:
                    neg_idx = rng.choice(
                        neg_idx_all,
                        size=neg_sample_size,
                        replace=False,
                    )
                else:
                    neg_idx = neg_idx_all

                neg_vals = A[neg_idx, k]
                mu_neg_acc += trimmed_mean(neg_vals, trim=trim)
                s_neg_acc += robust_std(neg_vals)

            mu_neg = mu_neg_acc / float(n_neg_repeats)
            s_neg = s_neg_acc / float(n_neg_repeats)

            sigma = np.sqrt((s_pos**2 + s_neg**2) / 2.0)
            M[k, c] = (mu_pos - mu_neg) / (sigma + eps)

    return M.astype(np.float32), valid_class_mask

def solve_assignment_ilp(
    M: np.ndarray,  # (P, C)
    *,
    n_prototypes_per_label: int,
    valid_class_mask: np.ndarray | None = None,
    invalid_score: float = INVALID_ASSOCIATION_SCORE,
) -> PrototypeILPAssignmentResult:
    """
    Solve the exact-quota prototype assignment problem by reducing it to a
    rectangular linear assignment problem.

    We create K identical slots for each class, where K = n_prototypes_per_label.
    Then we assign prototypes to class slots with maximum total association score.

    This supports:
      - exactly K prototypes per class
      - each prototype used at most once
      - extra prototypes left unused when P > C*K
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must have shape (P, C), got {M.shape}")

    P, C = M.shape

    if n_prototypes_per_label <= 0:
        raise ValueError("n_prototypes_per_label must be positive")

    if valid_class_mask is None:
        valid_class_mask = np.ones(C, dtype=bool)
    else:
        valid_class_mask = np.asarray(valid_class_mask, dtype=bool)
        if valid_class_mask.shape != (C,):
            raise ValueError(
                f"valid_class_mask must have shape ({C},), got {valid_class_mask.shape}"
            )

    invalid_classes = np.where(~valid_class_mask)[0]
    if invalid_classes.size > 0:
        raise ValueError(
            "Assignment requires every class to be eligible, "
            f"but classes {invalid_classes.tolist()} are under-supported."
        )

    K = n_prototypes_per_label
    n_slots = C * K

    if P < n_slots:
        raise ValueError(
            f"Infeasible assignment: need at least {n_slots} prototypes "
            f"for {C} classes x {K} slots, but only have {P}."
        )

    # Expand class scores into slot scores: (P, C*K)
    # slot_to_class[j] tells which class slot j belongs to
    slot_to_class = np.repeat(np.arange(C), K)
    score_matrix = M[:, slot_to_class]  # (P, C*K)

    # Forbidden assignments: set to a very bad score
    forbidden_mask = score_matrix <= invalid_score / 2.0
    if np.any(forbidden_mask):
        score_matrix = score_matrix.copy()
        score_matrix[forbidden_mask] = invalid_score

    # linear_sum_assignment minimizes cost, so negate scores to maximize
    cost_matrix = -score_matrix

    # rectangular assignment:
    # - one prototype per chosen slot
    # - one slot per chosen prototype
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Since P >= C*K, linear_sum_assignment returns assignments for all columns
    # only if columns <= rows. That is our case.
    if len(col_ind) != n_slots:
        raise RuntimeError(
            f"Assignment failed: expected {n_slots} assigned slots, got {len(col_ind)}"
        )

    assignment_matrix = np.zeros((P, C), dtype=np.int64)
    selected_indices_by_class = [[] for _ in range(C)]
    objective_value = 0.0

    for proto_idx, slot_idx in zip(row_ind, col_ind):
        c = int(slot_to_class[slot_idx])
        assignment_matrix[proto_idx, c] = 1
        selected_indices_by_class[c].append(int(proto_idx))
        objective_value += float(M[proto_idx, c])

    selected_indices_by_class = [
        sorted(v) for v in selected_indices_by_class
    ]

    for c, v in enumerate(selected_indices_by_class):
        if len(v) != K:
            raise RuntimeError(
                f"Class {c} received {len(v)} prototypes, expected {K}"
            )

    selected_indices_by_class = np.asarray(selected_indices_by_class, dtype=np.int64)

    return PrototypeILPAssignmentResult(
        selected_indices_by_class=selected_indices_by_class,
        assignment_matrix=assignment_matrix,
        association_matrix=M.astype(np.float32),
        objective_value=float(objective_value),
        valid_class_mask=valid_class_mask,
    )

# def compute_controlled_sharing(
#     M: np.ndarray,  # (P, C)
#     selected_indices_by_class: np.ndarray,  # (C, K)
#     *,
#     delta_M: float = 0.1,
#     tau_abs: float = 0.5,
#     s_max: int = 1,
# ) -> list[set[int]]:
#     """
#     Optional post-processing for interpretability only.

#     Returns:
#         shared[k] = set of additional class ids for prototype k

#     Notes:
#         - primary membership is determined by `selected_indices_by_class`
#         - shared memberships do NOT satisfy coverage/accounting in this repo
#         - downstream projection/classifier code currently does not consume sharing;
#           this helper is mainly for analysis / metadata
#     """
#     M = np.asarray(M, dtype=np.float64)
#     selected_indices_by_class = np.asarray(selected_indices_by_class, dtype=np.int64)

#     if M.ndim != 2:
#         raise ValueError(f"M must have shape (P, C), got {M.shape}")
#     if selected_indices_by_class.ndim != 2:
#         raise ValueError(
#             "selected_indices_by_class must have shape (C, K), "
#             f"got {selected_indices_by_class.shape}"
#         )
#     if s_max < 0:
#         raise ValueError("s_max must be >= 0")

#     P, C = M.shape
#     primary = np.full(P, -1, dtype=np.int64)

#     for c in range(selected_indices_by_class.shape[0]):
#         for k in selected_indices_by_class[c]:
#             if k < 0 or k >= P:
#                 raise ValueError(f"Prototype index {k} out of bounds for P={P}")
#             if primary[k] >= 0:
#                 raise ValueError(
#                     f"Prototype {k} was assigned to more than one primary class"
#                 )
#             primary[k] = c

#     shared = [set() for _ in range(P)]
#     for k in range(P):
#         c1 = primary[k]
#         if c1 < 0:
#             continue

#         scores = M[k]
#         order = np.argsort(-scores)

#         added = 0
#         for c in order:
#             c = int(c)
#             if c == c1:
#                 continue

#             if scores[c] >= scores[c1] - delta_M and scores[c] >= tau_abs:
#                 shared[k].add(c)
#                 added += 1
#                 if added >= s_max:
#                     break

#     return shared