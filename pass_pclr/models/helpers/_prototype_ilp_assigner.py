from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

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
    max_classes_per_prototype: int = 1,
    valid_class_mask: np.ndarray | None = None,
    invalid_score: float = INVALID_ASSOCIATION_SCORE,
) -> PrototypeILPAssignmentResult:
    """
    Solve the exact-quota prototype assignment problem by reducing it to a
    variant of the rectangular linear assignment problem.

    We create K identical slots for each class, where K = n_prototypes_per_label.
    Then we assign prototypes to class slots with maximum total association score.

    This supports:
      - exactly K prototypes per class
      - each prototype used at most R times
      - extra prototypes left unused when P*R > C*K
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must have shape (P, C), got {M.shape}")

    P, C = M.shape
    K = n_prototypes_per_label
    R = max_classes_per_prototype

    if K <= 0:
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

    n_slots = C * K
    if P * R < n_slots:
        raise ValueError(
            f"Infeasible assignment: need at least {n_slots} allowable assignments "
            f"for {C} classes x {K} slots, but only have {P} prototypes x {R} classes per prototype."
        )

    # Expand class scores into slot scores: (P, C*K)
    # slot_to_class[j] tells which class slot j belongs to
    slot_to_class = np.repeat(np.arange(C), K)
    score_matrix = M[:, slot_to_class]  # (P, C*K)

    # milp minimizes cost, so negate scores to maximize
    # milp also requires flat 1D array
    c_obj = -score_matrix.ravel()

    # mimic LAP rectangular assignment:
    # - one prototype per chosen slot
    # - one slot per chosen prototype
    # Let x be the optimal assignment matrix with shape (P, n_slots)
    # milp expects flat 1D array so x[p, s] = x_flat[p * n_slots + s]
    N = P * n_slots  # number of decision variables
    cell_lbs = np.zeros(N)
    cell_ubs = np.ones(N)

    # Forbidden assignments: set upper bound so that assignments are impossible
    forbidden = score_matrix <= invalid_score / 2.0
    cell_ubs[forbidden.ravel()] = 0.0

    # Row-sum constraint: sum_s x[p, s] <= R for each p
    # each prototype used at most R times (this just defines which variables to consider, bounds later)
    row_sums = np.kron(np.eye(P), np.ones((1, n_slots)))  # (P, N)

    # Col-sum constraint: sum_p x[p, s] = 1 for each s
    # each slot filled exactly once (this just defines which variables to consider, bounds later)
    col_sums = np.kron(np.ones((1, P)), np.eye(n_slots))  # (n_slots, N)

    # row-sums between 0 and R, col-sums = 1
    constraints = LinearConstraint(
        A=np.vstack([row_sums, col_sums]),  # linear equations defining constraints
        lb=np.concatenate([np.zeros(P), np.ones(n_slots) * R]),  # type: ignore
        ub=np.concatenate([np.ones(P), np.ones(n_slots)]),  # type: ignore
    )

    result = milp(
        c=c_obj,
        constraints=constraints,
        integrality=np.ones(N),
        bounds=Bounds(lb=cell_lbs, ub=cell_ubs),  # type: ignore
    )

    if not result.success:
        raise RuntimeError(f"MILP solver failed: {result.message}")

    # map flat assignments back to (proto, class) pairs
    x = np.round(result.x).reshape(P, n_slots)  # (P, n_slots)

    assignment_matrix = np.zeros((P, C), dtype=np.int64)
    selected_indices_by_class = [[] for _ in range(C)]
    objective_value = 0.0

    proto_indices, slot_indices = np.where(x == 1)
    for proto_idx, slot_idx in zip(proto_indices, slot_indices):
        c = int(slot_to_class[slot_idx])
        assignment_matrix[proto_idx, c] = 1
        selected_indices_by_class[c].append(int(proto_idx))
        objective_value += float(M[proto_idx, c])

    selected_indices_by_class = [sorted(v) for v in selected_indices_by_class]

    for c, v in enumerate(selected_indices_by_class):
        if len(v) != K:
            raise RuntimeError(f"Class {c} received {len(v)} prototypes, expected {K}")

    selected_indices_by_class = np.asarray(selected_indices_by_class, dtype=np.int64)

    return PrototypeILPAssignmentResult(
        selected_indices_by_class=selected_indices_by_class,
        assignment_matrix=assignment_matrix,
        association_matrix=M.astype(np.float32),
        objective_value=objective_value,
        valid_class_mask=valid_class_mask,
    )
