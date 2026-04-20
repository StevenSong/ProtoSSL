import json
import os
from argparse import ArgumentParser

import numpy as np
from fastdigest import TDigest
from tqdm import tqdm
from wfdb import rdsamp

from pass_pclr.datasets import get_heedb_metadata


# estimate per-lead 0.1st and 99.9th percentiles using t-digest
def get_lowers_uppers(paths: list[str]) -> tuple[list[float], list[float]]:
    print("Estimating per-lead 0.1st and 99.9th percentiles")
    per_lead_digest = [TDigest() for lead in range(12)]
    for fpath in tqdm(paths):
        x: np.ndarray
        x, meta = rdsamp(fpath)
        if x.shape[0] == 2500:
            # source data is 250 Hz
            # so duplicate for equal weighting against 500 Hz data
            # as fastdigest does not support weighted updates
            x = x.repeat(2, axis=0)
        for i in range(12):
            digest = per_lead_digest[i]
            lead = x[:, i]
            digest.batch_update(lead)
    lowers = [digest.percentile(0.1) for digest in per_lead_digest]
    uppers = [digest.percentile(99.9) for digest in per_lead_digest]
    return lowers, uppers


# cannot use t-digest to estimate trimmed std dev so iterate over dataset again to find trimmed stats
# use welford's algorithm for on-line calculation of variance
def get_trimmed_mean_std(
    paths: list[str],
    lowers: list[float],
    uppers: list[float],
) -> tuple[list[float], list[float]]:
    print("Computing per-lead clipped means and stds")
    per_lead_aggs = [(0, 0, 0) for lead in range(12)]  # mean, var, n
    for fpath in tqdm(paths):
        x: np.ndarray
        x, meta = rdsamp(fpath)
        x = np.clip(x, lowers, uppers)
        if x.shape[0] == 2500:
            # source data is 250 Hz
            # so duplicate for equal weighting against 500 Hz data
            # as fastdigest does not support weighted updates
            x = x.repeat(2, axis=0)
        x_mean = x.mean(axis=0)
        x_var = x.var(axis=0, ddof=1)
        for i in range(12):
            agg = per_lead_aggs[i]
            mean_var_n = (x_mean[i], x_var[i], 5000)
            per_lead_aggs[i] = parallel_welford_update(agg, mean_var_n)
    trimmed_means = [m for m, v, n in per_lead_aggs]
    trimmed_stds = [v**0.5 for m, v, n in per_lead_aggs]
    # unwrap numpy scalars if present?
    trimmed_means = np.asarray(trimmed_means).tolist()
    trimmed_stds = np.asarray(trimmed_stds).tolist()
    return trimmed_means, trimmed_stds


def parallel_welford_update(
    mean_var_n_A: tuple[float, float, int],
    mean_var_n_B: tuple[float, float, int],
) -> tuple[float, float, int]:
    # given the need to update the mean/var by 5k values, very slow to do single welford updates with python loop
    # instead, treat 5k values as batch and use numpy vectoried mean/var to merge with running mean/var (parallel welford)
    # inspired by https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # and https://gist.github.com/DevSlem/555e7caf4b843741682fbff64ae1cf15
    mean_a, var_a, n_a = mean_var_n_A
    mean_b, var_b, n_b = mean_var_n_B
    n = n_a + n_b
    M2_a = var_a * (n_a - 1)
    M2_b = var_b * (n_b - 1)

    delta = mean_a - mean_b
    M2 = M2_a + M2_b + delta**2 * n_a * n_b / n
    var = M2 / (n - 1)
    mean = (mean_a * n_a + mean_b * n_b) / n
    return mean, var, n


def run(dataset_path: str, output_path: str, subset: str, limit_n: int | None = None):
    os.makedirs(output_path, exist_ok=True)
    df = get_heedb_metadata(dataset_path)
    df = df[df["split"] == "train"]  # only derive stats over train

    if subset in {"mgb", "emory"}:
        df = df[df["source"] == subset]
    elif subset == "both":
        pass
    else:
        raise ValueError(f"Unknown how to handle {subset} subset")

    if limit_n is not None:
        print(f"!!! TRUNCATING TO {limit_n} SAMPLES, IS THIS A DEV RUN? !!!")
        df = df.iloc[:limit_n]
    paths = list(df["full_path"])
    lowers, uppers = get_lowers_uppers(paths)
    with open(
        os.path.join(output_path, f"{subset}_heedb_lowers_uppers.json"), "w"
    ) as f:
        json.dump(
            {
                "lowers": lowers,
                "uppers": uppers,
            },
            f,
            indent=4,
        )
    means, stds = get_trimmed_mean_std(paths, lowers, uppers)
    with open(
        os.path.join(output_path, f"{subset}_heedb_clipped_means_stds.json"), "w"
    ) as f:
        json.dump(
            {
                "clipped_means": means,
                "clipped_stds": stds,
            },
            f,
            indent=4,
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--subset", choices=["mgb", "emory", "both"], required=True)
    parser.add_argument("--limit_n", type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run(args.dataset_path, args.output_path, args.subset, args.limit_n)
