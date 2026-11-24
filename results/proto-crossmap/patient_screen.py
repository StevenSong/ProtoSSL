"""Patient-level screen: imputed PTB-XL statements vs EchoNext echo labels.

See PRESPEC-patient-screen.md. Two stages, run from results/proto-crossmap:

    nice python patient_screen.py discover   # EchoNext train -> patient-screen/{discovery,carried}.csv
    nice python patient_screen.py confirm    # EchoNext val+test -> patient-screen/confirmation.csv
"""

import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import patsy  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from protossl.datasets._ptbxl_dataset import get_ptbxl_labels  # noqa: E402
from protossl.defines import ECHONEXT_COMPOSITE_TARGET, ECHONEXT_TARGETS  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "patient-screen"
SHARED_ROOT = Path(
    os.environ.get("PROTOSSL_SHARED_ROOT", "/shared/gpudata/songs1/protossl-shared")
)
RUNS = SHARED_ROOT / "runs/recur/protossl-outputs-seed42/runs-echonext"
PROBS = {
    "train": RUNS / "protossl-heedb-pila-from-ptbxl/train_probs.npy",
    "test": RUNS / "protossl-heedb-pila-from-ptbxl/test_probs.npy",
    "val": RUNS / "protossl-heedb-pila-from-ptbxl-val/val_probs.npy",
}
DATA_DIR = Path("/opt/gpudata/ecg")
N_JOBS = int(os.environ.get("SCREEN_N_JOBS", 16))  # shared host

E_ALL = list(ECHONEXT_TARGETS)
E_TEST = [e for e in E_ALL if e != ECHONEXT_COMPOSITE_TARGET]
N_POS_MIN = 30
ALPHA = 0.05
OR_MIN = 1.1
MAX_CARRY = 20
MAX_CONTROL = 10
EXCLUDED = {("PEff", "ISCAS")}  # examined on val+test before this prespec
# statements examined against PEff and RVD on val+test in the crossmap's post hoc section
SEEN_VALTEST = {
    (e, s)
    for e in ["PEff", "RVD"]
    for s in [
        "ISCAS",
        "ISCAN",
        "ISCAL",
        "ISCLA",
        "ISCIN",
        "ISCIL",
        "ISC_",
        "INVT",
        "NT_",
        "NDT",
        "TAB_",
        "STE_",
        "PACE",
        "LVOLT",
        "LOWT",
        "ASMI",
        "INJAS",
        "RVH",
        "LPFB",
        "CRBBB",
        "AFIB",
        "STACH",
    ]
}
COV = (
    "bs(age_at_ecg, df=4) + C(sex) + bs(ventricular_rate, df=4) + pr_f + pr_missing"
    " + bs(qrs_duration, df=4) + bs(qt_f, df=4) + bs(acquisition_year, df=4)"
)


def load_split(splits: list[str], train_stats: dict | None = None):
    meta = pd.read_csv(DATA_DIR / "echonext/EchoNext_metadata_100k.csv")
    P = json.loads(
        (RUNS / "protossl-heedb-pila-from-ptbxl/label_names.json").read_text()
    )
    assert (
        json.loads(
            (RUNS / "protossl-heedb-pila-from-ptbxl-val/label_names.json").read_text()
        )
        == P
    )
    frames, probs = [], []
    for sp in splits:
        m = meta[meta["split"] == sp].reset_index(drop=True)
        p = np.load(PROBS[sp]).astype(np.float64)
        assert len(m) == len(p), sp
        frames.append(m)
        probs.append(p)
    d = pd.concat(frames, ignore_index=True)
    p = np.concatenate(probs).clip(1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    if train_stats is None:
        train_stats = {
            "mu": lg.mean(0),
            "sd": lg.std(0),
            "pr_median": float(d["pr_interval"].median()),
            "qt_median": float(d["qt_corrected"].median()),
        }
    Z = (lg - train_stats["mu"]) / train_stats["sd"]
    d["pr_missing"] = d["pr_interval"].isna().astype(float)
    d["pr_f"] = d["pr_interval"].fillna(train_stats["pr_median"])
    d["qt_f"] = d["qt_corrected"].fillna(train_stats["qt_median"])
    X_cov = np.asarray(patsy.dmatrix(COV, d, return_type="dataframe"))
    Y = d[[ECHONEXT_TARGETS[e] for e in E_ALL]].to_numpy().astype(float)
    return d, P, Z, X_cov, Y, train_stats


def fit(y, X, groups=None):
    """Logit; returns (beta, se, p) of the last column, or NaNs if it fails."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kw = (
                {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
                if groups is not None
                else {}
            )
            r = sm.Logit(y, X).fit(disp=0, maxiter=200, **kw)
            if not r.mle_retvals.get("converged", True):
                raise RuntimeError("not converged")
            return float(r.params[-1]), float(r.bse[-1]), float(r.pvalues[-1])
        except Exception:  # noqa: BLE001
            return np.nan, np.nan, np.nan


def residual_sd_r2(z_s, Z_adj):
    A = np.column_stack([np.ones(len(z_s)), Z_adj])
    r = z_s - A @ np.linalg.lstsq(A, z_s, rcond=None)[0]
    return float(r.std()), float(1 - r.var() / z_s.var())


def expected_sets(P):
    ex = pd.read_csv(HERE / "expected_pairs.csv")
    level = ex.set_index(["echonext", "ptbxl", "sign"])["level"].to_dict()
    adj = {
        e: sorted(
            set(ex.loc[(ex["echonext"] == e) & (ex["level"] == "expected"), "ptbxl"])
        )
        for e in E_ALL
    }
    for e in adj:
        assert all(s in P for s in adj[e])
    return level, adj


def tier(level, e, s, beta):
    lv = level.get((e, s, "+" if beta > 0 else "-"), "unexpected")
    return {
        "expected": "expected",
        "plausible": "not expected",
        "unexpected": "unexpected",
    }[lv]


def one_pair(e, s, P, Z, X_cov, Y, adj, groups, extra=None):
    y = Y[:, E_ALL.index(e)]
    js = P.index(s)
    ja = [P.index(t) for t in adj[e] if t != s]
    z_s = Z[:, [js]]
    out = {"echonext": e, "ptbxl": s, "n_adj": len(ja)}
    out["b_cov"], out["se_cov"], out["p_cov"] = fit(
        y, np.column_stack([X_cov, z_s]), groups
    )
    out["b_exp"], out["se_exp"], out["p_exp"] = fit(
        y, np.column_stack([X_cov, Z[:, ja], z_s]), groups
    )
    if extra is not None:
        out["b_exp_echo"], out["se_exp_echo"], out["p_exp_echo"] = fit(
            y, np.column_stack([X_cov, Z[:, ja], extra[e], z_s]), groups
        )
    out["resid_sd"], out["r2_adj"] = (
        residual_sd_r2(Z[:, js], Z[:, ja]) if ja else (float(Z[:, js].std()), 0.0)
    )
    return out


def discover():
    d, P, Z, X_cov, Y, stats = load_split(["train"])
    level, adj = expected_sets(P)
    ptbxl_db = pd.read_csv(DATA_DIR / "ptb-xl/ptbxl_database.csv", index_col="ecg_id")
    n_pos = pd.Series(
        get_ptbxl_labels(ptbxl_db[~ptbxl_db["strat_fold"].isin({9, 10})]).sum(0),
        index=P,
    ).astype(int)
    S_TEST = [s for s in P if n_pos[s] >= N_POS_MIN]
    pairs = [(e, s) for e in E_TEST for s in S_TEST]
    print(
        f"discovery: {len(d)} ECGs, {d['patient_key'].nunique()} patients; {len(pairs)} pairs "
        f"({len(E_TEST)} labels x {len(S_TEST)} statements)",
        flush=True,
    )
    groups = d["patient_key"].to_numpy()
    rows = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(one_pair)(e, s, P, Z, X_cov, Y, adj, groups) for e, s in pairs
    )
    res = pd.DataFrame(rows)
    res["n_pos_ptbxl"] = res["ptbxl"].map(n_pos)
    res["or_cov"] = np.exp(res["b_cov"])  # per train SD
    res["or_res"] = np.exp(res["b_exp"] * res["resid_sd"])
    res["z_exp"] = res["b_exp"] / res["se_exp"]
    res["tier"] = [
        tier(level, e, s, b) if np.isfinite(b) else "failed"
        for e, s, b in zip(res["echonext"], res["ptbxl"], res["b_exp"])
    ]
    res["seen_valtest"] = [
        (e, s) in SEEN_VALTEST for e, s in zip(res["echonext"], res["ptbxl"])
    ]
    bonf = ALPHA / len(pairs)
    res["passes"] = (
        (res["p_exp"] < bonf)
        & (res["p_cov"] < bonf)
        & (np.sign(res["b_cov"]) == np.sign(res["b_exp"]))
        & (np.abs(np.log(res["or_res"])) >= np.log(OR_MIN))
    )
    excl = pd.Series(
        [(e, s) in EXCLUDED for e, s in zip(res["echonext"], res["ptbxl"])]
    )
    res["candidate"] = (
        res["passes"] & res["tier"].isin(["unexpected", "not expected"]) & ~excl
    )
    cand = (
        res[res["candidate"]]
        .assign(
            _t=lambda x: (x["tier"] == "not expected").astype(int),
            _z=lambda x: -x["z_exp"].abs(),
        )
        .sort_values(["_t", "_z"])
    )
    ctrl = (
        res[res["passes"] & (res["tier"] == "expected")]
        .assign(_z=lambda x: -x["z_exp"].abs())
        .sort_values("_z")
    )
    carried = pd.concat(
        [
            cand.head(MAX_CARRY).assign(family="candidate"),
            ctrl.head(MAX_CONTROL).assign(family="control"),
        ]
    ).drop(columns=["_t", "_z"])
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "discovery.csv", index=False)
    carried.to_csv(OUT / "carried.csv", index=False)
    np.savez(OUT / "train_stats.npz", **{k: np.asarray(v) for k, v in stats.items()})
    print(f"failed fits: {int(res['b_exp'].isna().sum() + res['b_cov'].isna().sum())}")
    print(
        f"passing rules 1-3: {int(res['passes'].sum())}, by tier:",
        res.loc[res["passes"], "tier"].value_counts().to_dict(),
    )
    print(
        f"candidates: {int(res['candidate'].sum())}; carried {min(MAX_CARRY, len(cand))} + {min(MAX_CONTROL, len(ctrl))} controls"
    )


def confirm():
    carried = pd.read_csv(OUT / "carried.csv")
    st = np.load(OUT / "train_stats.npz")
    stats = {k: st[k] for k in st.files}
    d, P, Z, X_cov, Y, _ = load_split(["val", "test"], train_stats=stats)
    assert d["patient_key"].is_unique
    _, adj = expected_sets(P)
    extra = {e: Y[:, [E_ALL.index(o) for o in E_TEST if o != e]] for e in E_TEST}
    print(
        f"confirmation: {len(d)} ECGs/patients; carrying {len(carried)} pairs",
        flush=True,
    )
    rows = Parallel(n_jobs=min(N_JOBS, len(carried)))(
        delayed(one_pair)(e, s, P, Z, X_cov, Y, adj, None, extra)
        for e, s in zip(carried["echonext"], carried["ptbxl"])
    )
    conf = pd.DataFrame(rows).drop(columns=["resid_sd", "r2_adj"])
    conf = conf.add_suffix("_vt").rename(
        columns={"echonext_vt": "echonext", "ptbxl_vt": "ptbxl"}
    )
    out = carried.merge(conf, on=["echonext", "ptbxl"])
    out["or_res_vt"] = np.exp(out["b_exp_vt"] * out["resid_sd"])
    out["or_res_lo_vt"] = np.exp(
        (out["b_exp_vt"] - 1.96 * out["se_exp_vt"]) * out["resid_sd"]
    )
    out["or_res_hi_vt"] = np.exp(
        (out["b_exp_vt"] + 1.96 * out["se_exp_vt"]) * out["resid_sd"]
    )
    out["or_res_echo_vt"] = np.exp(out["b_exp_echo_vt"] * out["resid_sd"])
    out["or_cov_vt"] = np.exp(out["b_cov_vt"])  # z is in train-SD units
    out["p_holm_vt"] = np.nan
    for fam, g in out.groupby("family"):
        p = g["p_exp_vt"].fillna(1.0).to_numpy()
        out.loc[g.index, "p_holm_vt"] = multipletests(p, method="holm")[1]
    out["confirmed"] = (np.sign(out["b_exp_vt"]) == np.sign(out["b_exp"])) & (
        out["p_holm_vt"] < ALPHA
    )
    out.to_csv(OUT / "confirmation.csv", index=False)
    cols = [
        "family",
        "echonext",
        "ptbxl",
        "tier",
        "or_res",
        "or_res_vt",
        "or_res_lo_vt",
        "or_res_hi_vt",
        "p_holm_vt",
        "confirmed",
        "or_res_echo_vt",
        "seen_valtest",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["discover", "confirm"])
    {"discover": discover, "confirm": confirm}[ap.parse_args().stage]()
