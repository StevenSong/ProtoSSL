import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn.metrics import auc, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

PTBXL_GROUP_CSV_PATH = "/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet/scp_statementsRegrouped2.csv"
ECHONEXT_GROUP_CSV_PATH = "/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet/echonext_label_groups.csv"
ECHONEXT_ALIAS_PATH = "/opt/gpudata/steven/ecg-prototype-fm/configs/targets.yaml"

LABEL_SET_T = Literal["ptbxl", "echonext"]


# Radar chart implementation from matplotlib example
# this is duplicated from https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
            for label, angle in zip(self.get_xticklabels(), np.degrees(theta)):
                angle = angle % 360

                if angle == 0 or angle == 270:
                    label.set_horizontalalignment("center")
                elif 0 < angle < 180:
                    label.set_horizontalalignment("right")
                else:
                    label.set_horizontalalignment("left")

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()  # type: ignore
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


# Create radar chart for multilabel tasks
def plot_radar(
    experiments_styles: dict,
    *,  # enforce kwargs
    labels: list[str],
    title: str = "Multilabel AUROCs",
    legend_title: str | None = None,
    save_path: str | None = None,
):
    experiments = {k: v for k, (v, c, l, m) in experiments_styles.items()}
    colors = [c for k, (v, c, l, m) in experiments_styles.items()]
    line_styles = [l for k, (v, c, l, m) in experiments_styles.items()]
    marker_styles = [m for k, (v, c, l, m) in experiments_styles.items()]

    theta = radar_factory(len(labels), frame="polygon")

    # Prepare data for radar chart
    experiment_names = list(experiments.keys())
    multilabel_data = [
        experiments[exp]["multilabel_aurocs"] for exp in experiment_names
    ]

    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="radar"))

    # Plot each experiment
    for i, (exp_name, data) in enumerate(zip(experiment_names, multilabel_data)):
        ax.plot(
            theta,
            data,
            color=colors[i],
            linestyle=line_styles[i],
            marker=marker_styles[i],
            linewidth=1.5,
            label=exp_name,
        )
        ax.fill(theta, data, facecolor=colors[i], alpha=0.25)

    ax.set_varlabels(labels)  # type: ignore
    ax.set_title(
        title,
        # weight="bold",
        size="large",
        position=(0.5, 1.1),
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.set_rlim(0.40, 0.9)  # type: ignore
    ax.set_rgrids([0.5, 0.6, 0.7, 0.8, 0.9])  # type: ignore
    ax.legend(loc=(0.9, 0.95), title=legend_title)
    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path)


# Create ROC curves for composite task
def plot_roc(
    experiments_styles: dict,
    y_true: np.ndarray,
    *,  # enforce kwargs
    title: str = "Composite Task (SHD) ROC Curves",
    legend_title: str | None = None,
    save_path: str | None = None,
):
    experiments = {k: v for k, (v, c, l, m) in experiments_styles.items()}
    colors = [c for k, (v, c, l, m) in experiments_styles.items()]

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, (exp_name, exp_data) in enumerate(experiments.items()):
        y_prob = exp_data["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            label=f"{exp_name} (AUC = {roc_auc:.3f})",
            linewidth=2,
            color=colors[i],
        )

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", title=legend_title)
    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path)


def plot_lift(
    *,  # enforce kwargs
    data: pd.DataFrame,
    metric: str,
    baseline_model: str | None = None,
    palette: dict[str, str],
    title: str,
    ylim: tuple[float, float] | None = None,
    save_path: str | None = None,
):
    palette = palette.copy()
    min_size = data["Train Size"].min()
    max_size = data["Train Size"].max()
    fig, ax = plt.subplots(figsize=(6, 6))

    if baseline_model is not None:
        mask = data["Model"] == baseline_model
        assert (
            mask.sum() == 1
        ), f"Should only have 1 entry for baseline model: {baseline_model}"
        baseline_row = data[mask].iloc[0]
        ax.hlines(
            baseline_row[metric],
            min_size,
            max_size,
            colors=palette.pop(baseline_model),
            linestyles=":",
            label=baseline_model,
        )
        nonbaseline_data = data[~mask]
    else:
        nonbaseline_data = data

    sns.lineplot(
        nonbaseline_data,
        x="Train Size",
        y=metric,
        hue="Model",
        palette=palette,
        hue_order=list(palette.keys()),
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlim((min_size, max_size))
    ax.set_title(title)
    ax.legend(loc="lower right")
    if ylim is not None:
        ax.set_ylim(ylim)
    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path)


def get_labels(
    *,  # enforce kwargs
    label_set: LABEL_SET_T,
    label_group: int,
) -> list[str]:
    if label_set == "ptbxl":
        group_csv_path = PTBXL_GROUP_CSV_PATH
    elif label_set == "echonext":
        group_csv_path = ECHONEXT_GROUP_CSV_PATH
    else:
        raise ValueError(f"Bad label_set={label_set}")

    label_groups = pd.read_csv(group_csv_path, index_col=0)
    mask = label_groups["prototype_category"] == label_group
    labels = label_groups[mask].index.to_list()
    if label_set == "echonext":
        with open(ECHONEXT_ALIAS_PATH, "r") as f:
            aliases = yaml.safe_load(f)["target_columns"]
            aliases = {v: k for k, v in aliases.items()}
        labels = [aliases[x] for x in labels]
    return labels


def calc_ppc(
    *,  # enforce kwargs
    n_labels: int,
    n_prototypes: int,
    fallback_ppc: int = 5,
) -> int:
    if n_prototypes % n_labels != 0:
        print(
            f"Number of prototypes ({n_prototypes}) is not evenly divisble by number of labels ({n_labels})"
        )
        print(
            f"If you are doing label-free projection from pretraining over a different label set, you can ignore this warning"
        )
        ppc = fallback_ppc
    else:
        ppc = n_prototypes // n_labels
    return ppc


def plot_prototype_similarity_heatmap(
    *,  # enforce kwargs
    ckpt_path: str,
    title: str,
    label_set: LABEL_SET_T,
    label_group: int,
    save_path: str | None = None,
):
    labels = get_labels(
        label_set=label_set,
        label_group=label_group,
    )
    sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]

    # for 4d prototypes (from 2D partial or global convs)
    prototypes = sd["model.prototype_vectors"]
    prototypes = prototypes.view(prototypes.shape[0], -1)

    sims = cosine_similarity(prototypes)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(sims, ax=ax)
    ax.set_title(f"{title} (Avg Cos Sim = {sims.mean():0.3f})")
    ax.set_xlabel("Prototype Index")
    ax.set_ylabel("Prototype Index")
    n_prototypes = sims.shape[0]
    n_labels = len(labels)
    ppc = calc_ppc(n_labels=n_labels, n_prototypes=n_prototypes)
    ticks = list(range(0, n_prototypes + 1, ppc))
    ax.set_xticks(ticks=ticks, labels=ticks)  # type: ignore
    ax.set_yticks(ticks=ticks)
    ax.set_yticklabels(labels=ticks, rotation=0)  # type: ignore
    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path)


def plot_prototype_label_indicators(
    *,  # enforce kwargs
    metadata_path: str,
    title: str,
    label_set: LABEL_SET_T,
    label_group: int,
    save_path: str | None = None,
):
    labels = get_labels(
        label_set=label_set,
        label_group=label_group,
    )
    metadata = pd.read_json(metadata_path).T
    indicators = np.asarray(metadata["true_labels"].to_list())
    n_prototypes, n_labels = indicators.shape
    ppc = calc_ppc(n_labels=n_labels, n_prototypes=n_prototypes)
    assert n_labels == len(
        labels
    ), "Number of provided labels does not match number of loaded classes in prototype metadata"
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(indicators.T, ax=ax, cbar=False, cmap="viridis")
    ax.grid(axis="y")
    xticks = list(range(0, n_prototypes + 1, ppc))
    ax.set_xticks(ticks=xticks, labels=xticks)  # type: ignore
    ax.set_yticks(range(0, n_labels + 1))
    ax.set_yticks([x + 0.5 for x in range(0, n_labels)], minor=True)
    ax.set_yticklabels(labels, minor=True)
    ax.tick_params(
        axis="y",
        which="both",
        length=0,
    )
    ax.set_xlabel("Prototype Index")
    n_unique = metadata["ecg_id"].nunique()
    ax.set_title(f"{title} (N Unique Samples = {n_unique})")
    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path)
