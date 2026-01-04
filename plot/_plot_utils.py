import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn.metrics import auc, roc_curve

MULTITASK_LABELS = [
    "LVEF Lo",
    "LVWT Hi",
    "AS",
    "AR",
    "MR",
    "TR",
    "PR",
    "RVD",
    "PEff",
    "PASP Hi",
    "TRV Hi",
]


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


def _get_task_row(df, task, metrics_file):
    task_data = df[df["Label"] == task]
    if len(task_data) == 0:
        raise ValueError(f"Task {task} missing in {metrics_file}")
    elif len(task_data) > 1:
        raise ValueError(f"Task {task} has more than one entry in {metrics_file}")
    return task_data.iloc[0]


# Function to load all experiment data
def load_experiment_data(runs_dir="runs/", composite_idx=-1):
    """Load metrics from all experiment directories"""
    experiments = {}

    # Get all experiment directories
    for exp_dir in sorted(os.listdir(runs_dir)):
        exp_path = os.path.join(runs_dir, exp_dir)

        # Check if it's a directory
        if os.path.isdir(exp_path):
            metrics_file = os.path.join(exp_path, "metrics.csv")
            probs_file = os.path.join(exp_path, "probs.npy")

            # Check if metrics.csv exists
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)

                # Get AUROC values for multilabel tasks
                multilabel_aurocs = []
                multilabel_auprcs = []
                for task in MULTITASK_LABELS:
                    task_data = _get_task_row(df, task, metrics_file)
                    multilabel_aurocs.append(task_data["AUROC"])
                    multilabel_auprcs.append(task_data["AUPRC"])

                composite_data = _get_task_row(df, "SHD", metrics_file)
                multilabel_avg_data = _get_task_row(
                    df, "Multilabel Averaged", metrics_file
                )

                experiments[exp_dir] = {
                    "multilabel_aurocs": multilabel_aurocs,
                    "multilabel_avg_auroc": multilabel_avg_data["AUROC"],
                    "composite_auroc": composite_data["AUROC"],
                    "multilabel_avg_auprc": multilabel_avg_data["AUPRC"],
                    "composite_auprc": composite_data["AUPRC"],
                    "all_data": df,
                }

            # Check if probs.npy exists
            if os.path.exists(probs_file):
                probs = np.load(probs_file, allow_pickle=True)

                experiments[exp_dir]["y_prob"] = probs[:, composite_idx]

    return experiments


# Create radar chart for multilabel tasks
def plot_radar(
    experiments_styles: dict,
    *,  # enforce kwargs
    title: str = "Multilabel AUROCs",
    legend_title: str | None = None,
):
    experiments = {k: v for k, (v, c, l, m) in experiments_styles.items()}
    colors = [c for k, (v, c, l, m) in experiments_styles.items()]
    line_styles = [l for k, (v, c, l, m) in experiments_styles.items()]
    marker_styles = [m for k, (v, c, l, m) in experiments_styles.items()]

    theta = radar_factory(len(MULTITASK_LABELS), frame="polygon")

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

    ax.set_varlabels(MULTITASK_LABELS)  # type: ignore
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


# Create ROC curves for composite task
def plot_roc(
    experiments_styles: dict,
    y_true: np.ndarray,
    *,  # enforce kwargs
    title: str = "Composite Task (SHD) ROC Curves",
    legend_title: str | None = None,
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


def plot_lift(
    *,  # enforce kwargs
    data: pd.DataFrame,
    metric: str,
    baseline_model: str,
    palette: dict[str, str],
    title: str,
):
    palette = palette.copy()
    min_size = data["Train Size"].min()
    max_size = data["Train Size"].max()
    fig, ax = plt.subplots(figsize=(6, 6))

    mask = data["Model"] == baseline_model
    assert (
        mask.sum() == 1
    ), f"Should only have 1 entry for baseline model: {baseline_model}"
    baseline_row = data[mask].iloc[0]
    nonbaseline_data = data[~mask]

    ax.hlines(
        baseline_row[metric],
        min_size,
        max_size,
        colors=palette.pop(baseline_model),
        linestyles=":",
        label=baseline_model,
    )
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
    # ax.set_ylim((0.4, 0.9))
