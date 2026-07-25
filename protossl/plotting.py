"""Utilities for plotting ECGs from any of the ECG datasets in this package.

Waveforms in the datasets are clipped and z-normalized with per-lead (and, for
HEEDB, per-institution) statistics derived over the respective train sets. To
draw a clinical-style printout the waveforms first have to be put back into
physical units, which is what the helpers here take care of, so that the same
plot can be made for a sample from any dataset.

Typical use::

    from protossl.datasets import PtbxlECGDataset
    from protossl.plotting import plot_ecg

    ds = PtbxlECGDataset(dataset_path=..., split="test", sampling_rate=100)
    fig = plot_ecg(dataset=ds, sample_id=3875, chunk_idx=4)  # highlight window 4
    fig.savefig("case.png", dpi=300)
"""

from typing import Type

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from .datasets import (
    BaseTSDataset,
    CincECGDataset,
    Code15ECGDataset,
    EchoNextECGDataset,
    HeedbECGDataset,
    MimicECGDataset,
    PtbxlECGDataset,
    ZzuECGDataset,
)
from .datasets._mimic_dataset import standardize_lead_order as _MIMIC_LEAD_PERMUTATION
from .defines import (
    CINC_CLIPPED_MEANS,
    CINC_CLIPPED_STDS,
    CODE15_CLIPPED_MEANS,
    CODE15_CLIPPED_STDS,
    HEEDB_EUH_CLIPPED_MEANS,
    HEEDB_EUH_CLIPPED_STDS,
    HEEDB_MGB_CLIPPED_MEANS,
    HEEDB_MGB_CLIPPED_STDS,
    MIMIC_CLIPPED_MEANS,
    MIMIC_CLIPPED_STDS,
    PTBXL_CLIPPED_MEANS,
    PTBXL_CLIPPED_STDS,
    STANDARD_LEAD_ORDER,
    ZZU_CLIPPED_MEANS,
    ZZU_CLIPPED_STDS,
)

# print layout, all datasets emit waveforms in STANDARD_LEAD_ORDER
LEAD_TO_IDX = {name: i for i, name in enumerate(STANDARD_LEAD_ORDER)}
LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEADS = ["II"]

MM_PER_MV = 10  # gain
MM_PER_SEC = 25  # paper speed
ROW_SPACING = 50  # mm between printout rows
SEG_HEIGHT_IN = 1.4  # height of one row of the segment cutout
FULL_SEG_VGAP_REL = (
    0.1  # vertical gap between printout and cutout, relative to printout
)
PANEL_HGAP_REL = 0.15  # horizontal gap between panels, relative to printout width

DEFAULT_DURATION = 10.0  # seconds, all ECG datasets emit 10 second recordings

# per-lead unnormalization stats keyed by dataset class, given in the lead order
# the datasets actually emit. `None` marks data that is not normalized by this
# package (and is therefore already in physical units), a dict marks datasets
# whose samples are normalized with different stats (keyed by `_df["source"]`).
_MIMIC_PERM = np.asarray(_MIMIC_LEAD_PERMUTATION)
_DENORM_STATS: dict[
    Type[BaseTSDataset],
    tuple[list[float], list[float]] | dict[str, tuple[list[float], list[float]]] | None,
] = {
    PtbxlECGDataset: (PTBXL_CLIPPED_MEANS, PTBXL_CLIPPED_STDS),
    CincECGDataset: (CINC_CLIPPED_MEANS, CINC_CLIPPED_STDS),
    Code15ECGDataset: (CODE15_CLIPPED_MEANS, CODE15_CLIPPED_STDS),
    ZzuECGDataset: (ZZU_CLIPPED_MEANS, ZZU_CLIPPED_STDS),
    # MIMIC leads are reordered *after* normalization, so permute the stats the
    # exact same way to keep them aligned with the emitted lead order
    MimicECGDataset: (
        np.asarray(MIMIC_CLIPPED_MEANS)[_MIMIC_PERM].tolist(),
        np.asarray(MIMIC_CLIPPED_STDS)[_MIMIC_PERM].tolist(),
    ),
    HeedbECGDataset: {
        "mgb": (HEEDB_MGB_CLIPPED_MEANS, HEEDB_MGB_CLIPPED_STDS),
        "emory": (HEEDB_EUH_CLIPPED_MEANS, HEEDB_EUH_CLIPPED_STDS),
    },
    # EchoNext waveforms come prenormalized from the source, nothing to undo
    EchoNextECGDataset: None,
}


def resolve_index(
    dataset: BaseTSDataset,
    *,  # enforce kwargs
    index: int | None = None,
    sample_id: int | None = None,
) -> int:
    """Resolve a dataset row from either its positional index or its sample ID.

    :param dataset: dataset holding the sample
    :type dataset: BaseTSDataset
    :param index: positional index into the dataset
    :type index: int | None
    :param sample_id: sample ID (e.g. PTB-XL `ecg_id`), mutually exclusive with `index`
    :type sample_id: int | None
    """
    if (index is None) == (sample_id is None):
        raise ValueError("Pass exactly one of index or sample_id")
    if index is not None:
        return int(index)

    # cache the mapping on the dataset, it is a linear scan over all samples
    lookup = getattr(dataset, "_sample_id_to_idx", None)
    if lookup is None:
        lookup = {int(s): i for i, s in enumerate(dataset.sample_ids.tolist())}
        dataset._sample_id_to_idx = lookup  # type: ignore[attr-defined]
    if int(sample_id) not in lookup:  # type: ignore[arg-type]
        raise KeyError(
            f"sample_id {sample_id} not found in this dataset/split "
            f"(does it belong to a different split?)"
        )
    return lookup[int(sample_id)]  # type: ignore[arg-type]


def infer_sampling_rate(
    dataset: BaseTSDataset,
    duration: float = DEFAULT_DURATION,
) -> int:
    """Infer the sampling rate from the waveform length, all ECG datasets emit
    fixed length recordings of `duration` seconds.
    """
    n_timesteps = dataset.waveforms.shape[-1]
    sampling_rate = n_timesteps / duration
    if sampling_rate != int(sampling_rate):
        raise ValueError(
            f"Could not infer an integer sampling rate from {n_timesteps} timesteps "
            f"over {duration} seconds, pass sampling_rate explicitly"
        )
    return int(sampling_rate)


def get_denorm_stats(
    dataset: BaseTSDataset,
    index: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-lead means/stds used to normalize this sample, shaped (L, 1) for
    broadcasting over (L, T) waveforms. Returns `None` for datasets whose
    waveforms are not normalized by this package.
    """
    for cls in type(dataset).__mro__:
        if cls in _DENORM_STATS:
            stats = _DENORM_STATS[cls]
            break
    else:
        raise ValueError(
            f"No unnormalization stats known for {type(dataset).__name__}, "
            f"pass means/stds explicitly"
        )

    if stats is None:
        return None
    if isinstance(stats, dict):
        # e.g. HEEDB normalizes per source institution
        source = dataset._df.loc[index, "source"]  # type: ignore[attr-defined]
        if source not in stats:
            raise ValueError(
                f"{type(dataset).__name__} sample {index} has unknown source "
                f"'{source}', expected one of {sorted(stats)}"
            )
        stats = stats[source]

    means, stds = stats
    return np.asarray(means)[:, None], np.asarray(stds)[:, None]


def remove_baseline_wander(
    X: np.ndarray,
    sampling_rate: int,
    cutoff: float = 0.5,
    order: int = 1,
) -> np.ndarray:
    """Apply a high-pass Butterworth filter to remove baseline wander.

    No need to apply a low-pass filter if using 100 Hz data.

    :param X: waveform of shape (L, T)
    :type X: np.ndarray
    """
    assert X.ndim == 2
    X = X.astype(np.float64, copy=True)

    b, a = butter(order, cutoff / (sampling_rate / 2), btype="high", analog=False)
    for j in range(X.shape[0]):
        X[j, :] = filtfilt(b, a, X[j, :])

    return X


def get_ecg_waveform(
    *,  # enforce kwargs
    dataset: BaseTSDataset,
    index: int | None = None,
    sample_id: int | None = None,
    denormalize: bool = True,
    remove_baseline: bool = True,
    sampling_rate: int | None = None,
    duration: float = DEFAULT_DURATION,
    means: np.ndarray | list[float] | None = None,
    stds: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Get a single ECG as a (L, T) array in physical units (mV), ready to plot.

    Note that unnormalizing is not exactly the inverse of the dataset transform,
    the source waveforms are clipped before normalization.

    :param dataset: any ECG dataset from `protossl.datasets`
    :type dataset: BaseTSDataset
    :param index: positional index into the dataset, mutually exclusive with `sample_id`
    :type index: int | None
    :param sample_id: sample ID of the ECG, mutually exclusive with `index`
    :type sample_id: int | None
    :param denormalize: undo the per-lead z-normalization applied by the dataset
    :type denormalize: bool
    :param remove_baseline: high-pass filter the waveform to remove baseline wander
    :type remove_baseline: bool
    :param sampling_rate: sampling rate of the waveform, inferred if not given
    :type sampling_rate: int | None
    :param duration: recording length in seconds, used to infer the sampling rate
    :type duration: float
    :param means: per-lead means overriding the dataset's unnormalization stats
    :type means: np.ndarray | list[float] | None
    :param stds: per-lead stds overriding the dataset's unnormalization stats
    :type stds: np.ndarray | list[float] | None
    """
    idx = resolve_index(dataset, index=index, sample_id=sample_id)
    if sampling_rate is None:
        sampling_rate = infer_sampling_rate(dataset, duration)

    waveform = dataset[idx]["waveform"]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    signal = np.asarray(waveform, dtype=np.float64)  # (L, T)
    assert signal.ndim == 2, f"Expected a (L, T) waveform, got {signal.shape}"

    if denormalize:
        if (means is None) != (stds is None):
            raise ValueError("Pass both means and stds, or neither")
        if means is not None and stds is not None:
            stats = np.asarray(means)[:, None], np.asarray(stds)[:, None]
        else:
            stats = get_denorm_stats(dataset, idx)
        if stats is not None:
            signal = (signal * stats[1]) + stats[0]

    if remove_baseline:
        signal = remove_baseline_wander(signal, sampling_rate)

    return signal


def chunk_to_seconds(
    chunk_idx: int,
    *,  # enforce kwargs
    chunk_len: int,
    chunk_overlap: float,
    sampling_rate: int,
) -> tuple[float, float]:
    """Convert a prototype chunk index into a (start, end) window in seconds.

    Mirrors the windowing in `PrototypeEncoder.forward` for partial prototypes,
    where chunks are taken with `unfold(2, partial_len, partial_len * (1 - overlap))`.

    :param chunk_idx: index of the chunk, e.g. from `test_chunks.npy`
    :type chunk_idx: int
    :param chunk_len: window length in timesteps (the model's `partial_len`)
    :type chunk_len: int
    :param chunk_overlap: fractional overlap between windows (the model's `partial_overlap`)
    :type chunk_overlap: float
    :param sampling_rate: sampling rate of the waveform
    :type sampling_rate: int
    """
    step = int(chunk_len * (1 - chunk_overlap))
    start = (chunk_idx * step) / sampling_rate
    return start, start + (chunk_len / sampling_rate)


def create_ecg_axes(
    n_panels: int = 1,
    *,  # enforce kwargs
    show_segment: bool = True,
    duration: float = DEFAULT_DURATION,
    lead_layout: list[list[str]] = LEAD_LAYOUT,
    rhythm_leads: list[str] = RHYTHM_LEADS,
) -> tuple[plt.Figure, list[tuple[plt.Axes, np.ndarray | None]]]:
    """Make a figure with `n_panels` side-by-side printout/cutout panels.

    Each panel is laid out as::

        [.......... full printout ..........]
        [........... vertical gap ..........]
        [ seg ][ seg ][ seg ][ seg ]  < cutout, one subplot per lead of the layout

    :param n_panels: number of ECGs to show side by side
    :type n_panels: int
    :param show_segment: include the segment cutout axes below each printout
    :type show_segment: bool
    :returns: the figure and, per panel, its printout axes and cutout axes grid
    """
    if n_panels < 1:
        raise ValueError(f"Need at least one panel, got {n_panels}")

    n_seg_rows = len(lead_layout)
    n_seg_cols = len(lead_layout[0])
    n_full_rows = n_seg_rows + len(rhythm_leads)

    printout_width_in = (duration * MM_PER_SEC) / 25.4
    printout_height_in = (n_full_rows * ROW_SPACING) / 25.4

    fig_width_in = printout_width_in * (n_panels + PANEL_HGAP_REL * (n_panels - 1))
    fig_height_in = printout_height_in
    height_ratios = [printout_height_in]
    if show_segment:
        fig_height_in += printout_height_in * FULL_SEG_VGAP_REL
        fig_height_in += SEG_HEIGHT_IN * n_seg_rows
        height_ratios += [FULL_SEG_VGAP_REL * printout_height_in]
        height_ratios += [SEG_HEIGHT_IN] * n_seg_rows

    # per panel: n_seg_cols equal width columns, separated by a gap column
    width_ratios: list[float] = []
    for panel in range(n_panels):
        if panel > 0:
            width_ratios.append(PANEL_HGAP_REL)
        width_ratios += [1 / n_seg_cols] * n_seg_cols

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    gs = fig.add_gridspec(
        nrows=len(height_ratios),
        ncols=len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )

    panels: list[tuple[plt.Axes, np.ndarray | None]] = []
    for panel in range(n_panels):
        col_offset = panel * (n_seg_cols + 1)  # +1 for the gap column
        full_ax = fig.add_subplot(gs[0, col_offset : col_offset + n_seg_cols])
        if not show_segment:
            panels.append((full_ax, None))
            continue

        seg_axs: list[list[plt.Axes]] = []
        for i in range(n_seg_rows):
            seg_row: list[plt.Axes] = []
            seg_axs.append(seg_row)
            for j in range(n_seg_cols):
                ax = fig.add_subplot(gs[2 + i, col_offset + j])
                if i != 0 or j != 0:
                    ax.sharex(seg_axs[0][0])
                seg_row.append(ax)
        panels.append((full_ax, np.asarray(seg_axs)))

    return fig, panels


def plot_full(
    full_signal: np.ndarray,
    ax: plt.Axes,
    *,  # enforce kwargs
    sampling_rate: int,
    duration: float = DEFAULT_DURATION,
    gain: float = MM_PER_MV,
    lead_layout: list[list[str]] = LEAD_LAYOUT,
    rhythm_leads: list[str] = RHYTHM_LEADS,
    lead_to_idx: dict[str, int] = LEAD_TO_IDX,
) -> None:
    """Draw a clinical style printout of a (L, T) waveform onto `ax`."""
    n_seg_cols = len(lead_layout[0])
    n_full_rows = len(lead_layout) + len(rhythm_leads)
    n_timesteps = full_signal.shape[1]
    assert n_timesteps == round(duration * sampling_rate)
    full_signal = full_signal.T  # (T, L)

    ax.set_xlim(0, duration)
    ax.set_ylim(0, n_full_rows * ROW_SPACING)
    ax.axis("off")

    # draw paper grid, 1 small box = 0.04 sec x 1 mm, 1 large box = 5 small boxes
    for x in np.arange(0, duration + 0.04, 0.04):
        ax.axvline(x, color="pink", linewidth=0.5, zorder=0)
    for y in np.arange(0, n_full_rows * ROW_SPACING + 0.1, 1):
        is_large = (y * 0.1) % 0.5 == 0
        ax.axhline(
            y,
            color="pink" if not is_large else "red",
            linewidth=0.5 if not is_large else 1.0,
            zorder=0,
        )
    for x in np.arange(0, duration + 0.2, 0.2):
        ax.axvline(x, color="red", linewidth=1.0, zorder=0)

    label_fontsize = 12
    row_baseline = n_full_rows * ROW_SPACING - ROW_SPACING / 2
    col_duration = duration / n_seg_cols
    col_timesteps = n_timesteps // n_seg_cols

    for row_idx, lead_row in enumerate(lead_layout):
        for col_idx, lead in enumerate(lead_row):
            lead_idx = lead_to_idx[lead]
            start = col_idx * col_timesteps
            end = start + col_timesteps
            signal = full_signal[start:end, lead_idx] * gain
            t = np.linspace(
                col_idx * col_duration,
                (col_idx + 1) * col_duration,
                col_timesteps,
            )
            v_offset = row_baseline - row_idx * ROW_SPACING
            ax.plot(t, signal + v_offset, color="black", linewidth=1.0)
            ax.text(
                t[0] + 0.1,
                v_offset + 16,
                lead,
                fontsize=label_fontsize,
                fontweight="bold",
            )

    # rhythm leads: the full recording
    t = np.linspace(0, duration, n_timesteps)
    for i, lead in enumerate(rhythm_leads):
        v_offset = row_baseline - (len(lead_layout) + i) * ROW_SPACING
        signal = full_signal[:, lead_to_idx[lead]] * gain
        ax.plot(t, signal + v_offset, color="black", linewidth=1.0)
        ax.text(
            0.1,
            v_offset + 16,
            lead,
            fontsize=label_fontsize,
            fontweight="bold",
        )


def plot_segment(
    segment_signal: np.ndarray,
    axs: np.ndarray,
    *,  # enforce kwargs
    sampling_rate: int,
    lead_layout: list[list[str]] = LEAD_LAYOUT,
    lead_to_idx: dict[str, int] = LEAD_TO_IDX,
) -> None:
    """Draw a cutout of a (L, T) segment, one subplot per lead of the layout."""
    n_rows = len(lead_layout)
    n_cols = len(lead_layout[0])
    assert axs.shape == (n_rows, n_cols)  # to match layout
    t_steps = segment_signal.shape[1]
    t_segment = np.linspace(0, t_steps / sampling_rate, t_steps)

    for row in range(n_rows):
        for col in range(n_cols):
            lead = lead_layout[row][col]
            i = lead_to_idx[lead]
            ax: plt.Axes = axs[row, col]
            ax.plot(t_segment, segment_signal[i], color="black", linewidth=1.0)
            ax.set_xlim(0, t_segment[-1])
            ax.set_ylim(
                np.min(segment_signal[i]) - 0.5, np.max(segment_signal[i]) + 0.5
            )

            # draw lead label to the left of the plot
            ax.text(
                x=-0.1,
                y=0.8,
                s=lead,
                fontsize=10,
                fontweight="bold",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )

            # ECG grid
            for x in np.arange(t_segment[0], t_segment[-1] + 0.04, 0.04):
                ax.axvline(x, color="pink", linewidth=0.3, alpha=0.5)
            for x in np.arange(t_segment[0], t_segment[-1], 0.2):
                ax.axvline(x, color="red", linewidth=0.5, alpha=0.7)
            ax.axhline(0, color="red", linewidth=0.5, alpha=0.5)

            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)


def plot_ecg(
    *,  # enforce kwargs
    dataset: BaseTSDataset | None = None,
    index: int | None = None,
    sample_id: int | None = None,
    signal: np.ndarray | None = None,
    highlight: tuple[float, float] | None = None,
    chunk_idx: int | None = None,
    chunk_len: int | None = None,
    chunk_overlap: float = 0.5,
    show_segment: bool = True,
    title: str | None = None,
    title_kwargs: dict | None = None,
    full_ax: plt.Axes | None = None,
    seg_axs: np.ndarray | None = None,
    duration: float = DEFAULT_DURATION,
    sampling_rate: int | None = None,
    denormalize: bool = True,
    remove_baseline: bool = True,
    gain: float = MM_PER_MV,
    lead_layout: list[list[str]] = LEAD_LAYOUT,
    rhythm_leads: list[str] = RHYTHM_LEADS,
    means: np.ndarray | list[float] | None = None,
    stds: np.ndarray | list[float] | None = None,
) -> plt.Figure:
    """Plot a single ECG from any ECG dataset as a clinical style printout,
    optionally highlighting a window and calling it out as a cutout below.

    :param dataset: any ECG dataset from `protossl.datasets`, optional only if
        both `signal` and `sampling_rate` are given
    :type dataset: BaseTSDataset | None
    :param index: positional index into the dataset, mutually exclusive with `sample_id`
    :type index: int | None
    :param sample_id: sample ID of the ECG, mutually exclusive with `index`
    :type sample_id: int | None
    :param signal: preloaded (L, T) waveform in mV, skips loading from the dataset
    :type signal: np.ndarray | None
    :param highlight: window to highlight as (start, end) in seconds
    :type highlight: tuple[float, float] | None
    :param chunk_idx: prototype chunk to highlight, mutually exclusive with `highlight`
    :type chunk_idx: int | None
    :param chunk_len: window length in timesteps (the model's `partial_len`),
        defaults to one second
    :type chunk_len: int | None
    :param chunk_overlap: fractional overlap between windows (the model's `partial_overlap`)
    :type chunk_overlap: float
    :param show_segment: draw the cutout of the highlighted window below the printout
    :type show_segment: bool
    :param full_ax: existing axes to draw the printout on, e.g. from `create_ecg_axes`,
        a new single panel figure is made if not given
    :type full_ax: plt.Axes | None
    :param seg_axs: existing cutout axes matching the lead layout, required when
        drawing the cutout onto `full_ax`
    :type seg_axs: np.ndarray | None
    :returns: the figure that was drawn on
    """
    if dataset is None and (signal is None or sampling_rate is None):
        raise ValueError("Pass a dataset, or both signal and sampling_rate")
    if sampling_rate is None:
        assert dataset is not None
        sampling_rate = infer_sampling_rate(dataset, duration)
    if chunk_idx is not None:
        if highlight is not None:
            raise ValueError("Pass exactly one of highlight or chunk_idx")
        highlight = chunk_to_seconds(
            chunk_idx,
            chunk_len=chunk_len if chunk_len is not None else sampling_rate,
            chunk_overlap=chunk_overlap,
            sampling_rate=sampling_rate,
        )
    show_segment = show_segment and highlight is not None

    if signal is None:
        assert dataset is not None
        signal = get_ecg_waveform(
            dataset=dataset,
            index=resolve_index(dataset, index=index, sample_id=sample_id),
            denormalize=denormalize,
            remove_baseline=remove_baseline,
            sampling_rate=sampling_rate,
            duration=duration,
            means=means,
            stds=stds,
        )

    made_fig = full_ax is None
    if full_ax is None:
        fig, panels = create_ecg_axes(
            1,
            show_segment=show_segment,
            duration=duration,
            lead_layout=lead_layout,
            rhythm_leads=rhythm_leads,
        )
        full_ax, seg_axs = panels[0]
    else:
        fig = full_ax.get_figure()
        if show_segment and seg_axs is None:
            raise ValueError("Pass seg_axs alongside full_ax to draw the cutout")

    plot_full(
        signal,
        full_ax,
        sampling_rate=sampling_rate,
        duration=duration,
        gain=gain,
        lead_layout=lead_layout,
        rhythm_leads=rhythm_leads,
    )

    if highlight is not None:
        highlight_start, highlight_end = highlight
        full_ax.axvspan(
            highlight_start, highlight_end, color="blue", alpha=0.2, zorder=1
        )

    if show_segment:
        assert highlight is not None and seg_axs is not None
        ymin, _ = full_ax.get_ylim()
        raw_start = max(int(highlight[0] * sampling_rate), 0)
        raw_end = min(int(highlight[1] * sampling_rate), signal.shape[1])
        plot_segment(
            signal[:, raw_start:raw_end],
            seg_axs,
            sampling_rate=sampling_rate,
            lead_layout=lead_layout,
        )

        # callout lines from the highlighted window to the cutout
        for x_full, x_seg, seg_ax in [
            (highlight[0], 0, seg_axs[0, 0]),  # bottom left of window to cutout
            (highlight[1], 1, seg_axs[0, -1]),  # bottom right of window to cutout
        ]:
            fig.add_artist(
                patches.ConnectionPatch(
                    xyA=(x_full, ymin),
                    coordsA="data",
                    xyB=(x_seg, 1),  # top corner of the cutout
                    coordsB="axes fraction",
                    axesA=full_ax,
                    axesB=seg_ax,
                    color="black",
                    linewidth=1.5,
                )
            )

    if title is not None:
        full_ax.set_title(title, **(title_kwargs or {}))

    if made_fig:
        fig.tight_layout()
    return fig
