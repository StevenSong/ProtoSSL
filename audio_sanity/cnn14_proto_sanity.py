import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.models import PrototypeClassifier


DATASET_PATH = "/gpfs/data/bbj-lab/data/audioset/audioset"
OUT_DIR = "/gpfs/data/bbj-lab/users/sethis/ecg-prototype-fm/audio_sanity/cnn14_partial_proto_explanations_out"

SAMPLE_IDX = 0
TOPK_PROTOS = 6
SEARCH_BATCHES = 20
BATCH_SIZE = 4

CKPT_PATH = None
# Example:
# CKPT_PATH = "/gpfs/data/bbj-lab/users/sethis/ecg-prototype-fm/results_audioset/.../best.ckpt"

PROTO_H = 1
PROTO_W = 1
N_PROTOTYPES = 32


def save_waveform_plot(x: torch.Tensor, out_path: str, title: str, sample_rate: int = 32000):
    x = x.squeeze().cpu().numpy()
    t = [i / sample_rate for i in range(len(x))]
    plt.figure(figsize=(12, 3))
    plt.plot(t, x, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_spec_plot(spec: torch.Tensor, out_path: str, title: str):
    spec = spec.cpu().numpy()
    plt.figure(figsize=(12, 4))
    im = plt.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, 10.0, 0, spec.shape[0]],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Mel bin")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_heatmap(arr: torch.Tensor, out_path: str, title: str, xlabel: str, ylabel: str):
    arr = arr.cpu().numpy()
    plt.figure(figsize=(8, 4))
    im = plt.imshow(arr, origin="lower", aspect="auto", cmap="viridis")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_proto_strength_bar(strengths: torch.Tensor, out_path: str, title: str):
    strengths = strengths.cpu().numpy()
    xs = list(range(len(strengths)))
    plt.figure(figsize=(12, 4))
    plt.bar(xs, strengths)
    plt.xlabel("Prototype index")
    plt.ylabel("Max similarity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def overlay_heatmap_on_spec(spec: torch.Tensor, heatmap: torch.Tensor, out_path: str, title: str):
    spec_np = spec.cpu().numpy()
    heat_np = heatmap.cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.imshow(
        spec_np,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, 10.0, 0, spec_np.shape[0]],
    )
    im = plt.imshow(
        heat_np,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        alpha=0.45,
        extent=[0, 10.0, 0, spec_np.shape[0]],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Mel bin")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def overlay_box_on_spec(
    spec: torch.Tensor,
    out_path: str,
    title: str,
    t0: float,
    t1: float,
    f0: float,
    f1: float,
):
    spec_np = spec.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(
        spec_np,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, 10.0, 0, spec_np.shape[0]],
    )
    rect = patches.Rectangle(
        (t0, f0),
        t1 - t0,
        f1 - f0,
        linewidth=2,
        edgecolor="cyan",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def extract_patch_bounds_from_feature_grid(
    logmel: torch.Tensor,
    feat_hw: tuple[int, int],
    proto_hw: tuple[int, int],
    top_left_rc: tuple[int, int],
):
    """
    Map prototype top-left location on CNN latent feature grid
    back to an approximate region on the input log-mel spectrogram.
    """
    feat_h, feat_w = feat_hw
    proto_h, proto_w = proto_hw
    r, c = top_left_rc

    mel_bins, time_bins = logmel.shape

    row_edges = torch.linspace(0, mel_bins, feat_h + 1)
    col_edges = torch.linspace(0, time_bins, feat_w + 1)

    mel0 = int(row_edges[r].item())
    mel1 = int(row_edges[r + proto_h].item())
    t0_bin = int(col_edges[c].item())
    t1_bin = int(col_edges[c + proto_w].item())

    t0_sec = 10.0 * (t0_bin / time_bins)
    t1_sec = 10.0 * (t1_bin / time_bins)

    return mel0, mel1, t0_bin, t1_bin, t0_sec, t1_sec


def maybe_load_checkpoint(model, ckpt_path: str | None):
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint: {ckpt_path}\n"
        f"Missing keys: {len(missing)}\n"
        f"Unexpected keys: {len(unexpected)}"
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    DatasetCls, label_names = infer_dataset_class_from_path(DATASET_PATH)
    ds = DatasetCls(
        dataset_path=DATASET_PATH,
        split="train",
        sampling_rate=32000,
    )

    item = ds[SAMPLE_IDX]
    x = item["waveform"].unsqueeze(0)
    y = item["label"].unsqueeze(0)

    model = PrototypeClassifier(
        resnet_type="resnet18",
        conv_type="PANNS",
        audio_backbone_name="Cnn14",
        prototype_type="partial",
        n_prototypes=N_PROTOTYPES,
        n_binary_labels=y.shape[1],
        input_channels=1,
        prototype_h=PROTO_H,
        prototype_w=PROTO_W,
    )
    maybe_load_checkpoint(model, CKPT_PATH)
    model.eval()

    with torch.no_grad():
        losses, probs = model(x, y)

    encoder = model.encoder

    save_waveform_plot(x[0], os.path.join(OUT_DIR, "sample_waveform.png"), "Input waveform")
    logmel = encoder.resnet.logmel_for_viz(x)[0]
    save_spec_plot(logmel, os.path.join(OUT_DIR, "sample_logmel.png"), "Input log-mel spectrogram")

    feat_map = encoder.resnet.local_feature_map(x)  # (1, C, H_feat, W_feat)
    _, C, H_feat, W_feat = feat_map.shape

    local_sims = encoder.get_last_local_sims()[0]   # (L, P)
    grid_hw = encoder.get_last_local_grid_hw()      # (H_out, W_out)
    H_out, W_out = grid_hw
    proto_strength = local_sims.max(dim=0).values

    save_proto_strength_bar(
        proto_strength,
        os.path.join(OUT_DIR, "prototype_strengths.png"),
        "Prototype strengths on selected sample"
    )

    top_proto_idxs = torch.topk(proto_strength, k=min(TOPK_PROTOS, local_sims.shape[1])).indices.tolist()

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"Dataset class: {DatasetCls}\n")
        f.write(f"Num labels: {None if label_names is None else len(label_names)}\n")
        f.write(f"Sample idx: {SAMPLE_IDX}\n")
        f.write(f"Input waveform shape: {tuple(x.shape)}\n")
        f.write(f"Input logmel shape: {tuple(logmel.shape)}\n")
        f.write(f"Feature map shape (B,C,H,W): {tuple(feat_map.shape)}\n")
        f.write(f"Prototype patch size: {(PROTO_H, PROTO_W)}\n")
        f.write(f"Similarity grid (H_out, W_out): {(H_out, W_out)}\n")
        f.write(f"Top prototypes: {top_proto_idxs}\n")
        f.write(f"Positive labels in sample: {torch.where(y[0] > 0)[0].tolist()}\n")

    for rank, pidx in enumerate(top_proto_idxs):
        sim_map = local_sims[:, pidx].view(H_out, W_out)

        save_heatmap(
            sim_map,
            os.path.join(OUT_DIR, f"proto_rank{rank+1}_idx{pidx}_latent_similarity.png"),
            title=f"Prototype {pidx} latent similarity map",
            xlabel="Latent time position",
            ylabel="Latent frequency position",
        )

        sim_up = F.interpolate(
            sim_map[None, None],
            size=logmel.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        overlay_heatmap_on_spec(
            logmel,
            sim_up,
            os.path.join(OUT_DIR, f"proto_rank{rank+1}_idx{pidx}_overlay.png"),
            title=f"Prototype {pidx} explanation overlay",
        )

        flat_idx = int(local_sims[:, pidx].argmax().item())
        r = flat_idx // W_out
        c = flat_idx % W_out

        mel0, mel1, t0_bin, t1_bin, t0_sec, t1_sec = extract_patch_bounds_from_feature_grid(
            logmel=logmel,
            feat_hw=(H_feat, W_feat),
            proto_hw=(PROTO_H, PROTO_W),
            top_left_rc=(r, c),
        )

        overlay_box_on_spec(
            logmel,
            os.path.join(OUT_DIR, f"proto_rank{rank+1}_idx{pidx}_boxed_region.png"),
            title=f"Prototype {pidx} highlighted input region",
            t0=t0_sec,
            t1=t1_sec,
            f0=mel0,
            f1=mel1,
        )

        patch = logmel[mel0:mel1, t0_bin:t1_bin]
        if patch.numel() > 0:
            save_spec_plot(
                patch,
                os.path.join(OUT_DIR, f"proto_rank{rank+1}_idx{pidx}_cropped_patch.png"),
                title=f"Prototype {pidx} cropped input patch",
            )

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_score = {pidx: float("-inf") for pidx in top_proto_idxs}
    best_patch = {pidx: None for pidx in top_proto_idxs}
    best_meta = {pidx: None for pidx in top_proto_idxs}

    batches_seen = 0
    with torch.no_grad():
        for batch in loader:
            if batches_seen >= SEARCH_BATCHES:
                break
            batches_seen += 1

            xb = batch["waveform"]
            yb = batch["label"]

            _losses, _probs = model(xb, yb)
            feat_map_b = encoder.resnet.local_feature_map(xb)
            _, _, H_feat_b, W_feat_b = feat_map_b.shape
            local_sims_b = encoder.get_last_local_sims()  # (B, L, P)
            logmels_b = encoder.resnet.logmel_for_viz(xb)  # (B, mel, time)

            for b in range(xb.shape[0]):
                for pidx in top_proto_idxs:
                    sims_flat = local_sims_b[b, :, pidx]
                    score, flat_idx = sims_flat.max(dim=0)
                    score = float(score.item())
                    if score > best_score[pidx]:
                        r = int(flat_idx.item()) // W_out
                        c = int(flat_idx.item()) % W_out

                        mel0, mel1, t0_bin, t1_bin, t0_sec, t1_sec = extract_patch_bounds_from_feature_grid(
                            logmel=logmels_b[b],
                            feat_hw=(H_feat_b, W_feat_b),
                            proto_hw=(PROTO_H, PROTO_W),
                            top_left_rc=(r, c),
                        )
                        patch = logmels_b[b][mel0:mel1, t0_bin:t1_bin].cpu()
                        best_score[pidx] = score
                        best_patch[pidx] = patch
                        best_meta[pidx] = {
                            "batch_idx": batches_seen - 1,
                            "sample_in_batch": b,
                            "score": score,
                            "r": r,
                            "c": c,
                            "mel0": mel0,
                            "mel1": mel1,
                            "t0_sec": t0_sec,
                            "t1_sec": t1_sec,
                        }

    with open(os.path.join(OUT_DIR, "best_match_summary.txt"), "w") as f:
        for pidx in top_proto_idxs:
            f.write(f"Prototype {pidx}: {best_meta[pidx]}\n")

    for pidx in top_proto_idxs:
        patch = best_patch[pidx]
        if patch is not None and patch.numel() > 0:
            save_spec_plot(
                patch,
                os.path.join(OUT_DIR, f"proto_idx{pidx}_best_match_patch.png"),
                title=f"Prototype {pidx} best-match training patch | score={best_score[pidx]:.4f}",
            )

    print(f"Saved prototype explanation outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()