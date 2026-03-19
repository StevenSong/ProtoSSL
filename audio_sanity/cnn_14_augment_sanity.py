import os

import matplotlib.pyplot as plt
import torch

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.models.encoders import PANNSEncoder


DATASET_PATH = "/gpfs/data/bbj-lab/data/audioset/audioset"
OUT_DIR = "/gpfs/data/bbj-lab/users/sethis/ecg-prototype-fm/audio_sanity/cnn14_augment_sanity_out"
SAMPLE_IDX = 0


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


def save_waveform_diff_plot(x1: torch.Tensor, x2: torch.Tensor, out_path: str, title: str, sample_rate: int = 32000):
    d = (x1 - x2).squeeze().cpu().numpy()
    t = [i / sample_rate for i in range(len(d))]
    plt.figure(figsize=(12, 3))
    plt.plot(t, d, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude difference")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_spec_diff_plot(s1: torch.Tensor, s2: torch.Tensor, out_path: str, title: str):
    d = (s1 - s2).cpu().numpy()
    plt.figure(figsize=(12, 4))
    im = plt.imshow(
        d,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=[0, 10.0, 0, d.shape[0]],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Mel bin")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    DatasetCls, label_names = infer_dataset_class_from_path(DATASET_PATH)
    ds_train = DatasetCls(dataset_path=DATASET_PATH, split="train", sampling_rate=32000)
    ds_val = DatasetCls(dataset_path=DATASET_PATH, split="val", sampling_rate=32000)

    train_a = ds_train[SAMPLE_IDX]
    train_b = ds_train[SAMPLE_IDX]
    val_a = ds_val[SAMPLE_IDX]
    val_b = ds_val[SAMPLE_IDX]

    x_train_a = train_a["waveform"]
    x_train_b = train_b["waveform"]
    x_val_a = val_a["waveform"]
    x_val_b = val_b["waveform"]

    encoder = PANNSEncoder(audio_backbone_name="Cnn14")
    encoder.eval()

    with torch.no_grad():
        spec_train_a = encoder.logmel_for_viz(x_train_a.unsqueeze(0))[0]
        spec_train_b = encoder.logmel_for_viz(x_train_b.unsqueeze(0))[0]
        spec_val_a = encoder.logmel_for_viz(x_val_a.unsqueeze(0))[0]
        spec_val_b = encoder.logmel_for_viz(x_val_b.unsqueeze(0))[0]

    train_equal = torch.allclose(x_train_a, x_train_b)
    val_equal = torch.allclose(x_val_a, x_val_b)

    train_l1 = torch.mean(torch.abs(x_train_a - x_train_b)).item()
    val_l1 = torch.mean(torch.abs(x_val_a - x_val_b)).item()

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"Dataset class: {DatasetCls}\n")
        f.write(f"Num labels: {None if label_names is None else len(label_names)}\n")
        f.write(f"Sample idx: {SAMPLE_IDX}\n")
        f.write(f"train identical: {train_equal}\n")
        f.write(f"val identical: {val_equal}\n")
        f.write(f"train mean abs diff: {train_l1:.8f}\n")
        f.write(f"val mean abs diff: {val_l1:.8f}\n")
        f.write(f"train positives: {int(train_a['label'].sum().item())}\n")
        f.write(f"val positives: {int(val_a['label'].sum().item())}\n")

    save_waveform_plot(x_train_a, os.path.join(OUT_DIR, "train_waveform_a.png"), "Train sample A waveform")
    save_waveform_plot(x_train_b, os.path.join(OUT_DIR, "train_waveform_b.png"), "Train sample B waveform")
    save_waveform_diff_plot(
        x_train_a, x_train_b,
        os.path.join(OUT_DIR, "train_waveform_diff.png"),
        "Train waveform difference (same index, two calls)"
    )

    save_waveform_plot(x_val_a, os.path.join(OUT_DIR, "val_waveform_a.png"), "Val sample A waveform")
    save_waveform_plot(x_val_b, os.path.join(OUT_DIR, "val_waveform_b.png"), "Val sample B waveform")
    save_waveform_diff_plot(
        x_val_a, x_val_b,
        os.path.join(OUT_DIR, "val_waveform_diff.png"),
        "Val waveform difference (same index, two calls)"
    )

    save_spec_plot(spec_train_a, os.path.join(OUT_DIR, "train_logmel_a.png"), "Train sample A log-mel")
    save_spec_plot(spec_train_b, os.path.join(OUT_DIR, "train_logmel_b.png"), "Train sample B log-mel")
    save_spec_diff_plot(
        spec_train_a, spec_train_b,
        os.path.join(OUT_DIR, "train_logmel_diff.png"),
        "Train log-mel difference (same index, two calls)"
    )

    save_spec_plot(spec_val_a, os.path.join(OUT_DIR, "val_logmel_a.png"), "Val sample A log-mel")
    save_spec_plot(spec_val_b, os.path.join(OUT_DIR, "val_logmel_b.png"), "Val sample B log-mel")
    save_spec_diff_plot(
        spec_val_a, spec_val_b,
        os.path.join(OUT_DIR, "val_logmel_diff.png"),
        "Val log-mel difference (same index, two calls)"
    )

    print(f"Saved augmentation sanity outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()