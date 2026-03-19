import torch
from torch.utils.data import DataLoader

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.models import PrototypeClassifier


DATASET_PATH = "/gpfs/data/bbj-lab/data/audioset/audioset"


def main():
    DatasetCls, label_names = infer_dataset_class_from_path(DATASET_PATH)
    print("Dataset class:", DatasetCls)
    print("Num labels:", None if label_names is None else len(label_names))

    ds = DatasetCls(
        dataset_path=DATASET_PATH,
        split="train",
        sampling_rate=32000,
    )
    print("Dataset length:", len(ds))

    item = ds[0]
    print("Single sample keys:", item.keys())
    print("Single waveform shape:", item["waveform"].shape)
    print("Single label shape:", item["label"].shape)
    print("Single positive labels:", int(item["label"].sum().item()))

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    x = batch["waveform"]
    y = batch["label"]

    print("Batch waveform shape:", x.shape)
    print("Batch label shape:", y.shape)
    print("Batch positive labels per sample:", y.sum(dim=1))

    model = PrototypeClassifier(
        resnet_type="resnet18",   # ignored by PANNS path, but required by signature
        conv_type="PANNS",
        audio_backbone_name="Wavegram_Logmel_Cnn14", #Cnn14 #ResNet38
        prototype_type="partial",
        n_prototypes=32,
        n_binary_labels=y.shape[1],
        input_channels=1,
        prototype_h=1,
        prototype_w=1,
    )
    model.eval()

    with torch.no_grad():
        losses, probs = model(x, y)

    print("\n=== Classifier forward ===")
    print("Losses shape:", losses.shape)
    print("Probs shape:", probs.shape)
    print("NaN losses:", torch.isnan(losses).any().item())
    print("NaN probs:", torch.isnan(probs).any().item())

    encoder = model.encoder
    print("\n=== Encoder internals ===")

    # 1) raw PANNS latent map
    feat_map = encoder.resnet.local_feature_map(x)
    print("Cnn14 latent feature map shape (B,C,H,W):", feat_map.shape)

    # 2) prototype patch settings
    prototype_h, prototype_w = encoder.get_prototype_patch_hw()
    print("Prototype patch size (h,w):", (prototype_h, prototype_w))

    # 3) local patch vectors and argmax locations stored for projection
    embs, loc_idxs = encoder.get_last_embs_and_chunks()
    print("Stored local patch embeddings shape (B,L,D):", embs.shape)
    print("Stored best location indices shape (B,P):", loc_idxs.shape)

    # 4) local similarity tensor and spatial grid
    local_sims = encoder.get_last_local_sims()
    grid_hw = encoder.get_last_local_grid_hw()
    print("Stored local similarities shape (B,L,P):", local_sims.shape)
    print("Stored valid-location grid (H_out,W_out):", grid_hw)

    B, C, H, W = feat_map.shape
    B2, L, D = embs.shape
    B3, L2, P = local_sims.shape
    assert B == B2 == B3, "Batch dimension mismatch"

    expected_D = C * prototype_h * prototype_w
    assert D == expected_D, (
        f"Patch embedding dim mismatch: got {D}, expected {expected_D}"
    )

    H_out = H - prototype_h + 1
    W_out = W - prototype_w + 1
    assert grid_hw == (H_out, W_out), (
        f"Grid mismatch: got {grid_hw}, expected {(H_out, W_out)}"
    )
    assert L == H_out * W_out, (
        f"Num valid locations mismatch: got {L}, expected {H_out * W_out}"
    )
    assert L2 == L, "local_sims location count mismatch"
    assert P == 32, f"Expected 32 prototypes, got {P}"

    max_valid_idx = H_out * W_out - 1
    assert torch.all(loc_idxs >= 0).item(), "Negative best-location indices found"
    assert torch.all(loc_idxs <= max_valid_idx).item(), (
        f"Best-location indices exceed valid range 0..{max_valid_idx}"
    )

    prototypes = encoder.get_prototypes()
    print("Prototype tensor shape (P,D):", prototypes.shape)
    assert prototypes.shape == (32, expected_D), (
        f"Prototype shape mismatch: got {prototypes.shape}, expected {(32, expected_D)}"
    )

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()