import torch
from torch.utils.data import DataLoader

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.models import PrototypeClassifier


def main():
    dataset_path = "/gpfs/data/bbj-lab/data/audioset/audioset"

    DatasetCls, label_names = infer_dataset_class_from_path(dataset_path)
    print("Dataset class:", DatasetCls)
    print("Num label names:", None if label_names is None else len(label_names))

    ds_train = DatasetCls(
        dataset_path=dataset_path,
        split="train",
        sampling_rate=32000,
    )
    print("Dataset length:", len(ds_train))

    loader = DataLoader(
        ds_train,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))
    x = batch["waveform"]
    y = batch["label"]

    print("Input waveform shape:", x.shape)
    print("Label shape:", y.shape)
    print("Per-sample positive label counts:", y.sum(dim=1))

    model = PrototypeClassifier(
        resnet_type="resnet18",
        conv_type="1D",
        prototype_type="partial",
        n_prototypes=32,
        n_binary_labels=y.shape[1],
        input_channels=1,
        partial_len=32000,      # 1 second at 16 kHz
        partial_overlap=0.5,    # 50% overlap
    )

    model.eval()
    with torch.no_grad():
        losses, probs = model(x, y)

    print("Losses shape:", losses.shape)
    print("Probs shape:", probs.shape)
    print("Contains NaN in losses:", torch.isnan(losses).any().item())
    print("Contains NaN in probs:", torch.isnan(probs).any().item())
    print("Mean loss:", losses.mean().item())
    print("Min prob:", probs.min().item())
    print("Max prob:", probs.max().item())


if __name__ == "__main__":
    main()
