import torch
from torch.utils.data import DataLoader

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.models import ResNetClassifier

dataset_path = "/gpfs/data/bbj-lab/data/audioset/audioset"
DatasetCls, label_names = infer_dataset_class_from_path(dataset_path)

ds_train = DatasetCls(
    dataset_path=dataset_path,
    split="train",
    sampling_rate=16000,
)

loader = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=0)
batch = next(iter(loader))

model = ResNetClassifier(
    resnet_type="resnet18",
    conv_type="1D",
    n_binary_labels=batch["label"].shape[1],
    input_channels=1,
)

x = batch["waveform"]
y = batch["label"]

losses, probs = model(x, y)

print("input shape:", x.shape)
print("label shape:", y.shape)
print("losses shape:", losses.shape)
print("probs shape:", probs.shape)
print("contains nan in losses:", torch.isnan(losses).any().item())
print("contains nan in probs:", torch.isnan(probs).any().item())
