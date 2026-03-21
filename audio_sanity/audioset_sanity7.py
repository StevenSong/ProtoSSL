import torch
from torch.utils.data import DataLoader

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.models import PrototypeClassifier

dataset_path = "/gpfs/data/bbj-lab/data/audioset/audioset"
DatasetCls, label_names = infer_dataset_class_from_path(dataset_path)

ds = DatasetCls(
    dataset_path=dataset_path,
    split="train",
    sampling_rate=32000,
)

loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
batch = next(iter(loader))

model = PrototypeClassifier(
    resnet_type="resnet18",   # ignored by HTSAT path
    conv_type="HTSAT",
    prototype_type="partial",
    n_prototypes=32,
    n_binary_labels=batch["label"].shape[1],
    input_channels=1,
    prototype_h=1,
    prototype_w=1,
)

x = batch["waveform"]
y = batch["label"]

losses, probs = model(x, y)
print("x:", x.shape)
print("losses:", losses.shape)
print("probs:", probs.shape)
print("NaN losses:", torch.isnan(losses).any().item())
print("NaN probs:", torch.isnan(probs).any().item())