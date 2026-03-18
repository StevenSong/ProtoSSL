from torch.utils.data import DataLoader
from pass_pclr.datasets import infer_dataset_class_from_path

dataset_path = "/gpfs/data/bbj-lab/data/audioset/audioset"
DatasetCls, _ = infer_dataset_class_from_path(dataset_path)

ds_train = DatasetCls(
    dataset_path=dataset_path,
    split="train",
    sampling_rate=32000,
)

loader = DataLoader(
    ds_train,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

batch = next(iter(loader))
print("batch keys:", batch.keys())
print("batch waveform shape:", batch["waveform"].shape)
print("batch label shape:", batch["label"].shape)
print("batch patient_id shape:", batch["patient_id"].shape)
print("batch ecg_id shape:", batch["ecg_id"].shape)
print("per-sample label counts:", batch["label"].sum(dim=1))
