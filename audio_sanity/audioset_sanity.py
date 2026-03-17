from pass_pclr.datasets import infer_dataset_class_from_path

dataset_path = "/gpfs/data/bbj-lab/data/audioset/audioset"
DatasetCls, label_names = infer_dataset_class_from_path(dataset_path)

print("DatasetCls:", DatasetCls)
print("n_label_names:", None if label_names is None else len(label_names))

ds_train = DatasetCls(
    dataset_path=dataset_path,
    split="train",
    sampling_rate=16000,
)

ds_val = DatasetCls(
    dataset_path=dataset_path,
    split="val",
    sampling_rate=16000,
)

print("len(train):", len(ds_train))
print("len(val):", len(ds_val))

item = ds_train[0]
print("keys:", item.keys())
print("waveform shape:", item["waveform"].shape)
print("waveform dtype:", item["waveform"].dtype)
print("label shape:", item["label"].shape)
print("label dtype:", item["label"].dtype)
print("n positive labels in sample 0:", item["label"].sum().item())
print("patient_id:", item["patient_id"])
print("ecg_id:", item["ecg_id"])
