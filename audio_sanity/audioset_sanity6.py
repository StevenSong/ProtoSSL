from pass_pclr.datasets import infer_dataset_class_from_path

dataset_path = "/gpfs/data/bbj-lab/data/audioset/audioset"
DatasetCls, label_names = infer_dataset_class_from_path(dataset_path)

print("n_label_names:", len(label_names))

ds_train = DatasetCls(dataset_path=dataset_path, split="train", sampling_rate=32000)
ds_val = DatasetCls(dataset_path=dataset_path, split="val", sampling_rate=32000)
ds_test = DatasetCls(dataset_path=dataset_path, split="test", sampling_rate=32000)

print("len(train):", len(ds_train))
print("len(val):", len(ds_val))
print("len(test):", len(ds_test))
print("train label shape:", ds_train[0]["label"].shape)
print("val label shape:", ds_val[0]["label"].shape)
print("test label shape:", ds_test[0]["label"].shape)
