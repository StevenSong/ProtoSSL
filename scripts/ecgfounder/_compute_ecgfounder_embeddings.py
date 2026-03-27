import argparse
import os

import numpy as np
import torch
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from tqdm import tqdm

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.models.encoders import Net1D

CHECKPOINT_URL = "https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    output_path: str,
    batch_size: int = 512,
):
    # init parameters taken from ECGFounder/ptbxl_eval.py
    encoder = Net1D()

    # load pretrained weights
    sd = load_state_dict_from_url(
        url=CHECKPOINT_URL,
        model_dir="ecgfounder-checkpoint",  # local save to reuse
        map_location="cpu",
        weights_only=False,  # NOTE: this incurs risk of unpickling something malicious
    )["state_dict"]

    # not using classification head
    del sd["dense.weight"]
    del sd["dense.bias"]

    encoder.load_state_dict(sd)

    # freeze model
    for param in encoder.parameters():
        param.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.eval().to(device)

    # based on dataset implementation in ECGFounder/ptbxl_eval.py
    # the only transformation is normalization and resampling
    # which we do in the pass_pclr dataset implementations
    ds_cls, _ = infer_dataset_class_from_path(dataset_path)
    train_ds = ds_cls(
        dataset_path=dataset_path,
        split="train",
        sampling_rate=500,  # ECGFounder expects 500 Hz inputs
    )
    test_ds = ds_cls(
        dataset_path=dataset_path,
        split="test",
        sampling_rate=500,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    train_embs, test_embs = [], []
    with torch.inference_mode():
        for dl, embs in [(train_dl, train_embs), (test_dl, test_embs)]:
            for batch in tqdm(dl):
                x = batch["waveform"].to(device)
                batch_embs = encoder(x).detach().to("cpu").numpy()
                embs.append(batch_embs)
    train_embs = np.concatenate(train_embs)
    test_embs = np.concatenate(test_embs)

    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "train_embeds.npy"), train_embs)
    np.save(os.path.join(output_path, "test_embeds.npy"), test_embs)


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
