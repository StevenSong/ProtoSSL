import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from protossl.datasets import infer_dataset_class_from_path
from protossl.models.encoders import Net1D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    # 12_lead_ECGFounder.pth from https://huggingface.co/PKUDigitalHealth/ECGFounder
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--return-patches", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "test"],
    )
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    output_path: str,
    ckpt_path: str,
    batch_size: int = 512,
    return_patches: bool = False,
    splits: tuple[str, ...] = ("train", "test"),
):
    # init parameters taken from ECGFounder/ptbxl_eval.py
    encoder = Net1D(return_patches=return_patches)

    # load pretrained weights
    sd = torch.load(
        ckpt_path,
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
    # which we do in the protossl dataset implementations
    ds_cls, _, is_audio = infer_dataset_class_from_path(dataset_path)
    assert not is_audio
    os.makedirs(output_path, exist_ok=True)
    with torch.inference_mode():
        for split in splits:
            ds = ds_cls(
                dataset_path=dataset_path,
                split=split,
                sampling_rate=500,  # ECGFounder expects 500 Hz inputs
            )
            embs = []
            for batch in tqdm(DataLoader(ds, batch_size=batch_size)):
                x = batch["waveform"].to(device)
                embs.append(encoder(x).detach().to("cpu").numpy())
            np.save(
                os.path.join(output_path, f"{split}_embeds.npy"), np.concatenate(embs)
            )


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        ckpt_path=args.ckpt,
        batch_size=args.batch_size,
        return_patches=args.return_patches,
        splits=args.splits,
    )
