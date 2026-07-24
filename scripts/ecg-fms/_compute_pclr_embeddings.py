import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from protossl.datasets import infer_dataset_class_from_path
from protossl.models import BlackboxContraster


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    output_path: str,
    ckpt_path: str,
    batch_size: int = 512,
):
    # load ckpt for hyperparameters
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams: dict = ckpt["hyper_parameters"]  # type: ignore
    data_hparams: dict = ckpt["datamodule_hyper_parameters"]  # type: ignore

    # load and freeze model
    model = BlackboxContraster(
        contrastive_pair_mode=hparams["contrastive_pair_mode"],
        backbone_type=hparams["backbone_type"],
        conv_type=hparams["conv_type"],
        **hparams["model_kwargs"],
        pretrained_weights=ckpt_path,  # internally loads checkpoint again, but ok as model is small
    )
    for param in model.parameters():
        param.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = model.encoder
    encoder = encoder.eval().to(device)

    ds_cls, _, is_audio = infer_dataset_class_from_path(dataset_path)
    assert not is_audio
    train_ds = ds_cls(
        dataset_path=dataset_path,
        split="train",
        sampling_rate=data_hparams["sampling_rate"],
    )
    test_ds = ds_cls(
        dataset_path=dataset_path,
        split="test",
        sampling_rate=data_hparams["sampling_rate"],
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    train_embs, test_embs = [], []
    with torch.inference_mode():
        for dl, embs in [(train_dl, train_embs), (test_dl, test_embs)]:
            for batch in tqdm(dl):
                x = batch["waveform"].to(device)
                batch_embs = encoder(x).cpu()
                embs.append(batch_embs.numpy())
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
        ckpt_path=args.ckpt,
        batch_size=args.batch_size,
    )
