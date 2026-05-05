import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from protossl.datasets import infer_dataset_class_from_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--stmem-repo", required=True)
    parser.add_argument("--stmem-ckpt", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    output_path: str,
    stmem_repo: str,
    stmem_ckpt: str,
    batch_size: int = 512,
):
    sys.path.append(stmem_repo)
    from models.encoder.st_mem_vit import st_mem_vit_base  # type: ignore

    # init parameters taken from ST-MEM/configs/st_mem.yaml
    encoder = st_mem_vit_base(
        seq_len=2250,
        patch_size=75,
        num_leads=12,
        num_classes=1,  # we'll remove this
    )

    # load pretrained weights
    sd = torch.load(stmem_ckpt, map_location="cpu")["model"]

    # get rid of task head
    encoder.head = nn.Identity()
    encoder.load_state_dict(sd, strict=True)

    # freeze model
    for param in encoder.parameters():
        param.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.eval().to(device)

    # NOTE: ST-MEM does high/low pass filters, we skip those
    # but we do resample to 250 Hz and normalize (and crop later)
    ds_cls, _, is_audio = infer_dataset_class_from_path(dataset_path)
    assert not is_audio
    train_ds = ds_cls(
        dataset_path=dataset_path,
        split="train",
        sampling_rate=250,  # ST-MEM expects 250 Hz inputs
    )
    test_ds = ds_cls(
        dataset_path=dataset_path,
        split="test",
        sampling_rate=250,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    train_embs, test_embs = [], []
    with torch.inference_mode():
        for dl, embs in [(train_dl, train_embs), (test_dl, test_embs)]:
            for batch in tqdm(dl):
                full_x = batch["waveform"].to(device)
                emb_crops = []
                # 3 evenly spaced crops of length 2250, following ST-MEM repo
                for start_idx in [0, 125, 250]:
                    x = full_x[:, :, start_idx : start_idx + 2250]
                    emb_crops.append(encoder(x).cpu())
                batch_embs = torch.stack(emb_crops).mean(dim=0)
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
        stmem_repo=args.stmem_repo,
        stmem_ckpt=args.stmem_ckpt,
        batch_size=args.batch_size,
    )
