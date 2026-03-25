import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from tqdm import tqdm

from pass_pclr.datasets import infer_dataset_class_from_path

CHECKPOINT_URL = "https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecgfounder-repo", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    ecgfounder_repo: str,
    dataset_path: str,
    output_path: str,
    batch_size: int = 512,
):
    sys.path.append(ecgfounder_repo)
    from net1d import Net1D  # type: ignore

    # init parameters taken from ECGFounder/ptbxl_eval.py
    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
    )

    # load pretrained weights
    sd = load_state_dict_from_url(
        url=CHECKPOINT_URL,
        model_dir=os.path.join(
            ecgfounder_repo, "checkpoint"
        ),  # local save to prevent redownloading on subsequent run
        map_location="cpu",
        weights_only=False,  # NOTE: this incurs risk of unpickling something malicious
    )["state_dict"]

    model.load_state_dict(sd)

    # remove classification head so we just get the embeddings
    model.dense = nn.Identity()

    # freeze model
    for param in model.parameters():
        param.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)

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
                batch_embs = model(x).detach().to("cpu").numpy()
                embs.append(batch_embs)
    train_embs = np.concatenate(train_embs)
    test_embs = np.concatenate(test_embs)

    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "train_embeds.npy"), train_embs)
    np.save(os.path.join(output_path, "test_embeds.npy"), test_embs)


if __name__ == "__main__":
    args = parse_args()
    main(
        ecgfounder_repo=args.ecgfounder_repo,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
