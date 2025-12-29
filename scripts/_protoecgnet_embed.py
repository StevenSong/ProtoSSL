import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from tqdm import tqdm, trange


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--echonext-data", required=True)
    parser.add_argument("--protoecgnet-repo", required=True)
    parser.add_argument("--protoecgnet-checkpoints", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    echonext_data: str,
    protoecgnet_repo: str,
    protoecgnet_checkpoints: str,
    output_path: str,
):
    sys.path.append(os.path.join(protoecgnet_repo, "src"))
    from proto_models1D import ProtoECGNet1D  # type: ignore isort: skip
    from proto_models2D import ProtoECGNet2D  # type: ignore isort: skip

    repo_path = Path(protoecgnet_repo)
    label_df = pd.read_csv(repo_path / "scp_statementsRegrouped2.csv", index_col=0)
    assert "prototype_category" in label_df.columns

    counts = label_df["prototype_category"].value_counts()
    num_classes1 = counts.loc[1]
    num_classes3 = counts.loc[3]
    num_classes4 = counts.loc[4]

    checkpoint_path = Path(protoecgnet_checkpoints)
    weights_1d = checkpoint_path / "cat1.ckpt"
    weights_2d_partial = checkpoint_path / "cat3.ckpt"
    weights_2d_global = checkpoint_path / "cat4.ckpt"

    echonext_path = Path(echonext_data)
    X_train = np.load(echonext_path / "EchoNext_train_waveforms.npy")
    X_val = np.load(echonext_path / "EchoNext_val_waveforms.npy")
    X_test = np.load(echonext_path / "EchoNext_test_waveforms.npy")

    _output_path = Path(output_path)
    os.makedirs(_output_path, exist_ok=False)

    model_1d = ProtoECGNet1D(
        num_classes=num_classes1,
        single_class_prototype_per_class=5,
        joint_prototypes_per_border=0,
        proto_dim=512,
        backbone="resnet1d18",
        prototype_activation_function="log",
        latent_space_type="arc",
        add_on_layers_type="linear",
        class_specific=True,
        last_layer_connection_weight=1.0,
        m=0.05,
        dropout=0,
        custom_groups=True,
        label_set="1",
        pretrained_weights=weights_1d,
    ).to("cuda")

    model_2d_partial = ProtoECGNet2D(
        num_classes=num_classes3,
        single_class_prototype_per_class=18,
        joint_prototypes_per_border=0,
        proto_dim=512,
        backbone="resnet18",
        prototype_activation_function="log",
        latent_space_type="arc",
        add_on_layers_type="linear",
        class_specific=True,
        last_layer_connection_weight=1.0,
        m=0.05,
        dropout=0,
        custom_groups=True,
        label_set="3",
        pretrained_weights=weights_2d_partial,
        proto_time_len=3,
    ).to("cuda")

    model_2d_global = ProtoECGNet2D(
        num_classes=num_classes4,
        single_class_prototype_per_class=7,
        joint_prototypes_per_border=0,
        proto_dim=512,
        backbone="resnet18",
        prototype_activation_function="log",
        latent_space_type="arc",
        add_on_layers_type="linear",
        class_specific=True,
        last_layer_connection_weight=1.0,
        m=0.05,
        dropout=0,
        custom_groups=True,
        label_set="4",
        pretrained_weights=weights_2d_global,
        proto_time_len=32,
    ).to("cuda")

    for X, split in tqdm(
        [
            (X_train, "train"),
            (X_val, "val"),
            (X_test, "test"),
        ],
        desc="Split",
    ):
        sim1ds = []
        sim2d_partials = []
        sim2d_globals = []
        n_split = len(X)
        for i in trange(n_split):
            x = X[i : i + 1]  # (1, 1, 2500, 12)

            # downsample to 100 Hz
            x = resample_poly(x, up=2, down=5, axis=2)  # (1, 1, 1000, 12)
            assert x.shape == (1, 1, 1000, 12)

            # cast to tensor, move to GPU, cast to float32
            x = torch.as_tensor(x).to("cuda").to(torch.float32)  # (1, 1, 1000, 12)

            # rearrange lead/time axes
            x = x.mT  # (1, 1, 12, 1000)

            # remove channel dim for 1D branch
            x1d = x.squeeze(1)  # (1, 12, 1000)

            with torch.inference_mode():
                _, _, sim1d = model_1d(x1d)  # (1, N_1d_prototypes)
                _, _, sim2d_partial = model_2d_partial(x)
                _, _, sim2d_global = model_2d_global(x)

            sim1d = sim1d.cpu().numpy().squeeze()
            sim2d_partial = sim2d_partial.cpu().numpy().squeeze()
            sim2d_global = sim2d_global.cpu().numpy().squeeze()

            sim1ds.append(sim1d)
            sim2d_partials.append(sim2d_partial)
            sim2d_globals.append(sim2d_global)

        sim1ds = np.stack(sim1ds)
        sim2d_partials = np.stack(sim2d_partials)
        sim2d_globals = np.stack(sim2d_globals)

        np.save(_output_path / f"{split}_sim1ds.npy", sim1ds)
        np.save(_output_path / f"{split}_sim2d_partials.npy", sim2d_partials)
        np.save(_output_path / f"{split}_sim2d_globals.npy", sim2d_globals)


if __name__ == "__main__":
    args = parse_args()
    main(
        echonext_data=args.echonext_data,
        protoecgnet_repo=args.protoecgnet_repo,
        protoecgnet_checkpoints=args.protoecgnet_checkpoints,
        output_path=args.output_path,
    )
