# ProtoSSL: Self-Supervised Pretraining and Downstream Transfer for Projection-Based Prototype Models

This repo contains the source code for **ProtoSSL**, a novel framework for pretraining a foundational latent prototype bank on unlabeled data and transferring it to downstream tasks to create interpretable, projection-based prototype models. Our key innovation is to separate *motif discovery* from *label alignment*. ProtoSSL first learns a reusable latent prototype bank using a self-supervised objective applied directly to prototype activations, and then aligns these prototypes to downstream tasks through a novel and efficient assignment procedure. We study ProtoSSL and provide code for our experiments primarily over ECGs and additionally audio waveforms.

### Citation
```
@article{song2026protossl,
  title={ProtoSSL: Self-Supervised Pretraining and Downstream Transfer for Projection-Based Prototype Models},
  author={Song, Steven and Sethi, Sahil and Beaulieu-Jones, Brett and Grossman, Robert L},
  journal={arXiv preprint arXiv:2605.06943},
  year={2026}
}
```

## Repo Organization

| Path | Contents |
|------|----------|
| `protossl/` | The `protossl` package: datasets, models, and the Lightning trainers (`protossl.trainer` for ProtoSSL and baselines, `protossl.protoecgnet_trainer` for multi-arm ProtoECGNet) |
| `configs/` | Lightning CLI configs for pretraining (`pretrain-*.yaml`) and downstream tasks (`target-*.yaml`), with audio configs under `configs/audio/` |
| `scripts/` | Run scripts for every experiment in the paper, plus the slurm queue scripts that submit them. See [`scripts/README.md`](scripts/README.md) |
| `user-study/` | Sample preparation and analysis for the user study (`user-study/images` is a submodule of the de-identified study images) |
| `results/` | Notebooks that produce the figures and tables in the paper from the experiment outputs |
| `data-preprocessing/` | Notebooks that create the train/val/test splits and nested training subsets, plus precomputed HEEDB normalization statistics. See [`data-preprocessing/README.md`](data-preprocessing/README.md) |
| `external/` | Submodules for baselines: [ST-MEM](https://github.com/vuno/ST-MEM) and our fork of [IntroECG](https://github.com/PierreElias/IntroECG) (EchoNext minimodel) |

## Setup

```bash
# 1) clone repo, including submodules (only needed for the ST-MEM and EchoNext minimodel baselines, and the user study images)
git clone --recurse-submodules git@github.com:StevenSong/ProtoSSL.git
cd ProtoSSL

# 2) create and activate environment
# NOTE: if you don't use conda, make sure you're using the same python version, install from `requirements.txt`, and MAKE SURE YOU HAVE FFMPEG 5.* FOR TORCHCODEC (see below)
conda env create -f env.yaml
conda activate protossl

# 3) install protossl from editable source
pip install -e .

# 4) (contributors only) enable pre-commit hooks, see Contributing below
pre-commit install
```

<details>
<summary><b>Troubleshooting torchcodec</b></summary>

torchcodec is a bit fragile with dependencies. We've pinned `torch==2.7.0` which is compatible with `torchcodec==0.4.0`, both compiled against CUDA 12.8 (which we use on our machines). This torchcodec version is only compatible with `datasets==4.0.0`. If you see errors relating to torchcodec (you can diagnose this by just importing torchcodec), make sure the dependencies are compatible not just relative to versioning, but also relating to the CUDA versions. We also use `ffmpeg=5.*` installed via conda. If you see an error relating to not being able to find `libnppicc.so.12`, it might be that the linker can't find the binaries (which we ensure are available by installing `nvidia-npp-cu12`). To fix this, you can try setting the `LD_LIBRARY_PATH` environment variable:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH
# test by importing torchcodec in a python runtime
```
If this works, you can consider making the fix automatic via the following conda activate scripts:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# Set on activate
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/npp_lib.sh

# Unset on deactivate
echo 'export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s|$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib:||g")' > $CONDA_PREFIX/etc/conda/deactivate.d/npp_lib.sh
```
</details>

## Data

We do not redistribute any data; each dataset must be obtained from its original source under its own terms of use.

* **ECG pretraining:** HEEDB
* **ECG downstream:** EchoNext, PTB-XL, CinC 2020 (Georgia), MIMIC-IV-ECG, ZZU pECG, CODE-15%
* **Audio pretraining:** AudioSet
* **Audio downstream:** UrbanSound8K, IEMOCAP, VoxCeleb1

The `data-preprocessing/` notebooks create the splits and nested training subsets (e.g. `ptb-xl-1k`) used for our data-efficiency experiments.

## Reproducing Our Experiments

1. Obtain the datasets above and create the splits and training subsets with the `data-preprocessing/` notebooks
1. Find and replace the hardcoded paths in the run, queue, and pretrain scripts: `/opt/gpu_working/steven` (repo and outputs) and `/opt/gpudata` (source datasets)
1. Pretrain: run the scripts in [`scripts/pretrain/`](scripts/pretrain/) for ECG and `scripts/audio/run-pretrain-audio-*.sh` for audio
1. Submit the downstream experiments via slurm: [`scripts/queue-experiments.sh`](scripts/queue-experiments.sh) (ECG) and [`scripts/audio/queue-experiments.sh`](scripts/audio/queue-experiments.sh) (audio)
1. Produce figures and tables with the `results/` notebooks

See [`scripts/README.md`](scripts/README.md) for more detail. Each run script writes to `$RUN_DIR/<experiment-name>`, named `<model>-<pretraining data>-<transfer method>[-suffixes]`:

* **Models:** `protossl`, `supproto` (supervised prototypes), `blackbox`, and linear probes over foundation model embeddings (`ecgfounder-logreg`, `stmem-logreg`, `pclr-logreg`)
* **Transfer methods:** `pila` (project in linear assignment), `rila` (reproject in linear assignment), `pia` (project in assignment, using ProtoPool-based assignment), `pit` (project in transfer dataset without assignment or label supervision), `pip` (project in pretraining dataset without assignment or label supervision), `latent` (no projection), `direct` (trained only on the target dataset)
* **Suffixes:** `-ft` is fine-tuned, `-Nppl` uses N prototypes per label, `heedb-150` pretrains over the 150-label HEEDB set

### Environment Variables

| Variable | Effect |
|----------|--------|
| `CACHE_DIR` | Where preprocessed datasets are cached (default `~/.cache/protossl_cache`) |
| `INVALIDATE_CACHE` | Set to `"yes im sure"` to delete the cache on import |
| `USE_HEEDB_150` | Use the 150-label HEEDB label set instead of the default (set by the `*-heedb-150-*` pretraining scripts) |
| `HIGH_MEMORY` | Load all of HEEDB into memory (~500 GB of RAM) rather than streaming it from disk (set by the HEEDB pretraining scripts) |
| `US8K_TEST_FOLD`, `IEMOCAP_TEST_FOLD` | Held-out fold for UrbanSound8K and IEMOCAP cross-validation (default `0`) |

The ECGFounder baseline downloads its checkpoint from Hugging Face to `ecgfounder-checkpoint/` under the working directory (`scripts/` for the run scripts) on first use.

## Contributing

We use pre-commit hooks to maintain coding style and to enforce data stewardship. Source data should never be committed to this repo, including exploratory notebooks which may accidentally leak source data. The pre-commit hooks (which strip notebook outputs, scan for secrets, and block direct commits to `main`) do not necessarily prevent this and can be circumvented, but may help prevent obvious cases. Every tracked file must also be covered by `.github/CODEOWNERS`.

## License

Apache 2.0, see [`LICENSE`](LICENSE).
