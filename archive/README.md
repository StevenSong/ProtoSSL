# Archive

Code kept for the record that is no longer maintained or run.

## `audio/`

The paper's audio experiments (AudioSet pretraining; UrbanSound8K, IEMOCAP and VoxCeleb1 downstream), moved here from their old places in the repo:

| Path | Was |
|---|---|
| `audio/scripts/` | `scripts/audio/` |
| `audio/configs/` | `configs/audio/` |
| `audio/results/audio-results.ipynb` | `results/audio-results.ipynb` |
| `audio/data-preprocessing/hf_audioset_ids/` | `data-preprocessing/hf_audioset_ids/` |

The scripts predate the job runner in `slurm/`. They hardcode our old paths (`/opt/gpu_working/steven/...`), including the old locations of these configs, so they don't run as they are. The audio datasets and models are still in the `protossl` package.

They also need ffmpeg, which the uv env doesn't have, so they ran in a conda env. Its `env.yaml` and `requirements.txt` were removed from the repo when it moved to uv only; recreate it from the last `main` commit that has them:

```bash
git show c96c1b3:env.yaml > env.yaml
git show c96c1b3:requirements.txt > requirements.txt
conda env create -f env.yaml
conda activate protossl
pip install -e .
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
