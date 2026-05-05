# ProtoSSL: Interpretable Prototype Learning from Unlabeled Time-Series Data

This repo contains the source code for **ProtoSSL**, a novel framework for label-free learning of interpretable, projection-based prototypes that are readily adaptable to downstream tasks. Our key innovation is to separate *motif discovery* from *label alignment*. ProtoSSL first learns a reusable prototype bank using a self-supervised objective applied directly to prototype activations, and then aligns these prototypes to downstream tasks through a novel and efficient assignment procedure. We study ProtoSSL and provide code for our experiments primarily over ECGs and additionally audio waveforms.

### Citation
```
TODO: bibtex citation
```

## Contributing

**Repo Organization:** The outline of this repo is centered around the package `protossl`, defined in the subdirectory of the same name. While the package definition enables experimentation, the `scripts` folder defines all of the actual experiment run scripts. Most of the experimental comparisons we run have standalone implementations also contained within the `scripts` folder.

**Environment Setup:** We use pre-commit hooks to maintain coding style and to enforce data stewardship. Source data should never be committed to this repo, including exploratory notebooks which may accidentally leak source data. The pre-commit hooks do not necessarily prevent this and can be circumvented, but may help prevent obvious cases. To correctly set up your environment:
```bash
# 1) clone repo
git clone git@github.com:StevenSong/ProtoSSL.git
cd ProtoSSL

# 2) create and activate environment
# NOTE: if you don't use conda, make sure you're using the same python version, install from `requirements.txt`, and MAKE SURE YOU HAVE FFMPEG 5.* FOR TORCHCODEC (see below)
conda env create -f env.yaml
conda activate protossl

# 3) enable pre-commit hooks
pre-commit install

# 4) install protossl from editable source
pip install -e .

# 5) dev away
```

**torchcodec:** torchcodec is a bit fragile with dependencies. We've pinned `torch==2.7.0` which is compatible with `torchcodec==0.4.0`, both compiled against CUDA 12.8 (which we use on our machines). This torchcodec version is only compatible with `datasets==4.0.0`. If you see errors relating to torchcodec (you can diagnose this by just importing torchcodec), make sure the dependencies are compatible not just relative to versioning, but also relating to the CUDA versions. We also use `ffmpeg=5.*` installed via conda. If you see an error relating to not being able to find `libnppicc.so.12`, it might be that the linker can't find the binaries (which we ensure are available by installing `nvidia-npp-cu12`). To fix this, you can try setting the `LD_LIBRARY_PATH` environment variable:
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
