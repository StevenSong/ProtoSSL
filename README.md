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
# NOTE: you don't have to use conda, just make sure you're using the same python version and install from `requirements.txt` instead
conda env create -f env.yaml
conda activate protossl

# 3) enable pre-commit hooks
pre-commit install

# 4) install protossl from editable source
pip install -e .

# 5) dev away
```
