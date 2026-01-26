# FEMP: Foundational ECG Model of Prototypes

This repo contains the source code for our Foundational ECG Model of Prototypes (**FEMP**).

We introduce a novel framework **PASS-PCLR** (Prototype-Attended Self-Supervision for Patient Contrastive Learning of Representations) to do self-supervised pretraining of prototypes, and demonstrate that foundational prototype models trained using PASS-PCLR outperform label-supervised pretraining.

### Citation
```
TODO: bibtex citation
```

## Contributing

**Repo Organization:** The outline of this repo is centered around the package `pass_pclr`, defined in the subdirectory of the same name. While the package definition enables experimentation, the `scripts` folder defines all of the actual experiment run scripts. Most of the experimental comparisons we run have standalone implementations also contained within the `scripts` folder. The exceptions to this are the comparisons to literature models, whose forked/modified implementations are in submodules under the `external` folder, notably `ProtoECGNet` and the `Columbia Mini-Model`.

**Environment Setup:** We use precommit hooks to maintain coding style and to enforce data stewardship. Source data should generally never be committed to this repo, including exploratory notebooks which may accidentally leak source data. To correctly set up your environment:
```bash
# 1) clone repo
git clone git@github.com:StevenSong/ecg-prototype-fm.git
cd ecg-prototype-fm

# 2) create and activate environment
# NOTE: you don't have to use conda, just make sure you're using the same python version and install from `requirements.txt` instead
conda env create -f env.yaml
conda activate ecg

# 3) enable pre-commit hooks
pre-commit install

# 4) install pass_pclr from editable source
pip install -e .

# 5) dev away
```
