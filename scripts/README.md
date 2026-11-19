# Experiment Run Scripts

This folder contains the run scripts for all of our ECG experiments. Each script runs one experiment (or one experiment plus its fine-tuned `-ft` variant) and takes its inputs as arguments:

```bash
scripts/<N>-run-<experiment>.sh <dataset> <run> <seed>   # downstream experiments
scripts/pretrain/run-<model>-pretrain.sh <run>           # HEEDB pretraining
```

* `<dataset>`: a short name (`echonext`, `ptbxl`, `cinc`, `mimic`, `zzu`, `code15`), optionally with a training-subset suffix, e.g. `ptbxl-256`. Resolved to `/opt/gpudata/ecg/<dir>` by [`_datasets.sh`](_datasets.sh).
* `<run>`: a name for one full set of results, e.g. `v1`.
* `<seed>`: random seed. The paper uses 42, 67, 70, 73 and 99.

The audio scripts are archived in [`archive/audio/`](../archive/README.md); they predate this layout and hardcode our old paths.

## Running jobs

Jobs go through the runner in [`slurm/`](../slurm/), which runs a **pushed commit**, never a working tree:

```bash
# one experiment
slurm/submit.sh --gres=gpu:1 -- scripts/2-run-supproto-direct.sh ptbxl-256 v1 42

# pretraining (loads HEEDB into memory)
slurm/submit.sh --gres=gpu:1 --mem-per-gpu=500G -- scripts/pretrain/run-protossl-heedb-pretrain.sh v1

# every downstream experiment, once the run's pretrained models are published
scripts/queue-experiments.sh v1
```

`slurm/submit.sh` refuses to submit from a dirty tree or a commit that isn't on GitHub. On each node, `slurm/job.sh` makes a shallow clone of that commit under `~/protossl-run/commits/<sha>/`, runs `uv sync --locked`, and then runs the script there. Clones are never modified, so jobs at different commits can't interfere. Thread pools are capped at `--cpus-per-task` (default 72, a GPU's share of a node).

The scripts call `python` directly; outside the runner, activate the uv env (`.venv`) first.

## Outputs

Experiments are staged on node-local disk and copied to shared storage when they finish (without their `wandb/` dirs, which sync to the wandb server). Each copy is atomic, so a partial copy is never visible:

| | Path |
|---|---|
| Run root (node-local) | `$PROTOSSL_RUN_ROOT`, default `~/protossl-run`: commit clones, uv cache, logs, staged outputs |
| Shared root | `$PROTOSSL_SHARED_ROOT`, default `/shared/gpudata/songs1/protossl-shared` |
| Pretrained models | `<shared>/runs/<run>/<pretrain-exp>`, e.g. `runs/v1/protossl-heedb` |
| Downstream experiments | `<shared>/runs/<run>/protossl-outputs-seed<seed>/runs-<dataset>/<exp>` |
| Job logs | `<shared>/logs/<job-id>.out` |

The downstream layout matches our archived results, so the `results/` notebooks can read a new run by changing their results root. `wandb/` dirs aren't published, so `meta-results.ipynb`'s runtime comparison, which reads them, works only on the archive. Each published experiment has a `COMMIT` file. A script refuses to start if its experiment is already published to that run.

Downstream experiments read pretrained models from their own run; set `PROTOSSL_PRETRAIN_RUN=<other-run>` (passed through by `sbatch`) to use another run's.

## Caches and dependencies between jobs

* Datasets are preprocessed once per node into `$CACHE_DIR` (default `~/.cache/protossl_cache`). The first job on a node that needs a cache builds it and the others wait for it, so no job has to run first. [`0-run-cache-data.sh <dataset>`](0-run-cache-data.sh) warms a node without holding a GPU.
* `9-1`..`9-3` each start by running [`9-0-run-ecgfounder-patches.sh`](9-0-run-ecgfounder-patches.sh), which builds the ECGFounder patch embeddings once per node and keeps them in the node-local stage. They are several GB for the full datasets, so they are never published.

## Foundation model weights

Jobs never download weights. They read them from `/opt/gpudata/models/` and check them against the sha256 sums in [`_common.sh`](_common.sh):

| Baseline | Path |
|---|---|
| ECGFounder | `/opt/gpudata/models/PKUDigitalHealth/ECGFounder/12_lead_ECGFounder.pth` ([Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder)) |
| ST-MEM | `/opt/gpudata/models/st-mem/st_mem_vit_base_encoder.pth` (linked from [ST-MEM](https://github.com/vuno/ST-MEM)) |
