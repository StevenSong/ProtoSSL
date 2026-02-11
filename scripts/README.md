# Experiment Run Scripts

This folder contains the run scripts for all of our experiments. The run scripts in the root of the `scripts` directory are reusable, requiring only changing the input data paths and output directories via env vars.

We rarely call individual run scripts directly (with one exception, detailed below), instead relying on `slurm` to orchestrate our experiments. Jobs are submitted to slurm via the `queue-*-experiments.sh` scripts, which also set the appropriate env vars to control input/output paths for the run scripts.

The only run scripts we execute directly (or submit manually to slurm) are the pretraining jobs under the `pretrain` subdirectory.

If you're trying to replicate our experiments, we'd suggest the following:
1. Find and replace paths which start with `/opt/gpudata/steven` - these are the only places which contain hardcoded paths and should only be present within a handful of run scripts
1. Run `pretrain/run-pass-heedb-pretrain.sh`
1. Run all `queue-*-experiments.sh`
