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

The scripts predate the job runner in `slurm/`. They hardcode our old paths (`/opt/gpu_working/steven/...`), including the old locations of these configs, so they don't run as they are. They also need the conda env (`env.yaml`) for ffmpeg. The audio datasets and models are still in the `protossl` package.
