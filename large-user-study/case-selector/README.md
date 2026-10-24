# HEEDB case selector

Offline tool for a collaborator to browse the HEEDB test set (same split as the
user study models, 362,090 ECGs) and pick user study cases. The bundle is a
plain folder that opens by double-clicking `index.html`, with no install,
server, or network access needed.

- `build_case_selector.py` pre-renders every ECG with `protossl.plotting.plot_ecg`
  to PDF and writes the label/metadata index the viewer loads.
- `index.html` is the static viewer. It filters by age, year, sex and site, and by
  label (require/exclude, all/any, for any label source: physician overread,
  original MUSE, or ProtoECGNet fusion predictions), draws random matches or looks up an ECG ID, shows the
  printout and labels, and stores selections in `localStorage` with CSV/JSON
  export and import.
- `HOW_TO_USE.txt` holds the instructions for the collaborator. It is copied into the bundle.

## Build

```bash
export CACHE_DIR=/home/songs1/.cache/protossl_cache
# timing / smoke test on a random subset
python build_case_selector.py --out-dir /path/to/bundle-test --limit 3000
# full bundle, resumable, existing PDFs are skipped
python build_case_selector.py --out-dir /path/to/heedb-case-selector --workers 64
```

Fusion predictions come from `--fusion-dir`, which defaults to the patient-level
split retrain in `protoecgnet-heedb-v3`. The build binarizes `test_probs.npy` at
per-label Youden's J thresholds fit on `val_probs.npy` against the physician
overread, the same way `notebooks/muse-accuracy.ipynb` does. The thresholds are
saved in `index.js` metadata.

`--years` keeps only ECGs acquired in the given years, inclusive, so you can drop
ECGs whose acquisition-time MUSE labels came from older software. It takes years
and ranges separated by commas, for example `--years 2018-2022` or
`--years 2013,2016-2018`. It is applied before `--limit`, and the viewer header
shows it.

Measured on the cluster, 3k and 20k ECG batches:

- About 430 ECG/s steady state at 64 to 128 workers. Waveform reads are the
  bottleneck, so more workers don't help.
- The full set takes about 15 to 20 minutes.
- PDFs average 88 KB, about 31 GiB for the full set. `index.js` is about 14 MB.

`index.js` is written last and lists any ECG whose PDF failed to render, so the
viewer can say so. To refresh only the index, rerun with `--index-only`.

## Delivery

Copy the whole output folder onto an encrypted drive or into an encrypted
archive. It contains HEEDB data covered by the DUA. The collaborator should
open `index.html` in Chrome or Edge. Firefox also displays the PDFs inline.
Safari is untested. Every browser has an "Open PDF" link as a fallback.

## What comes back

The CSV/JSON export has one row per selected ECG with these fields:

- `ecg_id`, `patient_id`
- `selected_at`, the collaborator's `note`
- `filter_at_selection`, the filter that was active when the case was picked
- age, sex, year, site
- the positive labels under each label source

The JSON export also carries the bundle's build metadata, including the commit.
