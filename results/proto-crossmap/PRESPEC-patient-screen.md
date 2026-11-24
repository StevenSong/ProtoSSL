# EchoNext × imputed PTB-XL statements, patient-level screen: prespecification

Written before any patient-level association between an imputed PTB-XL statement and an
EchoNext label was computed on EchoNext train, and before any was computed on val/test
other than the one pair listed below. The commit that adds this file and
`patient_screen.py` is the record of what was fixed in advance.

## Why this follow-up

The prototype-level cross-mapping (`PRESPEC.md`, 754f94a) found no EchoNext → PTB-XL
pairing that was both unexpected and robust. That method was weak in three ways: the
prototypes have about 20 effective dimensions, so chance |ρ| reaches about 0.6; its
specificity null was underpowered; and the ILP assignment pigeonholes rare statements.
This screen asks the same question with patients as the unit: does the imputed
probability of a PTB-XL statement predict an echo finding that we would not expect it
to predict?

## Inputs

- Imputed PTB-XL statement probabilities on EchoNext (job 8, run `recur`, seed 42):
  `protossl-heedb-pila-from-ptbxl/{train,test}_probs.npy` and
  `protossl-heedb-pila-from-ptbxl-val/val_probs.npy` under
  `$PROTOSSL_SHARED_ROOT/runs/recur/protossl-outputs-seed42/runs-echonext/`. They come
  from PTB-XL's linear probe on PTB-XL's projected prototypes and never saw EchoNext
  labels.
- `EchoNext_metadata_100k.csv`: the 12 echo flags, age, sex, acquisition year, and the
  machine intervals (ventricular rate, PR, QRS, QTc).
- `expected_pairs.csv` as committed in 754f94a, unchanged. Tiers are defined as in
  `PRESPEC.md`: **expected**, **not expected** (listed only as plausible), and
  **unexpected** (not listed with that sign).

## What was already known when this was written

- On train: the prototype-level association matrices and every result in
  `echonext-ptbxl-crossmap.ipynb`, including its four unexpected survivors
  (PEff~−HVOLT, LVWT Hi~INJAL, PR~LMI, PEff~ISCAS).
- On val+test: **PEff ~ ISCAS** was examined at the patient level (OR 1.6–2.2 per SD).
  It is excluded from confirmation. Other statements were examined only *against* PEff
  and RVD in that post hoc section (T-wave regional statements); those pairs stay
  eligible but are flagged in the output as `seen_valtest`.
- The imputation quality on MIMIC: AUROC 0.96–0.97 for LBBB/RBBB/AF, 0.85–0.86 for
  infarct and LVH, 0.68–0.69 for ST-T and LAE.
- On train, the 71 imputed log-odds are nearly collinear: median VIF 417, LVH and
  PSVT/SVTAC over 2000. So adjusting each statement for all 70 others is not estimable;
  the adjustment below is limited to the expected statements.

## Data

- **Discovery:** EchoNext train, all 72,475 ECGs from 26,218 patients. Standard errors
  are clustered by patient.
- **Confirmation:** EchoNext val + test pooled, 10,068 patients with one ECG each. No
  patient is in both sets.
- Each statement enters as its standardized log-odds `z_s`: the probability clipped to
  [1e-6, 1 − 1e-6], logit, then centered and scaled with the **train** mean and SD.

## Models

For echo label `e` and statement `s`, logistic regressions:

- `COV`: B-splines (df 4) of age, ventricular rate, QRS, QTc and acquisition year; sex;
  PR interval, with missing values set to the train median plus a missing indicator.
- **M_cov:** `e ~ COV + z_s`.
- **M_exp (primary):** `e ~ COV + z_s + {z_t : t listed as expected for e, t ≠ s}`.
  This asks whether `s` carries information about `e` beyond the statements already
  expected to predict `e`. NORM and SR are expected (negative) for every label, so a
  general abnormal-ECG axis is always adjusted for.
- Effect size: the M_exp coefficient of `z_s` times the SD of the residual of `z_s`
  after regressing it on the adjustment statements (OLS, train). This is the OR per SD
  of the part of `s` not shared with the expected statements, `OR_res`. The R² of that
  OLS is reported.
- The tier of a pair is read from `expected_pairs.csv` using the sign of the M_exp
  coefficient.

## Discovery rule (train)

Tested pairs: the 11 echo labels other than the composite SHD, crossed with the 57
statements with at least 30 PTB-XL train positives: 627 pairs.

A pair is a **candidate** if all hold:

1. M_exp p < 0.05 / 627 (patient-clustered Wald);
2. M_cov p < 0.05 / 627, with the same sign as M_exp (guards against suppression);
3. |log OR_res| ≥ log 1.1;
4. tier is unexpected or not expected;
5. it is not PEff ~ ISCAS.

At most **20** candidates go forward: unexpected before not expected, then by M_exp
|z| within each tier.

**Positive control:** expected-tier pairs that meet rules 1–3. The top 10 by M_exp |z|
go forward as a separate family. They test whether the pipeline confirms what it should.

If a model fails to converge it is recorded and the pair cannot be a candidate.

## Confirmation rule (val + test)

Refit M_exp for each carried pair on val + test, with the same adjustment set. `COV`
splines take their knots from the data being fit. Standardization stays at the train
mean and SD; `OR_res` uses the train residual SD.

- A candidate is **confirmed** if its M_exp coefficient has the discovery sign and its
  Holm-adjusted p < 0.05 over the carried candidates.
- The positive-control family is judged the same way, with Holm over its own pairs.

Reported descriptively, without tests: M_cov on val + test, and M_exp plus the other 10
non-composite echo flags (does a co-occurring echo finding explain the pairing?).

## What a confirmed pair means and does not

- A confirmed pair is a reproducible association between an imputed ECG reading and an
  echo finding, beyond demographics, intervals and the expected readings. It is a
  hypothesis to explain, not a new ECG finding.
- The explanations to rule out first are the ones the prototype-level work already
  showed: treatment (AF→DIG), labels that co-occur within PTB-XL (RBBB→RVH), a
  co-occurring echo finding, and a known correlate missing from the provisional list.
- The imputed statements are a linear readout of PTB-XL's prototypes, so this is still
  prototype space. Weakly imputed statements (ST-T, LAE) can give false nulls.
- EchoNext has no diagnoses, medications or surgical history, so confounding by those
  cannot be adjusted for.
- Power on val + test is low for the rare labels (PR 41 positives, AR 128, PEff 121).
- A null result (no confirmed candidate) is reported as such.
