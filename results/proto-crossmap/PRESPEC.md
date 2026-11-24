# EchoNext → PTB-XL prototype cross-mapping: prespecification

Written before any EchoNext × PTB-XL association was computed. The commit that adds
this file and `expected_pairs.csv` is the record of what was fixed in advance.

## Question

Each of the 1000 pretrained (HEEDB) prototypes has an association with every EchoNext
label and every PTB-XL statement. Do any EchoNext labels (echo findings) rely on the
same prototypes as a PTB-XL statement that we would **not** expect to be related?

We want to rule out that a surprising pairing just reflects PTB-XL's assignment. That
assignment gives exactly 14 prototypes to each of its 71 statements (994 of 1000), so
weakly supported prototypes get pigeonholed into some statement anyway.

## Inputs

- `association_matrix.npy` (prototypes × labels) from
  `protossl-heedb-pila/learn-prototype-assignments/latest`. Use `runs-echonext` and
  `runs-ptbxl`, seeds 42, 67, 70, 73, 99, under `$PROTOSSL_OUTPUTS/ecg`.
- This matrix is the ILP's input: a robust effect size of each prototype's activation,
  positives vs balanced negatives, computed on **train** labels.
- The prototype bank is the same at every seed. Only the training subset and the
  negative resampling change.
- `assignment_metadata.csv` from the same directory, giving the ILP's assignment per
  seed.

## What was already known when this was written

- Earlier sessions saw the PTB-XL readings of the prototypes shared by EchoNext and
  MIMIC for `LVEF Lo` (MI, aneurysm, ILBBB) and `AS` (LVH, LPR, CRBBB, ILBBB). They also
  saw the EchoNext × MIMIC overlap screen, including `LVWT Hi` ~ ESRD.
- Positive control, using only classical datasets and no EchoNext:
  - Spearman ρ across the 1000 prototypes recovers the matching PTB-XL statement as the
    top hit for almost every CODE-15% and CinC label. Examples: RBBB→CRBBB 0.78/0.92,
    LAD→LAFB 0.91, LVH→VCLVH 0.91. The weakest correct matches are ST→STACH 0.55 and
    SB→SBRAD 0.60 (CODE-15%).
  - The median ρ over all pairs is about 0.1, a general "abnormal ECG" axis.
  - The control also turns up the kinds of surprise we expect to find here: AF→DIG
    (treatment, 0.58–0.74) and RBBB→RVH (labels co-occurring within PTB-XL, 0.70–0.81).
- The association matrices barely change across seeds: PTB-XL column correlation
  between seeds 42 and 67 has median 0.993. So seed stability can test the ILP
  assignment, but not the ILP-free statistic.
- 14 PTB-XL statements have fewer than 30 train positives: PRC(S), INJIL, 3AVB, 2AVB,
  INJLA, PMI, INJIN, TRIGU, PSVT, SVTAC, STE_, SEHYP, IPMI, TAB_. The ILP's
  `n_min=1` keeps them eligible, but their effect sizes are noisy.

## Expected pairs

`expected_pairs.csv` lists, for each EchoNext label, the PTB-XL statements expected
to be related, with a sign and a level:

- `expected`: an established ECG correlate of the echo finding.
- `plausible`: an indirect link through comorbidity, treatment, devices, or secondary
  physiology.

`SHD` inherits every pair of its components at the strongest level. NORM and SR are
expected negative for every label, and SARRH is plausibly negative.

A pair counts as **unexpected** if no row has its EchoNext label, PTB-XL statement and
sign. It counts as **not expected** if it is listed only as plausible.

The list is provisional: drafted from general clinical knowledge, pending cardiologist
review. Additions after the results are seen will be reported as post hoc and will not
change the prespecified classification.

## Analysis

1. **ILP-free map (primary).** For each of the 12 × 71 = 852 pairs, compute
   ρ(e, s): the Spearman correlation across the 1000 prototypes between EchoNext
   column e and PTB-XL column s, at seed 42. Report the min–max over the 5 seeds.
2. **Specificity.** Double-center the 12 × 71 ρ matrix (ρ minus its row mean and
   column mean, plus the grand mean). This separates statements tied to one echo
   label from statements, like most abnormal-ECG findings, that track every echo label.
3. **Candidates.** A pair is a candidate if:
   - |ρ| ≥ 0.5, the level of the weakest correct control matches; or
   - it is among the 3 statements with the highest ρ for its EchoNext label, so labels
     with weak ECG signal still get a ranked list.
4. **Survivors.** A candidate survives if all of these hold:
   - it is unexpected or not expected;
   - its PTB-XL statement has at least 30 train positives;
   - its specificity has the same sign as ρ;
   - ρ has the same sign at all 5 seeds.

   Survivors are reported in two tiers, unexpected and not expected.
5. **Pigeonhole check of the ILP map.** This answers the question as originally put.
   - For each seed and each EchoNext label, take its 14 assigned prototypes and read off
     each one's PTB-XL statement.
   - For each pair, count the shared prototypes at each seed, against a chance level of
     about 14 × 14 / 1000 ≈ 0.2.
   - For each shared prototype, find the rank of the assigned statement in that
     prototype's PTB-XL row (1 = its best statement), and its PTB-XL association score
     as a percentile of all 994 assigned scores.
   - An ILP pairing is **pigeonholed** if its prototypes are mostly (≥ half, pooled
     over seeds) at rank > 3 or below the 25th score percentile, or if its pair is not a
     step-3 candidate.
   - Every survivor from step 4 gets the same per-prototype table.

## What a survivor means and does not

- A survivor is a hypothesis to explain, not a finding.
- The first explanations to rule out are the ones the control already showed:
  treatment, comorbidity, and labels that co-occur within PTB-XL.
- No p-values are reported: prototypes are not independent, and the associations come
  from train labels.
- A held-out replication would need full-bank activations on the test splits of both
  datasets. That takes a GPU job, and is decided after we see the survivors.
