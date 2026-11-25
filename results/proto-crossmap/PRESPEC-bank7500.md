# EchoNext → PTB-XL prototype cross-mapping on the 7500-prototype bank: prespecification

Written before any EchoNext × PTB-XL association on the 7500-prototype bank was computed.
The commit that adds this file and `scripts/8-run-protossl-heedb-150-pila-from-ptbxl.sh` is
the record of what was fixed in advance.

This repeats the EchoNext analysis of `PRESPEC.md` (754f94a), which used the 1000-prototype
bank, on the 7500-prototype HEEDB-150 bank. Steps 1–4 and the expected-pairs list are
unchanged, so the two banks can be compared. Three things change:

- step 5 is adapted to a bank that PTB-XL's assignment covers only partly;
- the random-positives null is prespecified this time (step 6; it was post hoc on the 1000
  bank), with the null-supported rule of `PRESPEC-mimic-zzu.md` (66ddb6b);
- positive controls (step 7) and a comparison with the 1000 bank (step 8) are added.

## Question

As in 754f94a: do any EchoNext labels (echo findings) rely on the same pretrained prototypes
as a PTB-XL statement that we would **not** expect to be related? And is any such surprise
more than a weak prototype pigeonholed into PTB-XL's assignment?

Added here: does the larger bank change the answer? On the 1000 bank the map was null. The
activations had about 20 effective dimensions, and random positives reached |ρ| ≈ 0.6 at
every n. A bank of 7500 prototypes, pretrained on 150 HEEDB labels, might resolve finer
prototypes specific to one label.

## Inputs

- **Bank.** `protossl-heedb-150`: 7500 prototypes, pretrained without labels on HEEDB
  (`configs/pretrain-unsupervised-heedb-150.yaml`). Archived at
  `ecg-rebuttal/protossl-heedb-150`, and seeded into `protossl-shared/runs/paper/` for the
  PTB-XL jobs.
- **Association and assignment matrices.** `association_matrix.npy` and
  `assignment_metadata.csv` come from `protossl-heedb-150-pila/learn-prototype-assignments/latest`,
  at seeds 42, 67, 70, 73 and 99:
  - EchoNext: `$PROTOSSL_OUTPUTS/ecg-rebuttal/protossl-outputs-seed<seed>/runs-echonext/`.
    This was run during the paper rebuttal. Shape 7500 × 12; 168 assigned prototypes (12 × 14).
  - PTB-XL: `protossl-shared/runs/bank7500/protossl-outputs-seed<seed>/runs-ptbxl/`. Run on
    2026-11-25 at 66ddb6b (slurm 30715–30719, `PROTOSSL_PRETRAIN_RUN=paper`). Shape
    7500 × 71; 994 assigned prototypes (71 × 14).
  - The two datasets were run with different code versions. ProtoSSL's seed-67 reproduction
    check (2026-11-19) found PILA bit-identical across the code changes in between, so this
    isn't expected to matter.
- **Step 6 inputs:** activations of PTB-XL's 994 projected prototypes, seed 42.
  - PTB-XL train: PTB-XL's own `compute-embeddings/latest/train_embeds.npy`, 17418 × 994.
  - EchoNext train and test: job 8 on the 7500 bank. **Not run yet.**
    `scripts/8-run-protossl-heedb-150-pila-from-ptbxl.sh echonext bank7500 42` is added in this
    commit. Step 6 is done once that job is published, with no other change.
- **Expected pairs:** `expected_pairs.csv`, unchanged from 754f94a.

## What was already known when this was written

- Every 1000-bank result was known: `echonext-ptbxl-crossmap.ipynb` (prespecified sections
  and post hoc ones), `null_summary.csv`, the patient-level screen (8edf734), and the MIMIC
  and ZZU cross-maps (66ddb6b), which were also null. In particular:
  - Prespecified map: 40 survivors, mostly generic. The 4 unexpected survivors were PEff ~ −HVOLT,
    LVWT Hi ~ INJAL, PR ~ LMI and PEff ~ ISCAS.
  - Under the specificity null, nothing survived at q < 0.05. Textbook pairs reached only
    q 0.14–0.21.
  - In the ILP map, 66 of 84 pairings were pigeonholed. The striking ones sat on rare
    statements (e.g. TRV Hi → 2AVB, 12 train positives) and were small-n noise.
- About the 7500-bank runs, only these were seen: that the jobs finished and published, file
  listings, array shapes, and the row counts of `assignment_metadata.csv` (168 and 994). No
  association, assignment or activation value was looked at.
- The 7500 bank's PTB-XL linear-probe metrics were in the job logs, but weren't read for this.

## Expected pairs

`expected_pairs.csv` (754f94a) is used as is: 623 signed rows at levels *expected* and
*plausible*. SHD inherits its components' pairs. NORM and SR are expected negative for every
label, and SARRH is plausibly negative. The tiers are defined as in `PRESPEC.md`.

The 1000-bank analysis turned up omissions, such as PEff ~ −HVOLT (the low-voltage mirror).
They are deliberately **not** added, so that each pair has the same tier on both banks.

The list is provisional: it was drafted by an AI assistant from general clinical knowledge
and awaits cardiologist review.

## Analysis

K = 7500 throughout, except in step 6. There are 12 × 71 = 852 pairs.

1. **ILP-free map (primary).** ρ(e, s) is the Spearman correlation, across the 7500 bank
   prototypes, between EchoNext column e and PTB-XL column s, at seed 42. Report the min–max
   over the 5 seeds.
2. **Specificity.** Double-center the 12 × 71 ρ matrix, as in 754f94a.
3. **Candidates**, as in 754f94a: |ρ| ≥ 0.5, or among the label's 3 highest-ρ statements.
   The cutoff is kept for comparability, even though on the 1000 bank it sat inside the
   noise; step 6 carries the inference.
4. **Survivors**, as in 754f94a: a candidate survives if it is unexpected or not expected,
   has ≥ 30 PTB-XL train positives, has specificity of the same sign as ρ, and keeps the sign
   of ρ at all 5 seeds. Reported in two tiers: unexpected and not expected.
   - Any of the four unexpected 1000-bank survivors that survives again is reported as a
     **replication**, not as a new surprise.
5. **Pigeonhole check of the ILP map**, adapted. PTB-XL's ILP fills 994 of the 7500
   prototypes, so most of an EchoNext label's 14 prototypes may have no PTB-XL statement.
   - For each seed and each EchoNext label, take its 14 assigned prototypes. Report the
     fraction that PTB-XL also assigns, against a chance level of 994 / 7500 ≈ 0.13.
   - For each prototype that PTB-XL assigns, report as in 754f94a: its statement, the rank
     of that statement in the prototype's PTB-XL row, and its score percentile among the 994
     assigned scores.
   - For each prototype PTB-XL doesn't assign, report descriptively the statement at rank 1
     of its PTB-XL row, and that score's percentile among the 994 assigned scores. These
     don't form ILP pairings.
   - Count shared prototypes per pair and seed, against a chance level of
     14 × 14 / 7500 ≈ 0.026.
   - An ILP pairing is pigeonholed by the 754f94a rule. That is, the pairing is pigeonholed
     if either of these holds:
     - most of its shared prototypes (≥ half, pooled over seeds) sit at row rank > 3 or
       below the 25th score percentile;
     - the pair isn't a step-3 candidate.
   - Every step-4 survivor gets the same per-prototype table. Unassigned prototypes are
     marked in it.
6. **Random-positives null (prespecified now).** This is the method of section 4 of
   `echonext-ptbxl-crossmap.ipynb`, in the space of the 7500-bank PTB-XL model's 994
   projected prototypes, at seed 42.
   - Association matrices in projected space:
     - EchoNext train and test: from job 8's activations and the EchoNext labels.
     - PTB-XL train: from PTB-XL's train activations.
     - All are built as the ILP builds them (`build_association_matrix`, n_min 1, trim 0.10,
       10 negative repeats, balanced negatives).
   - ρ_proj and spec_proj are steps 1–2 recomputed in this space.
   - For each PTB-XL statement with n train positives, draw random sets of n PTB-XL train
     ECGs: 2000 sets for n ≤ 300, 200 for n ≤ 2000, 100 above.
   - Build each set's effect-size column the same way, then take ρ against the fixed
     EchoNext train columns.
   - `p_null` is two-sided on |ρ|. `p_spec_null` uses the specificity matrix rebuilt with
     the statement's column replaced. q values are BH over the 852 pairs.
   - A survivor is **null-supported** if `q_spec_null` < 0.05 and its specificity has the
     same sign in bank and projected space.
   - Also reported, descriptively:
     - ρ_proj on EchoNext **test**, which was held out from the association step.
     - A bridge between the two spaces. On the 1000 bank the projected prototypes were
       nearly the whole bank (994 of 1000). Here they're a subset (13%) that PTB-XL picked
       for strong PTB-XL associations. So also report bank-space ρ over those 994 prototypes,
       before projection, at seed 42. Report its Pearson r with ρ over all 7500, and with
       ρ_proj.
   - Run under the shared-host limits (≤ 16 niced workers).
7. **Positive controls (new).** Each label below should have one of the listed statements
   among its 3 highest-ρ statements, after removing NORM, SR and SARRH (these are trivially
   negative for every label). Each row lists that label's specific *expected* positive
   statements, from `expected_pairs.csv`.

   | label | any of |
   |---|---|
   | LVEF Lo | CLBBB, ILBBB, IVCD, ANEUR, IMI, ASMI, ILMI, AMI, ALMI, LMI, IPLMI, IPMI, PMI |
   | LVWT Hi | LVH, VCLVH, HVOLT, SEHYP |
   | AS | LVH, VCLVH, HVOLT, SEHYP |
   | AR | LVH, VCLVH, HVOLT |
   | PEff | LOWT, LVOLT |
   | RVD | IRBBB, CRBBB, RVH, RAO/RAE, LPFB |
   | PASP Hi | IRBBB, CRBBB, RVH, RAO/RAE, LPFB |
   | TRV Hi | IRBBB, CRBBB, RVH, RAO/RAE, LPFB |

   If most controls fail, the map isn't reading these labels, and their survivors aren't
   interpreted.

   **Power benchmarks.** Three pairs reached only q 0.14–0.21 under the specificity null on
   the 1000 bank: LVEF Lo ~ ILBBB, PEff ~ LVOLT (both expected) and PR ~ LPFB (plausible).
   Their `p_spec_null` and `q_spec_null` on the 7500 bank are the prespecified measure of
   whether the larger bank resolves label-specific prototypes.
8. **Comparison with the 1000 bank (new, descriptive).** All at seed 42:
   - Pearson r of ρ and of specificity across the 852 pairs, 1000 bank vs 7500 bank.
   - Each 1000-bank survivor's status on the 7500 bank, and the reverse.
   - The noise floor of ρ: the null's 99th percentile of |ρ|, by n, against ≈ 0.6 on the 1000
     bank. The same for specificity.
   - Effective dimensionality: the participation ratio, (Σλ)² / Σλ², of the eigenvalues of
     the correlation matrix of PTB-XL train activations in projected space. Computed the same
     way for both banks' PTB-XL models.

## How the result is read

- **Larger bank helps** if both hold:
  - the noise floor in step 8 (the null's 99th percentile of |ρ|) is below 0.5 at every n,
    against 0.55–0.65 on the 1000 bank;
  - at least two of the three power benchmarks reach q_spec_null < 0.05.
  - If so, null-supported unexpected or specific not-expected survivors are hypotheses to
    explain, as in 754f94a.
  - Treatment, comorbidity and co-labeling within PTB-XL are ruled out first.
  - A follow-up patient-level test would need its own prespec.
- **Larger bank doesn't help** if the benchmarks stay at q ≥ 0.05 and there are no
  null-supported pairs.
  - Then the prototype-level cross-map is null on both banks, and the cross-map approach is
    closed for EchoNext.
  - "Nothing found" is still not evidence that nothing is there, because the test is weak.
- Anything in between is reported as is, without a verdict.
- As in 754f94a, no p-values are reported for ρ itself: prototypes aren't independent, and
  the associations come from train labels. The step-6 null calibrates only the PTB-XL side,
  in projected space.
