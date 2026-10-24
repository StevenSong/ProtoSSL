"""Rules for tagging where a 12SL code sits in a HEEDB physician overread.

HEEDB physician overreads were free text mapped to 12SL codes by fuzzy matching, so a code
being present doesn't mean the current ECG shows it. `tag_read` gives every (read, label) the
lowest bucket over the label's occurrences:

    1 main         stated in the main read (before "when compared with ecg of")
    2 hedged       main read, directly preceded by "possible" or "cannot rule out"
    3 cmp_current  only in the serial comparison, as the current finding
                   ("X has replaced Y", "X is now present"); the comparison starts as MUSE
                   text and physicians often leave it as is, so it can contradict the main
                   read (e.g. for VT, 93% verbatim from the machine comparison, while the
                   machine's own main read never states VT)
    4 prior        only as the prior ECG's finding ("Y has replaced X", "X is no longer present")
    5 cmp_other    only in the comparison section, grammar not parsed
    6 dump         only inside an expansion block: a run of sibling statements that can't all
                   be true at once (every "tachycardia" statement, every "bradycardia"
                   statement, every "with ..." modifier)

Grammar notes, checked against the data:
  - in "X has replaced Y" (X, 1415, Y), X is the current ECG; X matches the main read ~98%
  - hedges precede the statement they modify ("cannot rule out" 1682 -> "anterior infarct")

The rules were validated only on VALIDATED_TARGETS. Any change to this file changes
RULES_HASH, which invalidates cached `audited_physician` and `audited_muse` labels.
`audited_muse` applies the same tagging to MUSE reads (codes_software), whose comparison
text has the same grammar; it found no expansion blocks there for these targets.
"""

import collections
import hashlib

import numpy as np

from ..defines import UNKNOWN_LABEL

BUCKETS = {
    1: "main",
    2: "hedged",
    3: "cmp_current",
    4: "prior",
    5: "cmp_other",
    6: "dump",
}
# `audited_physician` label per bucket: main and hedged are positive; absent and prior are
# negative ("Y has replaced X" says X is gone); the rest can't be told apart, so unknown
POSITIVE_BUCKETS = (1, 2)
UNKNOWN_BUCKETS = (3, 5, 6)

# the 20 HEEDB_TARGETS the rules were audited on, frozen so a code change is caught
VALIDATED_TARGETS = {
    "ANTERIOR INFARCT": [740],
    "ATRIAL FIBRILLATION": [161],
    "ATRIAL FLUTTER": [162, 273, 288],
    "ATRIAL-PACED RHYTHM": [295],
    "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK": [445],
    "INFERIOR INFARCT": [780],
    "LATERAL INFARCT": [760],
    "LEFT BUNDLE BRANCH BLOCK": [460],
    "NORMAL SINUS RHYTHM": [22],
    "PREMATURE ATRIAL COMPLEXES": [222],
    "PREMATURE VENTRICULAR COMPLEXES": [231],
    "RIGHT AXIS DEVIATION": [383],
    "RIGHT BUNDLE BRANCH BLOCK": [440],
    "SINUS BRADYCARDIA": [21],
    "SINUS RHYTHM": [19],
    "SINUS TACHYCARDIA": [23],
    "VENTRICULAR TACHYCARDIA": [248, 272, 277],
    "VENTRICULAR-PACED RHYTHM": [296],
    "WITH 1ST DEGREE AV BLOCK": [101],
    "WITH SINUS ARRHYTHMIA": [251],
}

# codes whose diagnoses_dictionary.csv text starts with "with " (same at MGB and Emory)
WITH_CODES = {101, 102, 103, 104, 105, 106, 107, 108, 111, 112, 113, 141, 142, 143, 144, 171, 172, 173, 174, 175, 181, 190, 211, 212, 242, 243, 245, 251, 252, 274, 275, 276, 277, 278, 283, 284, 285, 303, 326, 543, 544, 545, 795, 1084, 1254, 1255}  # fmt: skip

# expansion families: code -> multiplicity in the canonical block, and the number of distinct
# family codes a contiguous run needs to count as a block. copies of a code beyond its
# multiplicity are real statements (e.g. the real "sinus tachycardia" right after the block)
TACHY = {272: 1, 43: 1, 63: 1, 277: 2, 280: 1, 271: 1, 26: 1, 28: 1, 248: 1, 236: 1, 270: 1, 235: 1, 164: 1, 23: 1}  # fmt: skip
BRADY = {41: 1, 61: 1, 265: 1, 25: 1, 24: 1, 27: 1, 34: 1, 21: 1}
FAMILIES = [
    ("tachy", TACHY, 10),
    ("brady", BRADY, 6),
    ("with", {c: 1 for c in WITH_CODES} | {277: 2}, 20),
]
# short VT/SVT block: a tachy run that contains all four of these but is under the threshold
VT_SHORT = {248, 271, 272, 277}

COMPARE = 1301  # "when compared with ecg of"
HAS_REPLACED = 1415
# qualifier following the code: is now / are now / present / less or more frequent / not changed
CURRENT_NEXT = {1406, 1409, 1410, 1403, 1404, 1408, 1411}
PRIOR_NEXT = {1405, 1418}  # is no longer / are no longer
QUALIFIERS = set(range(1400, 1430))
HEDGES = {1680, 1682}  # possible / cannot rule out
PERIOD = 446

with open(__file__, "rb") as _f:
    RULES_HASH = hashlib.sha256(_f.read()).hexdigest()[:16]


def make_code_to_labels(targets: dict[str, list[int]]) -> dict[int, list[int]]:
    """Map each 12SL code to the column indices of the targets that use it."""
    code_to_labels: dict[int, list[int]] = collections.defaultdict(list)
    for i, codes in enumerate(targets.values()):
        for c in codes:
            code_to_labels[c].append(i)
    return dict(code_to_labels)


def audited_labels(buckets: np.ndarray) -> np.ndarray:
    """Map per-label buckets from `tag_read` to 1 / 0 / UNKNOWN_LABEL."""
    out = np.isin(buckets, POSITIVE_BUCKETS).astype(np.int64)
    out[np.isin(buckets, UNKNOWN_BUCKETS)] = UNKNOWN_LABEL
    return out


def audited_predictions(buckets: np.ndarray) -> np.ndarray:
    """Map per-label buckets from `tag_read` on a MUSE read to hard 1 / 0 predictions:
    a prediction can't be unknown, so anything outside the main read is 0."""
    return np.isin(buckets, POSITIVE_BUCKETS).astype(np.int64)


def expansion_mask(codes: list[int]) -> tuple[list[bool], set[str]]:
    mask = [False] * len(codes)
    kinds = set()
    for name, family, min_distinct in FAMILIES:
        i = 0
        while i < len(codes):
            if codes[i] not in family:
                i += 1
                continue
            j = i
            while j < len(codes) and codes[j] in family:
                j += 1
            run = codes[i:j]
            if len(set(run)) >= min_distinct:
                kinds.add(name)
                used = collections.Counter()
                for k in range(i, j):
                    used[codes[k]] += 1
                    mask[k] = mask[k] or used[codes[k]] <= family[codes[k]]
            elif name == "tachy" and VT_SHORT <= set(run):
                kinds.add("vt_short")
                for k in range(i, j):
                    mask[k] = mask[k] or codes[k] in VT_SHORT
            i = j
    return mask, kinds


def occurrence_bucket(codes: list[int], k: int, cut: int) -> int:
    nxt = codes[k + 1] if k + 1 < len(codes) else None
    if nxt == HAS_REPLACED or nxt in CURRENT_NEXT:
        return 3
    if nxt in PRIOR_NEXT:
        return 4
    # "Y has replaced X [modifiers]": walk back within the statement to the nearest qualifier
    j = k - 1
    while (
        j >= 0
        and k - j <= 3
        and codes[j] not in QUALIFIERS
        and codes[j] not in (PERIOD, COMPARE)
    ):
        j -= 1
    if j >= 0 and codes[j] == HAS_REPLACED:
        return 4
    if k >= cut:
        return 5
    if k > 0 and codes[k - 1] in HEDGES:
        return 2
    return 1


def tag_read(
    code_str, code_to_labels: dict[int, list[int]], n_labels: int
) -> tuple[np.ndarray, str]:
    """Per-label bucket (0 = absent) for one comma-separated code string, plus the
    expansion kinds found ("|"-joined)."""
    out = np.zeros(n_labels, dtype=np.uint8)
    if not isinstance(code_str, str) or not code_str:
        return out, ""
    codes = [int(c) for c in code_str.split(",")]
    if not any(c in code_to_labels for c in codes):
        return out, ""
    cut = codes.index(COMPARE) if COMPARE in codes else len(codes)
    mask, kinds = expansion_mask(codes)
    for k, c in enumerate(codes):
        if c not in code_to_labels:
            continue
        b = 6 if mask[k] else occurrence_bucket(codes, k, cut)
        for li in code_to_labels[c]:
            out[li] = b if out[li] == 0 else min(out[li], b)
    return out, "|".join(sorted(kinds))
