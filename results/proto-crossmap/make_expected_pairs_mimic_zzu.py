"""Build `expected_pairs_mimic_zzu.csv`: expected PTB-XL statements for each MIMIC and ZZU label.

Written before any MIMIC × PTB-XL or ZZU × PTB-XL association was computed (see
PRESPEC-mimic-zzu.md). The CSV is generated so the list can be reviewed by statement group
rather than row by row. Run from the repo root or from results/proto-crossmap:

    python results/proto-crossmap/make_expected_pairs_mimic_zzu.py

Levels follow PRESPEC.md (754f94a): `expected` is an established ECG correlate; `plausible`
is an indirect link (comorbidity, treatment, devices, secondary physiology). A label/statement/
sign not listed is `unexpected`. When a statement is listed twice for a label and sign, the
stronger level wins.
"""

from pathlib import Path

import pandas as pd

from protossl.defines import MIMIC_TARGETS, PTBXL_TARGETS, ZZU_TARGETS

HERE = Path(__file__).resolve().parent
P = list(PTBXL_TARGETS)

# ---- statement groups -------------------------------------------------------------------
MI = [
    "IMI",
    "ASMI",
    "ILMI",
    "AMI",
    "ALMI",
    "LMI",
    "IPLMI",
    "IPMI",
    "PMI",
    "QWAVE",
    "ANEUR",
]
INJ = ["INJAS", "INJAL", "INJIN", "INJLA", "INJIL", "STE_"]
STT = [
    "ISC_",
    "ISCAL",
    "ISCIN",
    "ISCIL",
    "ISCAS",
    "ISCLA",
    "ISCAN",
    "STD_",
    "NDT",
    "NST_",
    "NT_",
    "INVT",
    "LOWT",
    "TAB_",
]
NONSPEC_STT = ["NDT", "NST_", "NT_", "LOWT", "TAB_", "INVT"]
LVHG = ["LVH", "VCLVH", "HVOLT", "SEHYP", "LAO/LAE"]
RVG = ["RVH", "RAO/RAE", "IRBBB", "CRBBB", "LPFB"]
LVCOND = ["CLBBB", "ILBBB", "IVCD", "LAFB", "ABQRS"]
AVB = ["1AVB", "2AVB", "3AVB", "LPR"]
ATRIAL = ["AFIB", "AFLT", "PAC", "SVARR", "SVTAC", "PSVT"]
VENT = ["PVC", "BIGU", "TRIGU"]
NORMAL = ["NORM", "SR", "SARRH"]
ABN = [s for s in P if s not in NORMAL]  # every abnormal statement
ALL_GROUPS = MI + INJ + STT + LVHG + RVG + LVCOND + AVB + ATRIAL + VENT
_ungrouped = [s for s in ABN if s not in ALL_GROUPS]
assert sorted(_ungrouped) == sorted(
    ["DIG", "LNGQT", "EL", "WPW", "LVOLT", "STACH", "SBRAD", "PACE", "PRC(S)"]
), _ungrouped
assert len(set(ALL_GROUPS)) == len(ALL_GROUPS) and set(ALL_GROUPS) <= set(P)

RANK = {"expected": 2, "plausible": 1}


class Spec:
    """Accumulates (statement, sign) -> (level, basis) for one label."""

    def __init__(self):
        self.rows: dict[tuple[str, str], tuple[str, str]] = {}

    def add(self, statements, level: str, basis: str, sign: str = "+") -> "Spec":
        for s in [statements] if isinstance(statements, str) else statements:
            assert s in P, s
            old = self.rows.get((s, sign))
            if old is None or RANK[level] > RANK[old[0]]:
                self.rows[s, sign] = (level, basis)
        return self

    def base(self) -> "Spec":
        """NORM and SR expected negative, SARRH plausibly negative (as in 754f94a)."""
        self.add(
            ["NORM", "SR"],
            "expected",
            "an abnormal-ECG label is less common with a normal sinus ECG",
            "-",
        )
        return self.add(
            "SARRH",
            "plausible",
            "sinus arrhythmia marks younger, healthier patients",
            "-",
        )

    def generic(
        self, basis: str = "comorbid cardiovascular disease in older, sicker patients"
    ) -> "Spec":
        """Every abnormal statement except WPW plausibly positive.

        WPW is congenital and has no comorbidity or treatment path to most labels; it is added
        back where there is one.
        """
        return self.add([s for s in ABN if s != "WPW"], "plausible", basis)

    def tachy(
        self, level: str = "expected", basis: str = "sinus tachycardia in acute illness"
    ) -> "Spec":
        self.add("STACH", level, basis)
        return self.add("SBRAD", "plausible", "mirror of sinus tachycardia", "-")

    def lowvolt(self, level: str, basis: str) -> "Spec":
        self.add("LVOLT", level, basis)
        return self.add(["HVOLT", "VCLVH"], "plausible", "mirror of low voltage", "-")


# EchoNext lists as committed in 754f94a, reused for the MIMIC labels that match an echo label
_ECHONEXT = pd.read_csv(HERE / "expected_pairs.csv")


def inherit(spec: Spec, echo_labels: list[str]) -> Spec:
    for e in echo_labels:
        for r in _ECHONEXT[_ECHONEXT["echonext"] == e].itertuples():
            spec.add(r.ptbxl, r.level, f"as EchoNext {e} (754f94a): {r.basis}", r.sign)
    return spec


def critical(spec: Spec) -> Spec:
    """Acute critical illness: tachycardia, AF, nonspecific ST-T and QT changes, low voltage."""
    spec.base().generic().tachy()
    spec.add(["AFIB", "AFLT"], "expected", "new or rapid AF in critical illness")
    spec.add(NONSPEC_STT, "expected", "nonspecific ST-T changes in critical illness")
    spec.add(
        "LNGQT",
        "expected",
        "QT prolongation from drugs, electrolytes and critical illness",
    )
    spec.add("PVC", "expected", "ventricular ectopy in critical illness")
    return spec.lowvolt(
        "plausible", "edema, effusions, obesity and lung disease lower voltage"
    )


def respiratory(spec: Spec) -> Spec:
    critical(spec)
    spec.add(RVG, "expected", "acute or chronic cor pulmonale")
    return spec.lowvolt("expected", "hyperinflation and obesity lower voltage")


def long_mortality(spec: Spec) -> Spec:
    """Mortality over months to a year: chronic ECG markers of prognosis on top of acute illness."""
    critical(spec)
    spec.add(
        MI + LVHG + LVCOND + ["PACE", "1AVB"],
        "expected",
        "established prognostic ECG findings (infarct, LVH, conduction disease, pacing)",
    )
    return spec


def heart_failure(spec: Spec, echo_labels: list[str]) -> Spec:
    inherit(spec.base(), echo_labels)
    spec.generic("comorbid cardiovascular disease in heart failure")
    spec.add(
        ["AFIB", "AFLT", "PACE"],
        "expected",
        "AF and devices (CRT/ICD) in heart failure",
    )
    spec.add("DIG", "plausible", "digoxin in heart failure")
    return spec.lowvolt("plausible", "effusions and edema lower voltage")


L: dict[str, dict[str, Spec]] = {"mimic": {}, "zzu": {}}
m = L["mimic"]

# ---- MIMIC: acute illness and outcomes ------------------------------------------------
for lab in ["Severe hypoxemia", "Mechanical ventilation", "Respiratory failure"]:
    m[lab] = respiratory(Spec())
m["Pneumonitis due to inhalation of food or vomitus"] = critical(Spec())
for lab in ["Vasopressors", "Septicemia", "Gram negative septicemia", "Sepsis"]:
    m[lab] = critical(Spec())
m["ECMO"] = (
    critical(Spec())
    .add(INJ + MI + STT, "expected", "acute infarction and cardiogenic shock")
    .add(
        VENT + ["PACE"],
        "expected",
        "ventricular arrhythmia and devices before or on ECMO",
    )
)
m["Inotropes"] = critical(Spec()).add(
    MI + LVCOND + ["PACE"], "expected", "cardiogenic shock and advanced heart failure"
)
m["Cardiac arrest"] = (
    critical(Spec())
    .add(
        INJ + MI + STT + VENT,
        "expected",
        "acute coronary syndrome and ventricular arrhythmia",
    )
    .add(["3AVB", "2AVB", "SBRAD"], "plausible", "bradyarrhythmic arrest")
    .add("WPW", "plausible", "pre-excited AF degenerating to VF")
)
for lab in [
    "ICU admit within 24 hours",
    "ICU admit",
    "Mortality 24 hours",
    "Mortality 7 days",
    "Mortality stay",
]:
    m[lab] = critical(Spec())
for lab in [
    "Mortality 28 days",
    "Mortality 90 days",
    "Mortality 180 days",
    "Mortality 365 days",
]:
    m[lab] = long_mortality(Spec())
m["Delirium due to conditions classified elsewhere"] = (
    Spec().base().generic().tachy("plausible")
)

# ---- MIMIC: diagnosis groups --------------------------------------------------------------
m["Cardiac diagnosis"] = (
    Spec()
    .base()
    .generic()
    .add(
        ALL_GROUPS + ["PACE", "LNGQT", "WPW", "PRC(S)", "DIG", "SBRAD"],
        "expected",
        "a cardiac diagnosis is the reading's own disease",
    )
)
# Non-cardiac diagnosis is positive for ~99% of ECGs; its negatives are visits with only cardiac
# diagnoses, so the cardiac readings are expected *negative* and a normal ECG positive.
m["Non-cardiac diagnosis"] = (
    Spec()
    .add(
        ALL_GROUPS + ["PACE", "LNGQT", "WPW", "PRC(S)", "DIG"],
        "plausible",
        "negatives are visits with only cardiac diagnoses",
        "-",
    )
    .add(
        ["NORM", "SR", "SARRH"],
        "plausible",
        "negatives are visits with only cardiac diagnoses",
    )
    .add("STACH", "plausible", "sinus tachycardia in non-cardiac acute illness")
)

# ---- MIMIC: metabolic, renal, hepatic, hematologic --------------------------------------
for lab in ["Protein-calorie malnutrition", "Severe protein-calorie malnutrition"]:
    m[lab] = (
        Spec()
        .base()
        .generic()
        .tachy("plausible")
        .lowvolt("expected", "cachexia lowers voltage")
        .add(["LNGQT", "EL"], "expected", "electrolyte disturbance and QT prolongation")
        .add("SBRAD", "plausible", "bradycardia in severe malnutrition")
    )
m["Disorders of phosphorus metabolism"] = (
    Spec()
    .base()
    .generic()
    .add(
        ["EL", "LNGQT", "TAB_"], "expected", "electrolyte disturbance in renal failure"
    )
    .add(LVHG, "expected", "LVH in chronic kidney disease")
)
m["Hyperosmolality and/or hypernatremia"] = (
    Spec()
    .base()
    .generic()
    .tachy()
    .add(["EL", "LNGQT"], "plausible", "electrolyte disturbance")
)
m["Fluid overload"] = (
    heart_failure(Spec(), ["LVEF Lo"])
    .add(LVHG, "expected", "LVH in renal failure and heart failure")
    .add(["EL"], "plausible", "renal failure")
)
m["Pancytopenia"] = Spec().base().generic().tachy()
m["Other and unspecified coagulation defects"] = (
    Spec()
    .base()
    .generic()
    .tachy("plausible")
    .add(
        ["AFIB", "AFLT", "PACE"],
        "plausible",
        "anticoagulation for AF and valves is coded as a coagulation defect",
    )
)
m["Alcoholic liver damage"] = (
    Spec()
    .base()
    .generic()
    .tachy()
    .add("LNGQT", "expected", "QT prolongation in cirrhosis")
    .add(["AFIB"], "plausible", "alcohol-related AF")
    .lowvolt("plausible", "ascites and edema lower voltage")
)
m["Ascites (non malignant)"] = (
    Spec()
    .base()
    .generic()
    .tachy()
    .add("LNGQT", "expected", "QT prolongation in cirrhosis")
    .lowvolt("expected", "ascites lowers voltage")
    .add(RVG, "plausible", "right heart failure causes ascites")
)
m["End stage renal disease"] = (
    Spec()
    .base()
    .generic()
    .add(LVHG, "expected", "LVH in end stage renal disease")
    .add(
        ["EL", "LNGQT", "TAB_"],
        "expected",
        "electrolyte disturbance (peaked T, QT) in dialysis",
    )
    .add(["AFIB"], "expected", "AF in dialysis")
)
m["Pleurisy; pleural effusion"] = (
    Spec()
    .base()
    .generic()
    .tachy("plausible")
    .lowvolt("expected", "effusion lowers voltage")
    .add(["AFIB", "AFLT"], "plausible", "heart failure and post-thoracic-surgery AF")
)

# ---- MIMIC: cardiac diagnoses -------------------------------------------------------------
m["Nonrheumatic aortic valve disorders"] = (
    inherit(Spec().base(), ["AS", "AR"])
    .generic()
    .add(
        ["PACE", "CLBBB", "3AVB"],
        "expected",
        "conduction disease and pacing after TAVR/AVR",
    )
)
m["Heart valve replaced"] = (
    inherit(Spec().base(), ["AS", "AR", "MR"])
    .generic()
    .add(
        LVCOND + ["1AVB", "3AVB", "PACE", "AFIB", "AFLT"],
        "expected",
        "post-procedural conduction disease, pacing and AF after valve replacement",
    )
    .add("DIG", "plausible", "digoxin for post-operative AF")
)
m["Myocardial infarction"] = (
    Spec()
    .base()
    .generic()
    .add(MI + INJ + STT, "expected", "infarct, injury and ischemic ST-T")
    .add(
        LVCOND + VENT + ["AFIB"],
        "expected",
        "conduction disease, ectopy and AF with infarction",
    )
)
m["Other chronic ischemic heart disease, unspecified"] = (
    Spec()
    .base()
    .generic()
    .add(MI + STT, "expected", "old infarct and ischemic ST-T")
    .add(LVCOND + ["PACE"], "expected", "conduction disease and devices")
)
m["Chronic pulmonary heart disease"] = (
    inherit(Spec().base(), ["PASP Hi", "RVD"])
    .generic()
    .lowvolt("plausible", "hyperinflation lowers voltage")
)
m["Primary/intrinsic cardiomyopathies"] = (
    inherit(Spec().base(), ["LVEF Lo", "LVWT Hi"])
    .generic()
    .add("WPW", "plausible", "pre-excitation in HCM phenocopies (PRKAG2, Danon, Fabry)")
)
m["Paroxysmal ventricular tachycardia"] = (
    Spec()
    .base()
    .generic()
    .add(
        VENT + MI + ["LNGQT", "PACE"],
        "expected",
        "ectopy, scar, long QT and ICDs in VT",
    )
    .add(LVCOND, "expected", "conduction disease in cardiomyopathy")
    .add("WPW", "plausible", "pre-excited AF coded as VT")
)
for lab in ["Atrial fibrillation", "Atrial flutter"]:
    m[lab] = (
        Spec()
        .base()
        .generic()
        .add(
            ATRIAL + ["BIGU", "LAO/LAE", "PACE", "DIG"],
            "expected",
            "the arrhythmia, its substrate (LAE), treatment (digoxin) and pacing",
        )
        .add("SBRAD", "plausible", "rate control and sick sinus")
        .add("WPW", "plausible", "pre-excitation predisposes to AF and flutter")
    )
m["Congestive heart failure (chf) nos"] = heart_failure(Spec(), ["LVEF Lo"])
m["Heart failure with reduced ef [systolic or combined heart failure]"] = heart_failure(
    Spec(), ["LVEF Lo"]
)
m["Heart failure with preserved ef [diastolic heart failure]"] = heart_failure(
    Spec(), ["LVWT Hi"]
).add(LVHG, "expected", "LVH and LAE in HFpEF")

# ---- ZZU (pediatric) ------------------------------------------------------------------------
# Children's normal variants (RV dominance, juvenile T inversion in V1-V3, faster rates, sinus
# arrhythmia) also occur in ZZU negatives, so they are not treated as expected.
z = L["zzu"]
z["Myocarditis"] = (
    Spec()
    .base()
    .tachy()
    .lowvolt("expected", "myocardial edema and effusion lower voltage")
    .add(STT + INJ, "expected", "ST-T changes and myopericardial ST elevation")
    .add(
        AVB + LVCOND + ["CRBBB", "IRBBB"] + VENT + ["LNGQT"],
        "expected",
        "conduction block, ectopy and QT prolongation in myocarditis",
    )
    .add(
        MI + ATRIAL + ["PACE", "RAO/RAE", "LAO/LAE", "LVH", "VCLVH"],
        "plausible",
        "pseudo-infarct Q waves, atrial arrhythmia, temporary pacing, chamber enlargement",
    )
    .add(
        ["SBRAD", "PRC(S)", "DIG"],
        "plausible",
        "sinus node involvement, ectopy, digoxin for heart failure",
    )
)
z["Cardiomyopathy"] = (
    inherit(Spec().base(), ["LVEF Lo", "LVWT Hi"])
    .add("QWAVE", "expected", "deep septal Q waves in HCM")
    .add(
        ["WPW", "LPR"],
        "plausible",
        "pre-excitation in HCM phenocopies (PRKAG2, Danon, Pompe) and LVNC",
    )
    .add("STACH", "expected", "sinus tachycardia in pediatric DCM")
)
z["Kawasaki disease"] = (
    Spec()
    .base()
    .tachy()
    .add(["1AVB", "LPR"], "expected", "PR prolongation in acute Kawasaki disease")
    .add(
        NONSPEC_STT + ["LNGQT"],
        "expected",
        "nonspecific ST-T and QT changes in acute Kawasaki disease",
    )
    .lowvolt("plausible", "pericardial effusion lowers voltage")
    .add(
        MI
        + INJ
        + STT
        + VENT
        + ATRIAL
        + [
            "2AVB",
            "3AVB",
            "CRBBB",
            "IRBBB",
            "IVCD",
            "LAFB",
            "LAO/LAE",
            "LVH",
            "VCLVH",
            "RAO/RAE",
        ],
        "plausible",
        "coronary aneurysm ischemia, myocarditis and arrhythmia",
    )
)
z["Congenital heart disease"] = (
    Spec()
    .base()
    .add(
        RVG, "expected", "RV pressure/volume load and RBBB after repair (ASD, TOF, PS)"
    )
    .add(
        ["LAFB", "1AVB", "LPR"],
        "expected",
        "superior axis and PR prolongation in AV septal defect",
    )
    .add(
        ["LVH", "VCLVH", "HVOLT", "SEHYP", "LAO/LAE"],
        "expected",
        "LV volume load (VSD, PDA) and biventricular hypertrophy",
    )
    .add(
        [
            "ISCAS",
            "INVT",
            "STD_",
            "ISC_",
            "NT_",
            "NDT",
            "NST_",
            "TAB_",
            "ABQRS",
            "IVCD",
            "CLBBB",
            "ILBBB",
            "2AVB",
            "3AVB",
            "PACE",
            "AFLT",
            "AFIB",
            "PAC",
            "SVARR",
            "SVTAC",
            "PSVT",
            "PVC",
            "BIGU",
            "TRIGU",
            "WPW",
            "STACH",
            "LNGQT",
            "QWAVE",
        ],
        "plausible",
        "RV strain, post-operative conduction disease, pacing, arrhythmia and associated syndromes",
    )
    .add("ASMI", "plausible", "qR in V1 from RV hypertrophy reads as septal infarct")
    .add(
        ["SBRAD", "DIG"],
        "plausible",
        "post-operative sinus node dysfunction; digoxin for shunt heart failure",
    )
)

# ---- write ------------------------------------------------------------------------------------
assert set(m) == set(MIMIC_TARGETS), set(MIMIC_TARGETS) ^ set(m)
assert set(z) == set(ZZU_TARGETS), set(ZZU_TARGETS) ^ set(z)
rows = [
    {"dataset": ds, "label": lab, "ptbxl": s, "sign": sign, "level": lv, "basis": basis}
    for ds, specs in L.items()
    for lab, spec in specs.items()
    for (s, sign), (lv, basis) in sorted(
        spec.rows.items(), key=lambda kv: (P.index(kv[0][0]), kv[0][1])
    )
]
out = pd.DataFrame(rows)
assert not out.duplicated(["dataset", "label", "ptbxl", "sign"]).any()
out.to_csv(HERE / "expected_pairs_mimic_zzu.csv", index=False)
print(out.groupby(["dataset", "level", "sign"]).size().to_string())
