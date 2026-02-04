import os
from typing import Literal

STAGE_T = Literal[
    "learn-prototypes",
    "project-prototypes",
    "compute-embeddings",
    "train-classifier",
]
SPLIT_T = Literal["train", "val", "test"]
RESNET_T = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
CONV_T = Literal["1D", "2D"]
PROT_T = Literal["global", "partial"]

CACHE_DIR = os.environ.get(
    "CACHE_DIR", "/opt/gpudata/steven/ecg-prototype-fm/pass_pclr_cache"
)

ECHONEXT_TARGETS = {
    "LVEF Lo": "lvef_lte_45_flag",
    "LVWT Hi": "lvwt_gte_13_flag",
    "AS": "aortic_stenosis_moderate_or_greater_flag",
    "AR": "aortic_regurgitation_moderate_or_greater_flag",
    "MR": "mitral_regurgitation_moderate_or_greater_flag",
    "TR": "tricuspid_regurgitation_moderate_or_greater_flag",
    "PR": "pulmonary_regurgitation_moderate_or_greater_flag",
    "RVD": "rv_systolic_dysfunction_moderate_or_greater_flag",
    "PEff": "pericardial_effusion_moderate_large_flag",
    "PASP Hi": "pasp_gte_45_flag",
    "TRV Hi": "tr_max_gte_32_flag",
    "SHD": "shd_moderate_or_greater_flag",
}
ECHONEXT_COMPOSITE_TARGET = "SHD"

# PTB-XL per-lead stats derived over train set at 100 Hz source freq
PTBXL_LOWERS = [
    -0.578000009059906,
    -1.0010000467300415,
    -1.4780000448226929,
    -1.0700000524520874,
    -0.6340000033378601,
    -1.1770000457763672,
    -1.99399995803833,
    -2.9170000553131104,
    -2.80300097322464,
    -1.9490000009536743,
    -1.2680000066757202,
    -1.2860000133514404,
]
PTBXL_UPPERS = [
    1.281999945640564,
    1.2860000133514404,
    1.024999976158142,
    0.5490000247955322,
    1.2109999656677246,
    1.0720000267028809,
    1.194000005722046,
    1.5880000591278076,
    1.9220000505447388,
    2.3429999351501465,
    2.315000057220459,
    1.9420000314712524,
]
PTBXL_CLIPPED_MEANS = [
    -0.0014159958645543634,
    -0.001479711227877473,
    6.0939658138210604e-05,
    0.0014383493131816163,
    -0.0006760021908785217,
    -0.000655445949668584,
    -5.988515305221437e-05,
    -0.0007678032012820134,
    -0.001334909725610109,
    -0.0012739597437105195,
    -0.0009959602659567962,
    -0.0013185259187448554,
]
PTBXL_CLIPPED_STDS = [
    0.15117879228950215,
    0.1586419909185117,
    0.1545075591236697,
    0.13436559524360595,
    0.13091674024953823,
    0.13714748111111613,
    0.21859729369814795,
    0.3250053863256649,
    0.31942013115571544,
    0.2862004454780644,
    0.25799056957570754,
    0.22582135786289945,
]

PTBXL_CAT1_TARGETS = [
    "1AVB",
    "IVCD",
    "3AVB",
    "2AVB",
    "SR",
    "AFIB",
    "STACH",
    "SARRH",
    "SBRAD",
    "PACE",
    "SVARR",
    "BIGU",
    "AFLT",
    "SVTAC",
    "PSVT",
    "TRIGU",
]


# HEEDB per-lead stats derived over train set (data before 2021) at 250 or 500 Hz source freq
# 250 Hz samples were weighted equally to 500 Hz samples (see scripts/_compute_heedb_normalizations.py)
HEEDB_LOWERS = [
    -1.0119186174765404,
    -1.4390830926038203,
    -1.7198238541919237,
    -1.1813091118326642,
    -0.9979542501410862,
    -1.4866640276672654,
    -1.7597675495680554,
    -2.5090881325221543,
    -2.3733176732190997,
    -1.8311383205364702,
    -1.5223512471635987,
    -1.6062121084788448,
]
HEEDB_UPPERS = [
    1.3460418197872064,
    1.5057521770008284,
    1.3717106293294237,
    0.9382710634263587,
    1.2941140484114941,
    1.3165757358737804,
    1.193413322647441,
    1.714575963086197,
    1.7340191625060277,
    1.9032238785473585,
    1.8878508873821698,
    1.7253424693450583,
]
HEEDB_CLIPPED_MEANS = [
    -0.0006269784908739345,
    -0.0032141328048856037,
    -0.0025533607205344565,
    0.0019646277729164162,
    0.0010130726248037739,
    -0.002853690357906745,
    -0.005724228465493457,
    -0.004851620104470237,
    -0.003503455167473047,
    -0.0013349983274631835,
    -0.001980557557397161,
    -0.00492753875377437,
]
HEEDB_CLIPPED_STDS = [
    0.15993755158963452,
    0.1935362312377656,
    0.19184339812753923,
    0.1503154757829258,
    0.1483126573949368,
    0.17524810046198744,
    0.19213824105393,
    0.2794094184989707,
    0.2748612949764635,
    0.24639481851816147,
    0.22539578223901716,
    0.2110454711343999,
]
