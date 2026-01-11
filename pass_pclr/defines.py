import os
from typing import Literal

STAGE_T = Literal["learn-prototypes", "project-prototypes", "train-classifier"]
SPLIT_T = Literal["train", "val", "test"]
RESNET_T = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

CACHE_DIR = os.environ.get(
    "CACHE_DIR", "/opt/gpudata/steven/ecg-prototype-transfer/pass_pclr_cache"
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
