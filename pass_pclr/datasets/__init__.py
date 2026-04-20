from pathlib import Path
from typing import Type

import pandas as pd

from ..defines import (
    CINC_TARGETS,
    CODE15_TARGETS,
    ECHONEXT_TARGETS,
    HEEDB_TARGETS,
    MIMIC_TARGETS,
    PTBXL_TARGETS,
    ZZU_TARGETS,
)
from ._audioset_dataset import AudioSetDataset
from ._base_ecg_dataset import (
    BaseECGDataset,
    StreamingECGWaveforms,
    load_cached_data,
    validate_label_subset,
)
from ._cinc_dataset import CincECGDataset
from ._code15_dataset import Code15ECGDataset
from ._echonext_dataset import EchoNextECGDataset
from ._heedb_dataset import HeedbECGDataset, get_heedb_labels, get_heedb_metadata
from ._mimic_dataset import MimicECGDataset
from ._pclr_wrapper_dataset import PCLRWrapperDataset
from ._ptbxl_dataset import PtbxlECGDataset, get_ptbxl_labels
from ._zzu_dataset import ZzuECGDataset


def infer_dataset_class_from_path(
    dataset_path: str,
) -> tuple[
    Type[BaseECGDataset],
    list[str] | None,  # label names
]:
    dataset_path_lower = dataset_path.lower()

    echonext_indicators = ["echonext", "echo-next", "echo_next"]
    ptbxl_indicators = ["ptbxl", "ptb-xl", "ptb_xl"]
    cinc_indicators = ["cinc", "cinc-2020", "cinc_2020", "cinc2020"]
    heedb_indicators = ["heedb"]
    mimic_indicators = ["mimic"]
    zzu_indicators = ["zzu"]
    code15_indicators = ["code15"]
    audioset_indicators = ["audioset", "audio-set", "audio_set"]

    if any(x in dataset_path for x in echonext_indicators):
        return EchoNextECGDataset, list(ECHONEXT_TARGETS.keys())
    elif any(x in dataset_path for x in ptbxl_indicators):
        return PtbxlECGDataset, PTBXL_TARGETS
    elif any(x in dataset_path for x in cinc_indicators):
        return CincECGDataset, CINC_TARGETS
    elif any(x in dataset_path for x in heedb_indicators):
        return HeedbECGDataset, list(HEEDB_TARGETS.keys())
    elif any(x in dataset_path for x in mimic_indicators):
        return MimicECGDataset, MIMIC_TARGETS
    elif any(x in dataset_path for x in zzu_indicators):
        return ZzuECGDataset, list(ZZU_TARGETS)
    elif any(x in dataset_path for x in code15_indicators):
        return Code15ECGDataset, CODE15_TARGETS
    elif any(x in dataset_path_lower for x in audioset_indicators):
        class_csv = Path(dataset_path) / "audioset_train" / "class_labels_indices.csv"
        label_names = pd.read_csv(class_csv)["mid"].tolist()
        return AudioSetDataset, label_names
    raise ValueError(
        f"Could not infer BaseECGDataset subclass from dataset_path: {dataset_path}"
    )
