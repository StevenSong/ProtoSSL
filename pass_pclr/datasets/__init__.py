from typing import Type

from ..defines import CINC_TARGETS, ECHONEXT_TARGETS, PTBXL_TARGETS
from ._base_ecg_dataset import BaseECGDataset, StreamingECGWaveforms, load_cached_data
from ._cinc_dataset import CincECGDataset
from ._echonext_dataset import EchoNextECGDataset
from ._heedb_dataset import HeedbECGDataset
from ._pclr_wrapper_dataset import PCLRWrapperDataset
from ._ptbxl_dataset import PtbxlECGDataset, get_ptbxl_labels


def infer_dataset_class_from_path(
    dataset_path: str,
) -> tuple[
    Type[BaseECGDataset],
    list[str] | None,  # label names
]:
    echonext_indicators = ["echonext", "echo-next", "echo_next"]
    ptbxl_indicators = ["ptbxl", "ptb-xl", "ptb_xl"]
    cinc_indicators = ["cinc", "cinc-2020", "cinc_2020", "cinc2020"]
    heedb_indicators = ["heedb"]

    if any(x in dataset_path for x in echonext_indicators):
        return EchoNextECGDataset, list(ECHONEXT_TARGETS.keys())
    elif any(x in dataset_path for x in ptbxl_indicators):
        return PtbxlECGDataset, PTBXL_TARGETS
    elif any(x in dataset_path for x in cinc_indicators):
        return CincECGDataset, CINC_TARGETS
    elif any(x in dataset_path for x in heedb_indicators):
        return HeedbECGDataset, None
    raise ValueError(
        f"Could not infer BaseECGDataset subclass from dataset_path: {dataset_path}"
    )
