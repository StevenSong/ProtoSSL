from typing import Type

from ..defines import ECHONEXT_TARGETS
from ._base_ecg_dataset import BaseECGDataset, load_cached_data
from ._echonext_dataset import EchoNextECGDataset
from ._pclr_wrapper_dataset import PCLRWrapperDataset
from ._ptbxl_dataset import PtbxlECGDataset

# from ._heedb_dataset import HeedbECGDataset


def infer_dataset_class_from_path(
    dataset_path: str,
) -> tuple[
    Type[BaseECGDataset],
    list[str] | None,  # label names
]:
    echonext_indicators = ["echonext", "echo-next", "echo_next"]
    ptbxl_indicators = ["ptbxl", "ptb-xl", "ptb_xl"]

    if any(x in dataset_path for x in echonext_indicators):
        return EchoNextECGDataset, list(ECHONEXT_TARGETS.keys())
    elif any(x in dataset_path for x in ptbxl_indicators):
        return PtbxlECGDataset, None
    raise ValueError(
        f"Could not infer BaseECGDataset subclass from dataset_path: {dataset_path}"
    )
