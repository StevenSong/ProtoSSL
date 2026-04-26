from typing import Type

from ..defines import (
    AUDIOSET_TARGETS,
    CINC_TARGETS,
    CODE15_TARGETS,
    ECHONEXT_TARGETS,
    ESC_50_TARGETS,
    HEEDB_TARGETS,
    IEMOCAP_TARGETS,
    MIMIC_TARGETS,
    PTBXL_TARGETS,
    SPEECH_COMMANDS_V2_TARGETS,
    URBANSOUND8K_TARGETS,
    VOXCELEB1_ID_TARGETS,
    ZZU_TARGETS,
)
from ._audioset_contrastive_wrapper_dataset import AudioSetContrastiveWrapperDataset
from ._audioset_dataset import AudioSetDataset
from ._base_dataset import BaseTSDataset, load_cached_data, validate_label_subset
from ._cinc_dataset import CincECGDataset
from ._code15_dataset import Code15ECGDataset
from ._echonext_dataset import EchoNextECGDataset
from ._esc_50_dataset import Esc50Dataset
from ._heedb_dataset import HeedbECGDataset, get_heedb_labels, get_heedb_metadata
from ._iemocap_dataset import IemocapDataset
from ._mimic_dataset import MimicECGDataset
from ._pclr_wrapper_dataset import PCLRWrapperDataset
from ._ptbxl_dataset import PtbxlECGDataset, get_ptbxl_labels
from ._speech_commands_dataset import SpeechCommandsV2Dataset
from ._urbansound8k_dataset import UrbanSound8kDataset
from ._voxceleb1id_dataset import VoxCeleb1IdDataset
from ._zzu_dataset import ZzuECGDataset


def infer_dataset_class_from_path(
    dataset_path: str,
) -> tuple[
    Type[BaseTSDataset],
    list[str] | None,  # label names
    bool,  # is audio
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
    speech_cmds_indicators = ["speech-commands", "speech_commands", "speechcommands"]
    esc50_indicators = ["esc-50", "esc_50", "esc50"]
    us8k_indicators = ["urbansound", "urban_sound", "urban-sound"]
    voxceleb1_indicators = ["voxceleb1", "vox_celeb_1", "vox-celeb-1"]
    iemocap_indicators = ["iemocap"]

    if any(x in dataset_path_lower for x in echonext_indicators):
        return EchoNextECGDataset, list(ECHONEXT_TARGETS.keys()), False
    elif any(x in dataset_path_lower for x in ptbxl_indicators):
        return PtbxlECGDataset, PTBXL_TARGETS, False
    elif any(x in dataset_path_lower for x in cinc_indicators):
        return CincECGDataset, CINC_TARGETS, False
    elif any(x in dataset_path_lower for x in heedb_indicators):
        return HeedbECGDataset, list(HEEDB_TARGETS.keys()), False
    elif any(x in dataset_path_lower for x in mimic_indicators):
        return MimicECGDataset, MIMIC_TARGETS, False
    elif any(x in dataset_path_lower for x in zzu_indicators):
        return ZzuECGDataset, list(ZZU_TARGETS), False
    elif any(x in dataset_path_lower for x in code15_indicators):
        return Code15ECGDataset, CODE15_TARGETS, False
    elif any(x in dataset_path_lower for x in audioset_indicators):
        return AudioSetDataset, list(AUDIOSET_TARGETS), True
    elif any(x in dataset_path_lower for x in speech_cmds_indicators):
        return SpeechCommandsV2Dataset, SPEECH_COMMANDS_V2_TARGETS, True
    elif any(x in dataset_path_lower for x in esc50_indicators):
        return Esc50Dataset, ESC_50_TARGETS, True
    elif any(x in dataset_path_lower for x in us8k_indicators):
        return UrbanSound8kDataset, URBANSOUND8K_TARGETS, True
    elif any(x in dataset_path_lower for x in voxceleb1_indicators):
        return VoxCeleb1IdDataset, VOXCELEB1_ID_TARGETS, True
    elif any(x in dataset_path_lower for x in iemocap_indicators):
        return IemocapDataset, IEMOCAP_TARGETS, True
    raise ValueError(
        f"Could not infer BaseECGDataset subclass from dataset_path: {dataset_path}"
    )
