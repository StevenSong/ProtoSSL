import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

from ._base_ecg_dataset import BaseECGDataset


class PCLRWrapperDataset(Dataset):
    def __init__(self, dataset: BaseECGDataset):
        self.ds = dataset
        multi_sample_patients, self.patient_sample_map = get_sample_to_patient_mapping(
            patient_ids=self.ds.patient_ids.numpy(),
        )
        self.patient_ids = torch.as_tensor(multi_sample_patients)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        pid = self.patient_ids[i]
        sample_idxs = self.patient_sample_map[i].indices
        i1, i2 = np.random.choice(sample_idxs, 2, replace=False)
        x1 = self.ds[i1]
        x2 = self.ds[i2]
        assert x1["patient_id"] == pid
        assert x2["patient_id"] == pid
        assert x1["ecg_id"] != x2["ecg_id"]
        return {
            "patient_id": pid,
            "x1": x1["waveform"],
            "x2": x2["waveform"],
        }

    def __len__(self) -> int:
        return self.patient_ids.shape[0]


def get_sample_to_patient_mapping(
    *,  # enforce kwargs
    patient_ids: np.ndarray,
    min_samples_per_patient: int = 2,
) -> tuple[
    np.ndarray,  # patients with min_samples_per_patient
    csr_matrix,  # patient -> sample mapping
]:
    # Given upper bound of ~10M samples from ~2M patients, need to construct a
    # sparse matrix of masks to find all the samples belonging to each patient.

    # Below example to illustrate how the masks are constructed:
    # eg: patient_ids = [3, 4, 3, 9, 4, 3, 6, 6]

    (
        unique_patients,  # eg: [3, 4, 6, 9]
        unique_counts,  #   eg: [3, 2, 2, 1]
    ) = np.unique(patient_ids, return_counts=True)

    n_samples = len(patient_ids)  # eg: 8
    n_patients = len(unique_patients)  # eg: 4

    rows = np.searchsorted(  # find indices of patient_ids in unique_patients
        unique_patients,  # must be sorted in ascending, given by np.unique
        patient_ids,
        side="left",
    )  # eg: [0, 1, 0, 3, 1, 0, 2, 2]

    assert (
        rows < len(unique_patients)
    ).all()  # make sure no values are off the right edge
    assert (
        unique_patients[rows] == patient_ids
    ).all()  # in combination with above, make sure no values are off the left edge

    cols = np.arange(n_samples)  # eg: [0, 1, 2, 3, 4, 5, 6, 7]
    data = np.ones(n_samples, dtype=bool)  # eg: [1, 1, 1, 1, 1, 1, 1, 1]

    # eg:
    # [[1, 0, 1, 0, 0, 1, 0, 0],
    #  [0, 1, 0, 0, 1, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0, 1, 1],
    #  [0, 0, 0, 1, 0, 0, 0, 0]]
    # keep in mind the rows are patients, the cols are samples
    per_pt_samples = csr_matrix((data, (rows, cols)), shape=(n_patients, n_samples))

    # filter to multi-ecg patients
    mask = unique_counts >= min_samples_per_patient
    multi_sample_patients = unique_patients[mask]
    msp_per_pt_samples = per_pt_samples[mask, :]
    return multi_sample_patients, msp_per_pt_samples
