import numpy as np
import pyarrow.parquet as pq
from datasets import Features


class IndexedParquetDataset:
    def __init__(self, shard_files, features: Features):
        self.shards = sorted(shard_files)
        self.features = features
        # Build cumulative index
        self.shard_sizes = [pq.read_metadata(s).num_rows for s in self.shards]
        self.cumulative = np.cumsum([0] + self.shard_sizes)
        self._columns = list(features.keys())

    def __len__(self):
        return self.cumulative[-1]

    def __getitem__(self, idx) -> dict:
        # Binary search for the right shard
        shard_idx = np.searchsorted(self.cumulative[1:], idx, side="right")
        local_idx = idx - self.cumulative[shard_idx]

        # Read just the needed row
        pf = pq.ParquetFile(self.shards[shard_idx])
        offset = 0
        for rg in range(pf.metadata.num_row_groups):
            rg_size = pf.metadata.row_group(rg).num_rows
            if local_idx < offset + rg_size:
                # Unwrap lists and decode features (only selected ones)
                table = pf.read_row_group(rg, columns=self._columns)
                raw = {
                    k: v[0]
                    for k, v in table.slice(local_idx - offset, 1).to_pydict().items()
                }
                return self.features.decode_example(raw)
            offset += rg_size
        raise ValueError("Not sure how we got here!")
