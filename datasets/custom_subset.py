from collections.abc import Sequence

from torch.utils.data import Subset

from assignment.datasets.base_dataset import BaseDataset


class CustomSubset(Subset, BaseDataset):
    def __init__(self, dataset: BaseDataset, indices: Sequence[int]):
        super().__init__(dataset, indices)
        self.base_dataset = dataset

    def find_y_by_index(self, idx: int):
        original_idx = self.indices[idx]

        return self.base_dataset.find_y_by_index(original_idx)