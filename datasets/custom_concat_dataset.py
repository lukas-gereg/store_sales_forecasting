from torch.utils.data import ConcatDataset

from assignment.datasets.base_dataset import BaseDataset


class CustomConcatDataset(ConcatDataset, BaseDataset):
    def __init__(self, datasets: list[BaseDataset]):
        super().__init__(datasets)
        self.base_datasets = datasets

    def find_y_by_index(self, idx: int):
        pass