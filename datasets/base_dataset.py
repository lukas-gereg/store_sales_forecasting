from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):

    @abstractmethod
    def find_y_by_index(self, idx: int):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass