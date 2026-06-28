import torch
from torch.utils.data import DataLoader, Dataset

class TestDataset(Dataset):
    def __init__(self, data_path: str, transform_type: str = "none"):
        self.data_path = data_path
        self.transform_type = transform_type
        # Simulate loading data
        self.data = torch.randn(100, 3, 32, 32)
        self.labels = torch.randint(0, 10, (100,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TestDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, pin_memory=False, **kwargs):
        # Add custom logic, logging, or pre-processing flags here if needed
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            pin_memory=pin_memory, 
            **kwargs
        )

    def __len__(self):
        return len(self.dataset)
    
    def __get_item__():
        x, y 

    @staticmethod
    def collate_fn(batch):
        data, labels = zip(*batch)


def create_dataset(text: str):
    data = torch.randint(0, 100, (3, 32, 32))
    labels = torch.randint(0, 1, (3, 10))
    return data, labels


wow = 2