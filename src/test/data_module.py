from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.commons.dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(self, test_data_dir):
        super().__init__()
        self.test_data_dir = test_data_dir

        self.test_data = None

    def setup(self, stage: str = None):
        if stage == "test" and not self.test_data:
            self.test_data = Dataset(imgs_dir=self.test_data_dir)

    def test_dataloader(self):
        print(f"test_dataloader, {len(self.test_data)}")
        return self.__create_data_loader(dataset=self.test_data)

    def __create_data_loader(self, dataset):
        return DataLoader(dataset=dataset, batch_size=32)
