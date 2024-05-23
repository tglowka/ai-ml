from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from src.commons.dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(self, train_data_dir):
        super().__init__()
        self.train_data_dir = train_data_dir

        self.train_indices = None
        self.val_indices = None

        self.all_train_data = None

    def set_folds(self, train_indices, val_indices):
        self.train_indices = train_indices
        self.val_indices = val_indices

    def setup(self, stage):
        if stage == "fit" and not self.all_train_data:
            print("setup: fit stage - load data")
            self.all_train_data = Dataset(imgs_dir=self.train_data_dir)

    def train_dataloader(self):
        print(f"train_dataloader, min: {min(self.train_indices)}, max: {max(self.train_indices)}")
        train_data = Subset(self.all_train_data, self.train_indices)
        return self.__create_data_loader(dataset=train_data)

    def val_dataloader(self):
        print(f"val_dataloader, min: {min(self.val_indices)}, max: {max(self.val_indices)}")
        val_data = Subset(self.all_train_data, self.val_indices)
        return self.__create_data_loader(dataset=val_data)

    def __create_data_loader(self, dataset):
        return DataLoader(dataset=dataset, batch_size=32, num_workers=3)
