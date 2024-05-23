import os

import pytorch_lightning as pl
from sklearn.model_selection import KFold

from src.commons.model import Model
from src.train.data_module import DataModule


class ModelTrainer:

    def __init__(self, folds, epoch, learning_rate, debug):
        self.data_dir = self.__get_data_dir(debug)
        self.folds = folds
        self.epoch = epoch
        self.learning_rate = learning_rate

    def fit(self):
        fold_splits = self.__create_folds()
        datamodule = self.__create_datamodule()

        for _, (train_indices, val_indices) in enumerate(fold_splits):
            datamodule.set_folds(train_indices, val_indices)
            model = self.__create_model()
            trainer = self.__create_trainer()
            trainer.fit(model=model, datamodule=datamodule)

    def __get_data_dir(self, debug):
        return "dataset/faces_train_debug" if debug else "dataset/faces_train"

    def __create_folds(self):
        total_count = len(os.listdir(self.data_dir))
        kfold = KFold(n_splits=self.folds)
        return kfold.split(list(range(total_count)))

    def __create_datamodule(self):
        return DataModule(train_data_dir=self.data_dir)

    def __create_model(self):
        return Model(learning_rate=self.learning_rate)

    def __create_trainer(self):
        callbacks = [
            pl.callbacks.ModelCheckpoint(monitor="val/mae", dirpath="log/checkpoints", save_top_k=1,
                                         mode="min", save_weights_only=True, filename="best-checkpoint", )]
        return pl.Trainer(default_root_dir="log", callbacks=callbacks, logger=[], max_epochs=self.epoch,
                          num_sanity_val_steps=0, )
