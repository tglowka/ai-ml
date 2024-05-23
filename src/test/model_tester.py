import pytorch_lightning as pl

from src.commons.model import Model
from src.test.data_module import DataModule


class ModelTester:
    def __init__(self, ckpt_paths, debug) -> None:
        self.test_data_dir = self.__get_data_dir(debug)
        self.ckpt_paths = ckpt_paths

    def test(self):
        for ckpt_path in self.ckpt_paths:
            datamodule = self.__create_data_module()
            model = self.__create_model()
            trainer = self.__create_trainer()
            print(ckpt_path)
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    def __get_data_dir(self, debug):
        return "dataset/faces_test_debug" if debug else "dataset/faces_test"

    def __create_data_module(self):
        return DataModule(test_data_dir=self.test_data_dir)

    def __create_model(self):
        return Model(None)

    def __create_trainer(self):
        return pl.Trainer(default_root_dir="log", callbacks=[], logger=[])
