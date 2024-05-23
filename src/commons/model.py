import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule
from torchmetrics import MeanAbsoluteError


class PretrainedEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=1)

    def forward(self, x):
        x = self.model(x)
        return x


class Model(LightningModule):
    def __init__(self, learning_rate=None):
        super().__init__()

        self.learning_rate = learning_rate
        self.net = PretrainedEfficientNet()

        self.mse_loss = torch.nn.MSELoss()
        self.test_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

    def forward(self, input: torch.Tensor):
        return self.net(input)

    def predict_step(self, batch):
        return self.__model_step(batch)

    def __model_step(self, batch):
        inputs, targets = batch
        predictions = self.forward(inputs).clip(0, 1) * 80
        predictions = predictions.clip(1, 80)
        return predictions, targets

    def training_step(self, batch, _):
        predictions, targets = self.__model_step(batch)
        loss = self.mse_loss(predictions, targets)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        self.val_mae.reset()

    def validation_step(self, batch):
        predictions, targets = self.__model_step(batch)
        self.val_mae.update(predictions, targets)

    def on_validation_epoch_end(self):
        result = self.val_mae.compute()
        self.log("val/mae", result)
        print("val_mae", result)

    def on_test_epoch_start(self):
        self.test_mae.reset()

    def test_step(self, batch):
        predictions, targets = self.__model_step(batch)
        self.test_mae.update(predictions, targets)

    def on_test_epoch_end(self):
        result = self.test_mae.compute()
        self.log("test/mae", result)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
