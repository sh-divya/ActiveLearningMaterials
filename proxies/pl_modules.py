import pytorch_lightning as pl
import torch.optim as optim
import time
from torchmetrics import MeanAbsoluteError, MeanSquaredError


class ProxyModule(pl.LightningModule):
    def __init__(self, proxy, loss, config):
        super().__init__()
        self.model = proxy
        self.criterion = loss
        self.lr = config["optim"]["lr"]
        self.loss = 0
        self.config = config
        self.scheduler = config["optim"]["scheduler"]["gamma"]
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.best_mae = 10e6
        self.best_mse = 10e6
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x).squeeze(-1)
        loss = self.criterion(out, y)
        mae = self.mae(out, y)
        mse = self.mse(out, y)

        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_mse", mse)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x).squeeze(-1)    
        loss = self.criterion(out, y)
        mae = self.mae(out, y)
        mse = self.mse(out, y)

        self.log("val_loss", loss)
        self.log("val_mae", mae)
        self.log("val_mse", mse)
        return loss

    def on_validation_epoch_end(self):
        epoch_val_mae = self.mae.compute()
        epoch_val_mse = self.mse.compute()
        if epoch_val_mae < self.best_mae:
            self.best_mae = epoch_val_mae
        if epoch_val_mse < self.best_mse:
            self.best_mse = epoch_val_mse

    def on_validation_end(self) -> None:
        self.logger.experiment.summary["Overall MAE"] = self.best_mae
        self.logger.experiment.summary["Overall MSE"] = self.best_mse

    def test_step(self, batch, batch_idx):
        x, _ = batch
        s = time.time()
        _ = self.model(x).squeeze(-1)
        sample_inf_time = (time.time() - s) / batch[0][0].shape[0]
        
        self.log("sample_inf_time", sample_inf_time, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        if self.config["optim"]["scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        elif self.config["optim"]["scheduler"]["name"] == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config["optim"]["scheduler"]["gamma"], gamma=self.config["optim"]["scheduler"]["gamma"])
        else: 
            scheduler = None
            return optimizer
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'val_mae'}}
