import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torch_simple_timing import Timer


class ProxyModule(pl.LightningModule):
    def __init__(self, proxy, loss, config, gpu):
        super().__init__()
        self.model = proxy
        self.criterion = loss
        self.lr = config["optim"]["lr"]
        self.loss = 0
        self.config = config
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.best_mae = 10e6
        self.best_mse = 10e6
        self.save_hyperparameters(config)
        self.active_logger = config.get("debug") is None
        self.timer = Timer(gpu=gpu)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x).squeeze(-1)
        loss = self.criterion(out, y)
        mae = self.mae(out, y)
        mse = self.mse(out, y)
        lr = self.optimizers().param_groups[0]['lr']

        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_mse", mse)
        self.log("learning_rate", lr)

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
        self.mae.reset()
        self.mse.reset()

    def on_validation_end(self) -> None:
        if self.active_logger: 
            self.logger.experiment.summary["Best MAE"] = self.best_mae
            self.logger.experiment.summary["Best MSE"] = self.best_mse
        else: 
            print("Best MAE: ", self.best_mae)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        with self.timer.clock("forward-pass"):
            self.model(x).squeeze(-1)

    def test_epoch_end(self, outputs):
        stats = self.timer.stats()
        print(self.timer.display(stats=stats, precision=4))
        # self.logger.experiment.summary.update(stats)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        if self.config["optim"]["scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.config["optim"]["scheduler"]["decay_factor"],
                patience=self.config["optim"]["es_patience"]
            )
        elif self.config["optim"]["scheduler"]["name"] == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config["optim"]["scheduler"]["step_size"],
                gamma=self.config["optim"]["scheduler"]["decay_factor"],
            )
        else:
            scheduler = None
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_mae",
            },
        }
