import pytorch_lightning as pl
import torch


class ProxyModule(pl.LightningModule):
    def __init__(self, proxy, loss, acc, config):
        super().__init__()
        self.model = proxy
        self.criterion = loss
        self.accuracy = acc
        self.lr = config["optim"]["lr"]
        self.loss = 0
        self.config = config
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch
        inp = x.to(torch.float32)
        true = y.to(torch.float32)
        out = self.model(inp).squeeze(-1)
        loss = self.criterion(out, true)
        acc = self.accuracy(out, true)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        inp = x.to(torch.float32)
        true = y.to(torch.float32)
        out = self.model(inp).squeeze(-1)
        loss = self.criterion(out, true)
        acc = self.accuracy(out, true)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        inp = x.to(torch.float32)
        true = y.to(torch.float32)
        out = self.model(inp)
        loss = self.criterion(out, true)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
