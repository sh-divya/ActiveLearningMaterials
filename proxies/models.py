import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class ProxyMLP(nn.Module):
    def __init__(self, in_feat, hidden_layers):
        super(ProxyMLP, self).__init__()
        self.nn_layers = []
        self.modules = []

        for i in range(len(hidden_layers)):
            if i == 0:
                self.nn_layers.append(nn.Linear(in_feat, hidden_layers[i]))
            else:
                self.nn_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.modules.append(self.nn_layers[-1])
            self.nn_layers.append(nn.BatchNorm1d(hidden_layers[i]))
        self.nn_layers.append(nn.Linear(hidden_layers[-1], 1))
        self.modules.append(self.nn_layers[-1])
        self.nn_layers = nn.ModuleList(self.nn_layers)
        self.hidden_act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(p=0.5)
        # self.hidden_act = nn.ReLU()
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        for l, layer in enumerate(self.nn_layers):
            x = layer(x)
            # print(layer)
            if l == len(self.nn_layers) - 1:
                x = self.final_act(x)
            if l % 2 == 1:
                # print(self.hidden_act)
                # print(self.drop)
                x = self.hidden_act(x)
                x = self.drop(x)

        return x


class ProxyModel(pl.LightningModule):
    def __init__(self, proxy, loss, acc, lr):
        super().__init__()
        self.model = proxy
        self.criterion = loss
        self.accuracy = acc
        self.lr = lr
        self.loss = 0

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
        optimizer = optim.Adam(self.parameters(), self.lr)
        return optimizer
