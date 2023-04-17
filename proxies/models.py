import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class ProxyMLP(nn.Module):
    def __init__(self, in_feat, hidden_layers, cat=True):
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
        self.final_act = nn.Tanh()
        self.cat = cat

    def forward(self, x):
        if self.cat:
            x = torch.cat(x, dim=-1)
        for l, layer in enumerate(self.nn_layers):
            x = layer(x)
            if l == len(self.nn_layers) - 1:
                x = self.final_act(x)
            if l % 2 == 1:
                x = self.hidden_act(x)
                x = self.drop(x)

        return x


class ProxyEmbeddingModel(nn.Module):
    def __init__(
        self, comp_emb_layers, sg_emb_size, lattice_emb_layers, prediction_layers
    ):
        super().__init__()
        self.comp_emb_mlp = mlp_from_layers(comp_emb_layers)
        self.sg_emb = nn.Embedding(230, sg_emb_size)
        self.lattice_emb_mlp = mlp_from_layers(lattice_emb_layers)
        self.pred_inp_size = comp_emb_layers[-1] + sg_emb_size + lattice_emb_layers[-1]
        self.prediction_head = ProxyMLP(self.pred_inp_size, prediction_layers, False)

    def forward(self, x):
        comp_x = self.comp_emb_mlp(x[0])
        sg_x = x[1].long()
        sg_x = self.sg_emb(sg_x).squeeze(1)

        lattice_x = self.lattice_emb_mlp(x[2])

        x = torch.cat((comp_x, sg_x, lattice_x), dim=-1)
        return self.prediction_head(x)


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
        out = self.model(x).squeeze(-1)
        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x).squeeze(-1)
        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        return optimizer


def mlp_from_layers(layers, act=None, norm=True):
    nn_layers = []
    for i in range(len(layers)):
        try:
            nn_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if norm:
                nn_layers.append(nn.BatchNorm1d(layers[i + 1]))
            nn_layers.append(nn.LeakyReLU(True) if act is None else act)
        except IndexError:
            pass
    nn_layers = nn.Sequential(*nn_layers)
    return nn_layers
