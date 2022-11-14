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
    def __init__(self, proxy, loss, lr, device):
        super().__init__()
        self.model = proxy
        self.criterion = loss
        self.lr = lr
        self.loss = 0
        # self.device = device

    def training_step(self, batch, batch_idx):
        x, y = batch
        inp = x.to(torch.float32) # .to(self.device)
        true = y.to(torch.float32) # .to(self.device)
        out = self.model(inp).squeeze(-1).squeeze(-1)

        # l1_weights = torch.Tensor([
        #     m.weight.data.abs().sum()
        #     for m in model.modules
        # ]).sum()
        # l1_biases = torch.Tensor([
        #     m.bias.data.abs().sum()
        #     for m in model.modules
        # ]).sum()
        loss = self.criterion(out, true)
        # self.loss += loss.item()
        # if (batch_idx + 1) % 100 == 0:
        self.log('train_loss', loss)
        #     self.loss = 0
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        inp = x.to(torch.float32).to(self.device)
        true = y.to(torch.float32).to(self.device)
        out = self.model(inp).squeeze(-1).squeeze(-1)
        loss = self.criterion(out, true)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        inp = x.to(torch.float32).to(self.device)
        true = y.to(torch.float32).to(self.device)
        out = self.model(inp).squeeze(-1).squeeze(-1)
        loss = self.criterion(out, true)
        
        return loss
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        return optimizer
