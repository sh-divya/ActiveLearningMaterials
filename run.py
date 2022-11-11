import torch
import random
import torch.nn as nn
import torch.optim as optim
from crystal_data import CrystalDataset
from models import ProxyMLP
from torch.utils.data import DataLoader, random_split
import wandb as wb
import numpy as np
import os.path as osp
from torch.nn.init import xavier_uniform_

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def weights_init(m):
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)


def validate(model, loader, criterion, device):
    model.eval()
    with torch.no_grad():
        vl = 0
        for i, (x, y) in enumerate(loader):
            inp = x.to(device).to(torch.float32)
            true = y.to(device).to(torch.float32)
            out = model(inp).squeeze(-1).squeeze(-1)
            vl += criterion(out, true).item()
        vl = vl / (i + 1)

    return vl


def train(batch_size, lr, num_epochs, layers, lmda, split):
    device = torch.device("cpu")

    standard = {
        'mean':torch.load('./data/mean.pt'),
        'std':torch.load('./data/std.pt')
    }

    dataset = CrystalDataset('./data/li-ssb', './data/lissb.csv', True,
                             './data/skip.txt', './data/proxy.csv', 
                             './data/compile.csv', transform=standard)
    sets = random_split(dataset, split)
    trainset, valset, testset = sets

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    model = ProxyMLP(322, layers).to(device)
    model.apply(weights_init)
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=lmda)

    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for i, (x, y) in enumerate(trainloader):
            inp = x.to(device).to(torch.float32)
            true = y.to(device).to(torch.float32)
            out = model(inp).squeeze(-1).squeeze(-1)
            l1_weights = torch.Tensor([
                m.weight.data.abs().sum()
                for m in model.modules
            ]).sum()
            l1_biases = torch.Tensor([
                m.bias.data.abs().sum()
                for m in model.modules
            ]).sum()
            l = criterion(out, true) + lmda * (l1_weights + l1_biases)
            loss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # raise Exception
            if (i + 1) % 10 == 0:
                wb.log(
                    {
                        'TLoss': loss / 100
                    }
                )
                loss = 0
        vl = validate(model, valloader, criterion, device)
        wb.log({'VLoss': vl})
        
# random                
# similar distribution on y across the sets
# remove some elements

if __name__ == '__main__':
    wb.init(project='AL-Li', entity='sh-divya')
    config = {
        'lr': 1e-4,
        'batch': 64,
        'epochs': 5,
        'layers': [256, 256, 128],
        'split': [0.6, 0.2, 0.2],
        'lambda': 0.5
    }
    name = [key + '-' + str(config[key]) for key in ['lr', 'batch', 'lambda']]
    wb.run.name = 'Reduced' + '_'.join(name) + '_layers' # + '-'.join(config['layers'])
    wb.run.name = 'test'
    train(
        config['batch'],
        config['lr'], config['epochs'],
        config['layers'], config['lambda'],
        config['split']
    )