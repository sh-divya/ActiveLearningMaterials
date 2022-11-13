import torch
import random
import torch.nn as nn
from crystal_data import CrystalDataset
from models import ProxyMLP, ProxyModel
from torch.utils.data import DataLoader, random_split
import wandb as wb
import numpy as np
import os.path as osp
from torch.nn.init import xavier_uniform_
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def weights_init(m):
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)


def train(batch_size, lr, num_epochs, layers, split, logger):
    device = torch.device("cpu")

    standard = {
        'mean':torch.load('./data/mean.pt'),
        'std':torch.load('./data/std.pt')
    }

    dataset = CrystalDataset('./data/li-ssb', './data/lissb.csv', True,
                             './data/skip.txt', './data/proxy.csv', 
                             './data/compile.csv' , transform=standard)
    sets = random_split(dataset, split)
    trainset, valset, testset = sets

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    model = ProxyMLP(322, layers).to(device).to(torch.float32)
    model.apply(weights_init)
    criterion = nn.BCELoss()

    model = ProxyModel(model, criterion, lr, device)
    trainer = pl.Trainer(max_epochs=num_epochs, logger=logger)
    trainer.fit(model=model, train_dataloaders=trainloader,
                val_dataloaders=valloader)


# random                
# similar distribution on y across the sets
# remove some elements

if __name__ == '__main__':
    # wb.init(project='AL-Li', entity='sh-divya')
    config = {
        'lr': 1e-2,
        'batch': 128,
        'epochs': 5,
        'layers': [512, 512, 256],
        'split': [0.6, 0.2, 0.2],
        'lambda': 0.2
    }
    name = [key + '-' + str(config[key]) for key in ['lr', 'batch']] #, 'lambda']]
    name = 'test'
    logger = WandbLogger(project='AL-Li', name=name)
    # logger = None
    train(
        config['batch'],
        config['lr'], config['epochs'],
        config['layers'],
        config['split'], logger
    )