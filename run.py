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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
        'mean':torch.load('./data/mean322.pt'),
        'std':torch.load('./data/std322.pt')
    }

    dataset = CrystalDataset('./data/li-ssb', './data/lissb.csv', True,
                             './data/skip.txt', './data/proxy.csv', 
                             './data/compile.csv' , transform=standard, subset=False)

    sets = random_split(dataset, split)
    trainset, valset, testset = sets

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    model = ProxyMLP(322, layers).to(device).to(torch.float32)
    model.apply(weights_init)
    criterion = nn.BCELoss()
    early = EarlyStopping(monitor='val_loss', patience=3, mode='min')

    model = ProxyModel(model, criterion, lr, device)
    trainer = pl.Trainer(max_epochs=num_epochs, logger=logger, log_every_n_steps=1, callbacks=early)
    trainer.fit(model=model, train_dataloaders=trainloader,
                val_dataloaders=valloader)
    logger.experiment.finish()


# random                
# similar distribution on y across the sets
# remove some elements

if __name__ == '__main__':
    # wb.init(project='AL-Li', entity='sh-divya')

    layer_search = [
        # [512, 512],
        # [1024, 1024],
        [512, 512, 256]
        # [1024, 1024, 512],
        # [512, 512, 512, 256],
        # [512, 1024, 1024, 512]
    ]

    # layer_search = [
        # [256, 256],
        # [128, 256, 128],
        # [256, 256, 256]
        # [128, 256, 256, 128]
    # ]

    for layer in layer_search:
        config = {
            'lr': 1e-4,
            'batch': 32,
            'epochs': 20,
            'layers': layer,
            'split': [0.6, 0.2, 0.2],
        }
        name = [key + '-' + str(config[key]) for key in ['lr', 'batch']]
        # name = 'test'
        name = 'AllFeat' + '_'.join(name) + '_layer-' + ''.join([str(l) for l in layer])
        logger = WandbLogger(project='AL-Li', name=name)
        # logger = None
        train(
            config['batch'],
            config['lr'], config['epochs'],
            config['layers'],
            config['split'], logger
        )