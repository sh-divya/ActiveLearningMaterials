import torch
import random
import torch.nn as nn
from crystal_data import CrystalDataset
from models import ProxyMLP, ProxyModel
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import  train_test_split
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

def proportional_split(dataset, split=None):
    idx = list(range(len(dataset)))
    y = dataset.get_all_y()
    x_train, x_test, y_train, y_test = train_test_split(idx, y,
                                                        train_size=split[0], test_size=split[1] + split[2],
                                                        shuffle=True, stratify=y)
    val_to_test = [split[1] / sum(split[1:]), split[2] / sum(split[1:])]
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        train_size=val_to_test[0], test_size=val_to_test[1],
                                                        shuffle=True, stratify=y_test)

    return Subset(dataset, x_train), Subset(dataset, x_val), Subset(dataset, x_test)

def train(batch_size, lr, num_epochs, layers, split, logger):
    device = torch.device("cpu")

    standard = {
        'mean':torch.load('./data/mean85.pt'),
        'std':torch.load('./data/std85.pt')
    }

    dataset = CrystalDataset('./data/li-ssb', './data/lissb.csv', True,
                             './data/skip.txt', './data/proxy.csv', 
                             './data/compile.csv' , transform=standard, subset=True)


    sets = proportional_split(dataset, split)
    trainset, valset, testset = sets

    # sets = random_split(dataset, split)
    # trainset, valset, testset = sets

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # raise Exception
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    model = ProxyMLP(85, layers).to(device).to(torch.float32)
    model.apply(weights_init)
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    # for x, y in trainloader:
    #     print(x.shape)
    #     print(y.shape)
    #     out = model(x.to(torch.float32))
    #     error = criterion(out, y.to(torch.float32))
    # raise Exception
    early = EarlyStopping(monitor='val_acc', patience=3, mode='max')

    model = ProxyModel(model, criterion, lr, device)
    trainer = pl.Trainer(max_epochs=num_epochs, logger=logger,
                        log_every_n_steps=1, callbacks=early, min_epochs=20)
    trainer.fit(model=model, train_dataloaders=trainloader,
                val_dataloaders=valloader)
    # raise Exception
    logger.experiment.config['LR'] = lr
    logger.experiment.config['batch'] = batch_size
    logger.experiment.config['layers'] = layers
    logger.experiment.finish()


# random                
# similar distribution on y across the sets
# remove some elements

if __name__ == '__main__':

    layer_search = [
        # [512, 512],
        # [1024, 1024],
        # [512, 512, 256],
        # [1024, 1024, 512],
        # [512, 512, 512, 256],
        # [512, 1024, 1024, 512]
    ]

    layer_search = [
        [256, 256]
        # [128, 128],
        # [128, 256, 128],
        # [256, 256, 256],
        # [128, 256, 256, 128]
    ]

    for layer in layer_search:
        config = {
            'lr': 1e-3,
            'batch': 64,
            'epochs': 50,
            'layers': layer,
            'split': [0.6, 0.2, 0.2],
        }
        name = [key + '-' + str(config[key]) for key in ['lr', 'batch']]
        # name = 'test'
        name = 'MSE_CompOnly' + '_'.join(name) + '_layer-' + ''.join([str(l) for l in layer])
        logger = WandbLogger(project='AL-Li', name=name)
        # logger = None
        train(
            config['batch'],
            config['lr'], config['epochs'],
            config['layers'],
            config['split'], logger
        )