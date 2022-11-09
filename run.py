import torch
import torch.nn as nn
import torch.optim as optim
from crystal_data import CrystalDataset
from models import ProxyMLP
from torch.utils.data import DataLoader, random_split
import wandb as wb


def val():
    pass

def train(batch_size, lr, num_epochs):
    device = torch.device("cpu")
    norm = None
    dataset = CrystalDataset('./data/li-ssb', './data/lissb.csv',
                             './data/compile.csv', './data/skip.txt',
                             True)
    trainset, valset, testset = random_split(len(CrystalDataset), [0.6, 0.2, 0.2])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    model = ProxyMLP(126, [256, 512, 512, 128, 64])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(lr=1e-3)

    for epoch in range(num_epochs):
        loss = 0
        for i, (x, y) in enumerate(trainloader):
            out = model(x)
            l = criterion(out, y)
            loss += l.item()
            if (i + 1) % 100 == 0:
                loss = 0

# random                
# similar distribution on y across the sets
# remove some elements

if __name__ == '__main__':
    pass
