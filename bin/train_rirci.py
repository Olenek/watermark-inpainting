import torch

from lib.dataset.dataset import get_dataloaders
from lib.models.rirci import RIRCIModel
from lib.training.trainer import Trainer

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using: ', device)

train_loader, val_loader = get_dataloaders(data_root='../data/27kpng', batch_size=16)

trainer = Trainer(RIRCIModel, epochs=100, train_loader=train_loader, val_loader=val_loader, device=device)

trainer.train_model()
