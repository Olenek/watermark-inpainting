import torch

from lib.dataset.dataset import get_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using: ', device)

train_loader, val_loader = get_dataloaders(data_root='../data/27kpng', batch_size=16)

from lib.training.trainer import Trainer
from lib.models.rirci import RIRCIModel

trainer = Trainer(RIRCIModel, epochs=100, train_loader=train_loader, val_loader=val_loader, device=device)

trainer.train_model(dry_run=True)
