import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class WatermarkRemovalDataset(Dataset):
    def __init__(self, root_dir: str, phase: str = 'train'):
        """
        Args:
            root_dir (string): Path to the dataset directory (data/dataset)
            phase (string): 'train' or 'val'
        """
        self.root_dir = os.path.join(root_dir, f'{phase}_images')
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.wm_transform = transforms.Compose([  # applied to mask and watermark
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.image_files = sorted(os.listdir(os.path.join(self.root_dir, 'image')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir, 'image', self.image_files[idx])
        wm_path = os.path.join(self.root_dir, 'wm', self.image_files[idx])
        mask_path = os.path.join(self.root_dir, 'mask', self.image_files[idx])
        target_path = os.path.join(self.root_dir, 'target', self.image_files[idx])

        image = self.transform(Image.open(image_path).convert('RGB'))
        wm = self.wm_transform(Image.open(wm_path).getchannel('A'))
        mask = self.wm_transform(Image.open(mask_path).convert('L'))
        target = self.transform(Image.open(target_path).convert('RGB'))


        return {
            'image': image,
            'wm': wm,
            'mask': mask,
            'target': target
        }


def get_dataloaders(data_root='data/dataset', batch_size=8, num_workers=4):
    train_dataset = WatermarkRemovalDataset(
        root_dir=data_root,
        phase='train',
    )

    val_dataset = WatermarkRemovalDataset(
        root_dir=data_root,
        phase='val',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader