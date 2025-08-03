import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import (
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    DATA_DIR
)

def get_transforms(split):
    if split == 'train':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
        ])

def load_split(split, data_dir):
    path = os.path.join(data_dir, split)
    dataset = datasets.ImageFolder(path, transform=get_transforms(split))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == 'train'),
        num_workers=NUM_WORKERS
    )
    print(f"{split.capitalize()} set: {len(dataset)} samples, classes: {dataset.classes}")
    return loader

def get_dataloaders(data_dir=DATA_DIR):
    train_loader = load_split('train', data_dir)
    val_loader = load_split('valid', data_dir)
    test_loader = load_split('test', data_dir)
    return train_loader, val_loader, test_loader

def test_dataloaders():
    train_loader, val_loader, test_loader = get_dataloaders()

    names = ['Train', 'Validation', 'Test']
    loaders = [train_loader, val_loader, test_loader]

    for name, loader in zip(names, loaders):
        print(f" {name} Loader:")
        for images, labels in loader:
            print(f"Images shape: {images.shape}")
            print(f"Labels: {labels.tolist()}\n")
            break  

