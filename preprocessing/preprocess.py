import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_dataloaders(data_dir):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageFolder(
        root=f"{data_dir}/train",
        transform=transform
    )

    val_dataset = ImageFolder(
        root=f"{data_dir}/val",
        transform=transform
    )

    test_dataset = ImageFolder(
        root=f"{data_dir}/test",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16
    )

    return train_loader, val_loader, test_loader