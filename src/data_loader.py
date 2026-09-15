import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_transforms(img_size=224):
    train_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_t, val_t

def make_loaders(data_dir, batch_size=32, val_split=0.2, img_size=224, num_workers=4):
    train_t, val_t = get_transforms(img_size)
    # Use ImageFolder that expects class subfolders
    dataset = datasets.ImageFolder(root=data_dir, transform=train_t)
    num_val = int(len(dataset) * val_split)
    num_train = len(dataset) - num_val
    train_ds, val_ds = random_split(dataset, [num_train, num_val])
    # fix transform for val subset to use val transforms
    val_ds.dataset.transform = val_t

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    class_to_idx = dataset.class_to_idx
    return train_loader, val_loader, class_to_idx
