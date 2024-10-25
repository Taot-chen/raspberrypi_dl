import torch
import torchvision
from torchvision import transforms
 
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    random_grayscale_p = 0.2
    transform_train = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomGrayscale(p=random_grayscale_p),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transform_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    class_names = train_dataset.classes
    return train_loader, val_loader, class_names
