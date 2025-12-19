
import torch
import torch.utils
import torch.utils.data.distributed

import torchvision
from torchvision import datasets, transforms

def load_data(args):
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size,num_workers=4)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, shuffle=False,  batch_size=args.batch_size, num_workers=4)
    return train_loader, val_loader
 
