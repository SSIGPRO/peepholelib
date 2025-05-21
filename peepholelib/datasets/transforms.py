import torch
from torchvision import transforms

means = {
        'cifar10': torch.tensor([0.424, 0.415, 0.384]),
        'cifar100': torch.tensor([0.438, 0.418, 0.377]),
        'imagenet': torch.tensor([0.485, 0.456, 0.406]),
        }

stds = {
        'cifar10': torch.tensor([0.283, 0.278, 0.284]),
        'cifar100': torch.tensor([0.300, 0.287, 0.294]),
        'imagenet': torch.tensor([0.283, 0.278, 0.284]),
        }

#-----------------------------
# VGG16 on CIFAR100 
#-----------------------------

vgg16_cifar100 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=means['cifar100'], std=stds['cifar100'])
])

vgg16_cifar100_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    vgg16_cifar100
])

#-----------------------------
# VGG16 on CIFAR10 
#-----------------------------

vgg16_cifar10 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=means['cifar10'], std=stds['cifar10'])
    ])

vgg16_cifar10_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    vgg16_cifar10
    ])

#-----------------------------
# VGG16 on Imagenet 
#-----------------------------

vgg16_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=means['imagenet'], std=stds['imagenet'])
])

vgg16_imagenet_augmentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET), 
    vgg16_imagenet
    ])

#-----------------------------
# Mobilenet on CIFAR 100
#-----------------------------

mobilenet_v2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']),
    ]) 

mobilenet_v2_cifar10_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    mobilenet_v2
    ])
