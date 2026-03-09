import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

means = {
        'CIFAR10': torch.tensor([0.424, 0.415, 0.384]).view(1,3,1,1),
        'CIFAR100': torch.tensor([0.438, 0.418, 0.377]).view(1,3,1,1),
        'ImageNet': torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1),
        'SVHN': torch.tensor([0.438, 0.444, 0.473]).view(1,3,1,1)
        }

stds = {
        'CIFAR10': torch.tensor([0.283, 0.278, 0.284]).view(1,3,1,1),
        'CIFAR100': torch.tensor([0.300, 0.287, 0.294]).view(1,3,1,1),
        'ImageNet': torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1),
        'SVHN': torch.tensor([0.198, 0.201, 0.197]).view(1,3,1,1),
        }

#-----------------------------
# VGG16 
#-----------------------------

vgg16_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#-----------------------------
# VGG16 Augmentations
#-----------------------------

vgg16_cifar_augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
])

vgg16_imagenet_augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(), 
    ])

vgg16_svhn_augmentations = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN), 
    transforms.ToTensor(),
])

#-----------------------------
# ResNet50 on CIFAR100 
#-----------------------------

resnet50_transform = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

#-----------------------------
# ResNet50 Augmentations
#-----------------------------

resnet50_cifar100_augmentations = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    transforms.ToTensor(),
])

resnet50_imagenet_augmentations = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET), 
    transforms.ToTensor(),
])

#-----------------------------
# ViT
#-----------------------------

vit_b_16_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

#-----------------------------
# ViT Augmentations
#-----------------------------

vit_b_16_cifar100_augmentations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
])

vit_b_16_imagenet_augmentations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET), 
    transforms.ToTensor(),
    ])

vit_b_16_svhn_augumentations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN), 
    transforms.ToTensor(),
])

#-----------------------------
# Swin
#-----------------------------

swin_b_transform = transforms.Compose([
    transforms.Resize(272, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

#-----------------------------
# Swin Augmentations
#-----------------------------

swin_b_cifar100_augmentations = transforms.Compose([
    transforms.Resize(272, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
])

swin_b_imagenet_augmentations = transforms.Compose([
    transforms.Resize(272, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
])

#-----------------------------
# Mobilenet on CIFAR 100
#-----------------------------

mobilenet_v2_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ]) 

mobilenet_v2_cifar10_augumentations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    transforms.ToTensor(),
    ])
