import torch
from torchvision import transforms

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
# VGG16 on CIFAR100 
#-----------------------------

vgg16_cifar100 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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
])

vgg16_imagenet_augmentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET), 
    vgg16_imagenet
    ])

#-----------------------------
# VGG16 on SVHN 
#-----------------------------

vgg16_svhn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

vgg16_svhn_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN), 
    vgg16_svhn
])

#-----------------------------
# ViT on CIFAR 100
#-----------------------------

vit_b_16_cifar100 = transforms.Compose([
    # uses interpolation=InterpolationMode.BILINEAR by default 
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

vit_b_16_cifar100_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    vit_b_16_cifar100
    ])

#-----------------------------
# ViT on ImageNet
#-----------------------------

vit_b_16_imagenet = transforms.Compose([
    # uses interpolation=InterpolationMode.BILINEAR by default 
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

vit_b_16_imagenet_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET), 
    vit_b_16_imagenet
    ])

#-----------------------------
# ViT on SVHN 
#-----------------------------

vit_b_16_svhn = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

vit_b_16_svhn_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN), 
    vit_b_16_svhn
])

#-----------------------------
# Mobilenet on CIFAR 100
#-----------------------------

mobilenet_v2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ]) 

mobilenet_v2_cifar10_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    mobilenet_v2
    ])
