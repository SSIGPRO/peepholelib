import torch
from torchvision import transforms

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
