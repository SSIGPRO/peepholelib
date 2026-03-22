import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


#-----------------------------
# Parsed for transforms
#-----------------------------

class TransformWrap():
    def __init__(self, **kwargs):
        self.transform = kwargs['transform']
        self.input_key = kwargs.get('input_key', 'image')
        return

    def __call__(self, x):
        x[self.input_key] = self.transform(x[self.input_key]) 
        return x

#-----------------------------
# Means and Stds for normalization
#-----------------------------

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
# VGG16 Augmentations
#-----------------------------
vgg16_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

vgg16_cifar_augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
])

vgg16_imagenet_augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    ])

vgg16_svhn_augmentations = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN), 
])

#-----------------------------
# ResNet50 Augmentations
#-----------------------------

resnet50_transform = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
])

resnet50_cifar100_augmentations = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
])

resnet50_imagenet_augmentations = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET), 
])

#-----------------------------
# WRN-28-10 on CIFAR100
#-----------------------------

wrn_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
])

#-----------------------------
# WRN-28-10 Augmentations
#-----------------------------

wrn_cifar100_augmentations = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

#-----------------------------
# ViT Augmentations
#-----------------------------

vit_b_16_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
])

vit_b_16_cifar100_augmentations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
])

vit_b_16_imagenet_augmentations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET), 
    ])

vit_b_16_svhn_augumentations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN), 
])

#-----------------------------
# Swin Augmentations
#-----------------------------

swin_b_transform = transforms.Compose([
    transforms.Resize(272, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
])

swin_b_cifar100_augmentations = transforms.Compose([
    transforms.Resize(272, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
])

swin_b_imagenet_augmentations = transforms.Compose([
    transforms.Resize(272, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
])

#-----------------------------
# Mobilenet on CIFAR 100
#-----------------------------

mobilenet_v2_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    ]) 

mobilenet_v2_cifar10_augumentations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    ])

#-----------------------------
# WRN-28-10 on CIFAR100
#-----------------------------

wrn_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
])

#-----------------------------
# WRN-28-10 Augmentations
#-----------------------------

wrn_cifar100_augmentations = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])