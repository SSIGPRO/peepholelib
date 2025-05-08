from torchvision import transforms

vgg16_cifar100 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
])

vgg16_cifar10 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
])

vgg16_cifar100_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    vgg16_cifar100
    ])

vgg16_cifar10_augumentations = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), 
    vgg16_cifar10
    ])

vgg16_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

vgg16_imagenet_augmentations = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET), 
    vgg16_imagenet
    ])

mobile_netv2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 