from torchvision.transforms import Compose
import torch

def foo(x):
    print('foo')
    return x

def bar(x):
    print('bar')
    return x

if __name__ == '__main__':
    t = Compose([
        foo,
        bar
        ])

    x = torch.rand(2,3,4)
    print(x)

    t(x)


