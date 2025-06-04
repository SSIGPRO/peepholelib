import torch

class foo():
    def __init__(self):
        self.a = torch.rand(5, 2)
    
    def __getitem__(self, idx):
        return self.a[idx,:]

if __name__ == '__main__':
    a = foo()
    print(a.a)
    print(a[2])
