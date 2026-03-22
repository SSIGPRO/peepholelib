import torch

a = 1000
b = 5000
seed = 42

generator = None
generator = torch.Generator(device='cpu')
generator.manual_seed(seed)

gaussian_m = torch.randn(a, b, generator=generator)
orthogonal_m, _ = torch.linalg.qr(gaussian_m, mode='reduced')
orthogonal_m = orthogonal_m.T.contiguous()
I = torch.eye(orthogonal_m.shape[0], device=orthogonal_m.device)
check = orthogonal_m @ orthogonal_m.T
print(torch.allclose(check, I, atol=1e-6))

R = torch.randn(a, b, generator=generator) / (b ** 0.5)
I = torch.eye(R.shape[0], device=R.device)
check = R @ R.T
print(torch.allclose(check, I, atol=1e-6))