import torch

a = 5
b = 4
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

orthogonal_m2, _ = torch.linalg.qr(gaussian_m[:, :-1], mode='reduced')
orthogonal_m2 = orthogonal_m2.T.contiguous()

print((orthogonal_m2 == orthogonal_m[:-1,:]).sum())
print(orthogonal_m2.shape[0]*orthogonal_m2.shape[1])

dist = torch.linalg.norm(orthogonal_m[:-1, :] - orthogonal_m2, ord='fro')
print(f"Frobenius distance: {dist.item()}")
