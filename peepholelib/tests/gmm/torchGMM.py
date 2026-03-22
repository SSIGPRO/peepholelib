import torch
from cuda_selector import auto_cuda
import torch.distributions as D
from torchgmm.bayes import GaussianMixture as tGMM
from tqdm import tqdm

import logging
for name in logging.root.manager.loggerDict:
    if 'lightning' in name or 'pytorch' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

# --- config ---
nl_model   = 5      # number of classifiers (one per model class)
nl_class   = 3       # GMM components per classifier
n_features = 20
N          = 100   # total samples for timing
batch_size = 32
use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

torch.manual_seed(42)

# --- fit one tGMM per "model class" ---
classifiers = []
for c in tqdm(range(nl_model)):
    fit_data = torch.randn(400, n_features)  # slightly different distribution per class
    gmm = tGMM(
        num_components=nl_class,
        trainer_params=dict(
            num_nodes=1,
            max_epochs=50,
            accelerator=device.type,
            devices=[device.index],
            enable_progress_bar=False
        )
    )
    converged = False
    while not converged:
        gmm.fit(fit_data.clone())
        converged = not gmm.predict_proba(fit_data[:1]).isnan().any()
    classifiers.append(gmm)

# --- dataset and dataloader ---
dataset = torch.utils.data.TensorDataset(torch.randn(N, n_features))
loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

def sync():
    if use_cuda: torch.cuda.synchronize()

# --- build distribution once (new path) ---
weights = torch.stack([classifiers[c].model_.component_probs for c in range(nl_model)]).to(device)
means = torch.stack([classifiers[c].model_.means for c in range(nl_model)]).to(device)
prec_chol = torch.stack([classifiers[c].model_.precisions_cholesky for c in range(nl_model)]).to(device)
mix  = D.Categorical(probs=weights)
comp = D.Independent(D.Normal(means, 1.0 / prec_chol), 1)
dist = D.MixtureSameFamily(mix, comp)

# --- correctness check on one batch ---
probe = torch.randn(batch_size, n_features).to(device)
old_probe = torch.stack(
    [classifiers[c].score_samples(probe).squeeze(0) for c in range(nl_model)],
    dim=1
)

new_probe = -dist.log_prob(probe.unsqueeze(1))
print("probe device:", probe.device)
print("new_probe device:", new_probe.device)
print("is cuda:", new_probe.is_cuda)

assert probe.is_cuda
assert new_probe.is_cuda

quit()
print(f"max abs diff: {(old_probe - new_probe.cpu()).abs().max().item():.6e}")
torch.testing.assert_close(old_probe, new_probe.cpu(), atol=1e-4, rtol=1e-4)
print("PASSED: outputs match\n")

# --- time old path ---
import time
sync()
t0 = time.perf_counter()
for (batch,) in tqdm(loader):
    batch = batch.to(device)
    _ = torch.stack(
        [classifiers[c].score_samples(batch).squeeze(0) for c in range(nl_model)],
        dim=1
    )
sync()
t_old = time.perf_counter() - t0
print(f"old (score_samples): {t_old:.3f}s")

# --- time new path ---
sync()
t0 = time.perf_counter()
for (batch,) in tqdm(loader):
    batch = batch.to(device)
    _ = dist.log_prob(batch.unsqueeze(1))
sync()
t_new = time.perf_counter() - t0
print(f"new (MixtureSameFamily): {t_new:.3f}s")
print(f"speedup: {t_old / t_new:.1f}x")

