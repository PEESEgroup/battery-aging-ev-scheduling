
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandardNormalizer:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardNormalizer":
        x = np.asarray(x, dtype=np.float32)
        self.mean_ = np.nanmean(x, axis=0).astype(np.float32)
        self.std_ = np.nanstd(x, axis=0).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer not fitted.")
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer not fitted.")
        x = np.asarray(x, dtype=np.float32)
        return (x * self.std_ + self.mean_).astype(np.float32)

    def state_dict(self) -> Dict:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.mean_ = None if state["mean"] is None else np.asarray(state["mean"], dtype=np.float32)
        self.std_ = None if state["std"] is None else np.asarray(state["std"], dtype=np.float32)


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, c: np.ndarray, r: Optional[np.ndarray] = None) -> None:
        self.x = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
        self.c = torch.tensor(np.asarray(c, dtype=np.float32), dtype=torch.float32)
        self.r = None if r is None else torch.tensor(np.asarray(r, dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        if self.r is None:
            return self.x[idx], self.c[idx]
        return self.x[idx], self.c[idx], self.r[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalCVAE(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, z_dim: int = 16, hidden_dims: Optional[List[int]] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.encoder = MLP(x_dim + c_dim, hidden_dims, 2 * z_dim, dropout=dropout)
        self.prior_net = MLP(c_dim, hidden_dims, 2 * z_dim, dropout=dropout)
        self.decoder = MLP(z_dim + c_dim, hidden_dims, x_dim, dropout=dropout)

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.encoder(torch.cat([x, c], dim=-1))
        return torch.chunk(h, 2, dim=-1)

    def prior(self, c: torch.Tensor):
        h = self.prior_net(c)
        return torch.chunk(h, 2, dim=-1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        q_mu, q_logvar = self.encode(x, c)
        p_mu, p_logvar = self.prior(c)
        z = self.reparameterize(q_mu, q_logvar)
        x_hat = self.decode(z, c)
        return x_hat, q_mu, q_logvar, p_mu, p_logvar

    @staticmethod
    def kl_gaussian(q_mu, q_logvar, p_mu, p_logvar):
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-8) - 1.0)
        return kl.sum(dim=-1).mean()

    def loss(self, x: torch.Tensor, c: torch.Tensor, beta: float = 1.0):
        x_hat, q_mu, q_logvar, p_mu, p_logvar = self.forward(x, c)
        recon = ((x - x_hat) ** 2).mean()
        kl = self.kl_gaussian(q_mu, q_logvar, p_mu, p_logvar)
        loss = recon + beta * kl
        return loss, {"loss": float(loss.detach().cpu()), "recon": float(recon.detach().cpu()), "kl": float(kl.detach().cpu())}

    @torch.no_grad()
    def sample(self, c: torch.Tensor) -> torch.Tensor:
        p_mu, p_logvar = self.prior(c)
        z = self.reparameterize(p_mu, p_logvar)
        return self.decode(z, c)


class DiffusionNoiseScheduler:
    def __init__(self, timesteps: int = 30, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


class ConditionalDenoiser(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, r_dim: int, hidden_dims: Optional[List[int]] = None, t_embed_dim: int = 16):
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.t_embed = nn.Embedding(512, t_embed_dim)
        self.model = MLP(x_dim + c_dim + r_dim + x_dim + t_embed_dim, hidden_dims, x_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor, base_x: torch.Tensor, r: torch.Tensor):
        te = self.t_embed(t.long())
        return self.model(torch.cat([x_t, te, c, base_x, r], dim=-1))


def q_sample(scheduler: DiffusionNoiseScheduler, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar = scheduler.alpha_bars[t].unsqueeze(-1)
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise, noise


def diffusion_loss(denoiser, scheduler, x0, c, base_x, r):
    batch_size = x0.shape[0]
    device = x0.device
    t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)
    x_t, noise = q_sample(scheduler, x0, t)
    pred_noise = denoiser(x_t, t, c, base_x, r)
    loss = ((noise - pred_noise) ** 2).mean()
    return loss, {"loss": float(loss.detach().cpu())}


@torch.no_grad()
def sample_with_diffusion(denoiser, scheduler, base_x, c, r, tail_strength: float = 0.5):
    x = base_x + tail_strength * torch.randn_like(base_x)
    for step in reversed(range(scheduler.timesteps)):
        t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
        beta_t = scheduler.betas[step]
        alpha_t = scheduler.alphas[step]
        alpha_bar_t = scheduler.alpha_bars[step]
        pred_noise = denoiser(x, t, c, base_x, r)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8) * pred_noise)
        if step > 0:
            x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
        else:
            x = mean
    return x


@dataclass
class TrainingConfig:
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 50
    diffusion_epochs: int = 30
    beta: float = 1.0
    seed: int = 42
    device: str = "cpu"


def fit_cvae_model(model, x: np.ndarray, c: np.ndarray, cfg: TrainingConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    loader = DataLoader(ArrayDataset(x, c), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    hist = []
    for epoch in range(cfg.epochs):
        logs = []
        for xb, cb in loader:
            xb = xb.to(device); cb = cb.to(device)
            loss, lg = model.loss(xb, cb, beta=cfg.beta)
            opt.zero_grad(); loss.backward(); opt.step()
            logs.append(lg)
        hist.append({k: float(np.mean([d[k] for d in logs])) for k in logs[0]})
        hist[-1]["epoch"] = epoch + 1
    return hist


def fit_diffusion_model(denoiser, scheduler, x_true, base_x, c, r, cfg: TrainingConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    denoiser.to(device)
    scheduler.to(device)
    ds = ArrayDataset(x_true, c, r)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    base_tensor = torch.tensor(np.asarray(base_x, dtype=np.float32), dtype=torch.float32)
    opt = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr)
    hist = []
    for epoch in range(cfg.diffusion_epochs):
        logs = []
        start_idx = 0
        for xb, cb, rb in loader:
            bs = xb.shape[0]
            bb = base_tensor[start_idx:start_idx+bs]
            start_idx += bs
            xb = xb.to(device); cb = cb.to(device); rb = rb.to(device); bb = bb.to(device)
            loss, lg = diffusion_loss(denoiser, scheduler, xb, cb, bb, rb)
            opt.zero_grad(); loss.backward(); opt.step()
            logs.append(lg)
        hist.append({k: float(np.mean([d[k] for d in logs])) for k in logs[0]})
        hist[-1]["epoch"] = epoch + 1
    return hist
