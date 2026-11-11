from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
    """Simple feature normalizer with safe handling for zero variance."""
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardNormalizer":
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer is not fitted.")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer is not fitted.")
        return x * self.std_ + self.mean_

    def state_dict(self) -> Dict:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.mean_ = None if state["mean"] is None else np.array(state["mean"], dtype=np.float32)
        self.std_ = None if state["std"] is None else np.array(state["std"], dtype=np.float32)


class ArrayDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        c: np.ndarray,
        r: Optional[np.ndarray] = None,
    ) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)
        self.r = None if r is None else torch.tensor(r, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.r is None:
            return self.x[idx], self.c[idx]
        return self.x[idx], self.c[idx], self.r[idx]


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation, nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalCVAE(nn.Module):
    """
    CVAE with:
      - encoder q(z|x,c)
      - conditional prior p(z|c)
      - decoder p(x|z,c)

    The decoder outputs the mean of a Gaussian likelihood with fixed variance.
    """
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        z_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.encoder = MLP(x_dim + c_dim, hidden_dims, 2 * z_dim, dropout=dropout)
        self.prior_net = MLP(c_dim, hidden_dims, 2 * z_dim, dropout=dropout)
        self.decoder = MLP(z_dim + c_dim, hidden_dims, x_dim, dropout=dropout)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, c], dim=-1))
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def prior(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.prior_net(c)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        q_mu, q_logvar = self.encode(x, c)
        p_mu, p_logvar = self.prior(c)
        z = self.reparameterize(q_mu, q_logvar)
        x_hat = self.decode(z, c)
        return x_hat, q_mu, q_logvar, p_mu, p_logvar

    @staticmethod
    def kl_gaussian(q_mu, q_logvar, p_mu, p_logvar) -> torch.Tensor:
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * (
            p_logvar - q_logvar
            + (q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-8)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    def loss(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        beta: float = 1.0,
        recon_weight: float = 1.0,
        phy_penalty: Optional[torch.Tensor] = None,
        phy_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_hat, q_mu, q_logvar, p_mu, p_logvar = self.forward(x, c)
        recon = ((x - x_hat) ** 2).mean()
        kl = self.kl_gaussian(q_mu, q_logvar, p_mu, p_logvar)
        total = recon_weight * recon + beta * kl
        if phy_penalty is not None and phy_weight > 0:
            total = total + phy_weight * phy_penalty.mean()
        logs = {
            "loss": float(total.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
        }
        if phy_penalty is not None:
            logs["phy"] = float(phy_penalty.mean().detach().cpu())
        return total, logs

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n: int = 1) -> torch.Tensor:
        if c.ndim == 1:
            c = c.unsqueeze(0)
        if c.shape[0] == 1 and n > 1:
            c = c.repeat(n, 1)
        p_mu, p_logvar = self.prior(c)
        z = self.reparameterize(p_mu, p_logvar)
        return self.decode(z, c)


class DiffusionNoiseScheduler:
    def __init__(self, timesteps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> "DiffusionNoiseScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


class ConditionalDenoiser(nn.Module):
    """
    Conditional diffusion refiner:
    predicts noise for x_t using [x_t, t_embed, cond, base_x, risk].
    """
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        r_dim: int,
        hidden_dims: Optional[List[int]] = None,
        t_embed_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.r_dim = r_dim
        self.t_embed = nn.Embedding(512, t_embed_dim)
        in_dim = x_dim + t_embed_dim + c_dim + x_dim + r_dim
        self.model = MLP(in_dim, hidden_dims, x_dim, dropout=dropout)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        base_x: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.t_embed(t.long())
        return self.model(torch.cat([x_t, t_emb, c, base_x, r], dim=-1))


def q_sample(
    scheduler: DiffusionNoiseScheduler,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar = scheduler.alpha_bars[t].unsqueeze(-1)
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
    return xt, noise


def diffusion_loss(
    denoiser: ConditionalDenoiser,
    scheduler: DiffusionNoiseScheduler,
    x0: torch.Tensor,
    c: torch.Tensor,
    base_x: torch.Tensor,
    r: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size = x0.shape[0]
    device = x0.device
    t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)
    xt, noise = q_sample(scheduler, x0, t)
    pred_noise = denoiser(xt, t, c, base_x, r)
    loss = ((noise - pred_noise) ** 2).mean()
    return loss, {"loss": float(loss.detach().cpu())}


@torch.no_grad()
def sample_with_diffusion(
    denoiser: ConditionalDenoiser,
    scheduler: DiffusionNoiseScheduler,
    base_x: torch.Tensor,
    c: torch.Tensor,
    r: torch.Tensor,
    tail_strength: float = 0.5,
) -> torch.Tensor:
    """
    Denoising refinement:
    start from a noisy perturbation around the CVAE baseline sample.
    tail_strength controls how strongly the final sample deviates toward stressed tails.
    """
    device = base_x.device
    x = base_x + tail_strength * torch.randn_like(base_x)
    for step in reversed(range(scheduler.timesteps)):
        t = torch.full((x.shape[0],), step, device=device, dtype=torch.long)
        beta_t = scheduler.betas[step]
        alpha_t = scheduler.alphas[step]
        alpha_bar_t = scheduler.alpha_bars[step]

        pred_noise = denoiser(x, t, c, base_x, r)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
        mean = coef1 * (x - coef2 * pred_noise)

        if step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean
    return x


@dataclass
class TrainingConfig:
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 50
    beta: float = 1.0
    recon_weight: float = 1.0
    phy_weight: float = 0.0
    diffusion_epochs: int = 30
    device: str = "cpu"
    seed: int = 42


def fit_cvae_model(
    model: ConditionalCVAE,
    x: np.ndarray,
    c: np.ndarray,
    train_cfg: TrainingConfig,
    phy_penalty_fn=None,
) -> List[Dict[str, float]]:
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    model.to(device)
    loader = DataLoader(
        ArrayDataset(x, c),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    history: List[Dict[str, float]] = []

    for epoch in range(train_cfg.epochs):
        model.train()
        losses = []
        for xb, cb in loader:
            xb = xb.to(device)
            cb = cb.to(device)
            phy_penalty = None
            if phy_penalty_fn is not None:
                phy_penalty = phy_penalty_fn(xb)
            loss, logs = model.loss(
                xb,
                cb,
                beta=train_cfg.beta,
                recon_weight=train_cfg.recon_weight,
                phy_penalty=phy_penalty,
                phy_weight=train_cfg.phy_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(logs)
        agg = {k: float(np.mean([d[k] for d in losses])) for k in losses[0].keys()}
        agg["epoch"] = epoch + 1
        history.append(agg)
    return history


def fit_diffusion_model(
    denoiser: ConditionalDenoiser,
    scheduler: DiffusionNoiseScheduler,
    x_true: np.ndarray,
    base_x: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    train_cfg: TrainingConfig,
) -> List[Dict[str, float]]:
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    denoiser.to(device)
    scheduler.to(device)
    ds = ArrayDataset(x_true, c, r)
    loader = DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=False)
    base_tensor = torch.tensor(base_x, dtype=torch.float32)
    opt = torch.optim.Adam(denoiser.parameters(), lr=train_cfg.lr)
    history: List[Dict[str, float]] = []

    for epoch in range(train_cfg.diffusion_epochs):
        denoiser.train()
        epoch_logs = []
        start_idx = 0
        for batch_idx, batch in enumerate(loader):
            xb, cb, rb = batch
            bs = xb.shape[0]
            bb = base_tensor[start_idx : start_idx + bs]
            start_idx += bs

            xb = xb.to(device)
            cb = cb.to(device)
            rb = rb.to(device)
            bb = bb.to(device)

            loss, logs = diffusion_loss(denoiser, scheduler, xb, cb, bb, rb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_logs.append(logs)

        agg = {k: float(np.mean([d[k] for d in epoch_logs])) for k in epoch_logs[0].keys()}
        agg["epoch"] = epoch + 1
        history.append(agg)
    return history


def save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_2d_array(df_or_array) -> np.ndarray:
    if isinstance(df_or_array, pd.DataFrame):
        return df_or_array.values.astype(np.float32)
    arr = np.asarray(df_or_array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}")
    return arr


def dataframe_or_none(arr: np.ndarray, columns: Optional[List[str]]) -> pd.DataFrame:
    if columns is None:
        columns = [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=columns)
