"""Small, train/validation separated models for a genuinely shared information set."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


def raw_features(windows):
    return np.column_stack([windows["past"], windows["L"]])


def lpv_features(windows):
    past, length = windows["past"], windows["L"]
    # Match recent-history information while improving numerical conditioning.
    state = np.column_stack([past[:, 0], (past[:, 0]-past[:, 1])/.01,
                             (past[:, 0]-2*past[:, 1]+past[:, 2])/.01**2])
    return np.column_stack([state, state*length[:, None], state/np.maximum(length[:, None], .01)])


class LinearForecaster:
    def __init__(self, alpha=0.0):
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, x, y):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)


class BoostedTrees:
    """Python tree baseline, NOT a bit-identical MATLAB fitrensemble replication."""
    def fit(self, x, y):
        self.models = []
        for j in range(y.shape[1]):
            model = HistGradientBoostingRegressor(max_iter=100, learning_rate=.08,
                max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False, random_state=0)
            model.fit(x, y[:, j])
            self.models.append(model)
        return self

    def predict(self, x):
        return np.column_stack([m.predict(x) for m in self.models])


class GPResidual:
    """Fixed-kernel, capped-size GP reference; no access to test scenarios in fit."""
    def fit(self, x, y, max_points=512):
        self.scaler = StandardScaler().fit(x)
        index = np.linspace(0, len(x)-1, min(max_points, len(x))).astype(int)
        kernel = RBF(length_scale=np.full(x.shape[1], 3.0)) + WhiteKernel(noise_level=.03)
        self.model = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                             alpha=1e-6, normalize_y=True)
        self.model.fit(self.scaler.transform(x[index]), y[index])
        return self

    def predict(self, x):
        return np.concatenate([self.model.predict(self.scaler.transform(x[i:i+1000]))
                               for i in range(0, len(x), 1000)])


def sensitivity_basis(sensitivity, relative_tolerance=1e-5):
    """Rank-aware orthonormal span of trajectory sensitivities, one per window."""
    u, s, _ = np.linalg.svd(sensitivity, full_matrices=False)
    keep = s > np.maximum(s[:, :1] * relative_tolerance, 1e-10)
    return u * keep[:, None, :]


@dataclass
class NeuralConfig:
    hidden: int = 20
    epochs: int = 80
    patience: int = 12
    batch_size: int = 512
    learning_rate: float = .003
    projection_weight: float = 0.0


class NeuralForecaster:
    def __init__(self, config: NeuralConfig, seed=0):
        self.config, self.seed = config, seed

    def fit(self, x, y, x_val, y_val, sensitivity=None):
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.x_mean, self.x_std = x.mean(0), np.maximum(x.std(0), 1e-5)
        self.y_mean, self.y_scale = y.mean(0), max(float(y.std()), .01)
        xx = torch.as_tensor((x-self.x_mean)/self.x_std, dtype=torch.float32)
        yy = torch.as_tensor((y-self.y_mean)/self.y_scale, dtype=torch.float32)
        xv = torch.as_tensor((x_val-self.x_mean)/self.x_std, dtype=torch.float32)
        yv = torch.as_tensor((y_val-self.y_mean)/self.y_scale, dtype=torch.float32)
        basis = None if sensitivity is None else torch.as_tensor(sensitivity_basis(sensitivity), dtype=torch.float32)
        mean = torch.as_tensor(self.y_mean/self.y_scale, dtype=torch.float32)
        h = self.config.hidden
        self.network = nn.Sequential(nn.Linear(x.shape[1], h), nn.Tanh(),
                                     nn.Linear(h, h), nn.Tanh(), nn.Linear(h, y.shape[1]))
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        best, stale, state = float("inf"), 0, None
        self.log = []
        for epoch in range(self.config.epochs):
            self.network.train()
            losses = []
            order = rng.permutation(len(xx))
            for start in range(0, len(xx), self.config.batch_size):
                ids = order[start:start+self.config.batch_size]
                prediction = self.network(xx[ids])
                loss = torch.mean((prediction-yy[ids])**2)
                if basis is not None and self.config.projection_weight:
                    # Projection acts on the complete residual, including its mean.
                    coordinates = torch.einsum('nhp,nh->np', basis[ids], prediction+mean)
                    loss = loss + self.config.projection_weight * torch.mean(coordinates**2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.network.eval()
            with torch.no_grad():
                validation = float(torch.mean((self.network(xv)-yv)**2))
            self.log.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_loss": validation})
            if validation < best - 1e-7:
                best, stale, state = validation, 0, copy.deepcopy(self.network.state_dict())
            else:
                stale += 1
            if stale >= self.config.patience:
                break
        self.network.load_state_dict(state)
        self.best_validation_mse_rad2 = best * self.y_scale**2
        self.n_parameters = sum(p.numel() for p in self.network.parameters())
        return self

    def predict(self, x):
        self.network.eval()
        with torch.no_grad():
            xx = torch.as_tensor((x-self.x_mean)/self.x_std, dtype=torch.float32)
            out = self.network(xx).numpy()
        return out*self.y_scale + self.y_mean

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.network.state_dict(), "x_mean": self.x_mean,
                    "x_std": self.x_std, "y_mean": self.y_mean, "y_scale": self.y_scale,
                    "config": vars(self.config), "seed": self.seed,
                    "n_parameters": self.n_parameters, "training_log": self.log}, path)

