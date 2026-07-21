from __future__ import annotations

import copy
from enum import Enum
from itertools import cycle
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Divergence(Enum):
    """Supported $\phi$-divergence objectives for RDR estimation."""

    HELLINGER = "hellinger"
    KL = "kl"
    CHISQ = "chisq"


class RDRTrainer:
    """Train a neural network to estimate the relative density ratio $r(x)=2p(x)/(p(x)+q(x))$."""

    def __init__(
        self,
        model: nn.Module,
        divergence: Union[Divergence, str] = Divergence.HELLINGER,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        device: Optional[Union[str, torch.device]] = None,
        mixture_ratio: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.divergence = self._normalize_divergence(divergence)
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mixture_ratio = mixture_ratio
        self.eps = eps
        self.validation_history: List[float] = []
        self.best_epoch: Optional[int] = None
        self.stopped_epoch: Optional[int] = None
        self.test_loss: Optional[float] = None
        self.model.to(self.device)

    @staticmethod
    def _normalize_divergence(divergence: Union[Divergence, str]) -> Divergence:
        if isinstance(divergence, Divergence):
            return divergence
        if isinstance(divergence, str):
            value = divergence.lower()
            if value in {"hellinger", "squared_hellinger", "squared-hellinger"}:
                return Divergence.HELLINGER
            if value in {"kl", "kullback_leibler"}:
                return Divergence.KL
            if value in {"chisq", "chi2", "chi-square"}:
                return Divergence.CHISQ
        raise ValueError(f"Unsupported divergence: {divergence}")

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        batch = torch.as_tensor(batch, dtype=torch.float32)
        return batch.to(self.device)

    def _loss(self, x_real: torch.Tensor, x_gen: torch.Tensor) -> torch.Tensor:
        # The model owns the output activation and returns the final ratio.
        # Evaluate the combined batch once so batch-normalization sees p and q
        # together. The denominator expectation is then formed under
        # m = mixture_ratio * p + (1 - mixture_ratio) * q.
        x_mix = torch.cat([x_real, x_gen], dim=0)
        score = self.model(x_mix).reshape(-1)
        pos = score[: x_real.size(0)]
        neg = score[x_real.size(0) :]

        if self.divergence == Divergence.HELLINGER:
            h_pos = torch.mean(pos.clamp_min(self.eps).pow(-0.5))
            h_mix = self.mixture_ratio * torch.mean(pos.clamp_min(self.eps).pow(0.5)) + (
                1.0 - self.mixture_ratio
            ) * torch.mean(neg.clamp_min(self.eps).pow(0.5))
            divergence_term = h_pos + h_mix - 2.0
        elif self.divergence == Divergence.KL:
            mean_mix = self.mixture_ratio * torch.mean(pos) + (1.0 - self.mixture_ratio) * torch.mean(neg)
            divergence_term = -(1.0 + torch.mean(torch.log(pos.clamp_min(self.eps))) - mean_mix)
        elif self.divergence == Divergence.CHISQ:
            mean_mix_sq = self.mixture_ratio * torch.mean(pos.pow(2)) + (
                1.0 - self.mixture_ratio
            ) * torch.mean(neg.pow(2))
            divergence_term = -(2.0 * torch.mean(pos) - mean_mix_sq + 1.0)
        else:
            raise RuntimeError("Unknown divergence")

        return divergence_term

    @staticmethod
    def _split_tensor(
        x: torch.Tensor,
        validation_fraction: float,
        test_fraction: float,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
            raise ValueError("validation_fraction and test_fraction must be nonnegative and sum to less than 1")

        n = x.size(0)
        n_validation = int(round(n * validation_fraction))
        n_test = int(round(n * test_fraction))
        if validation_fraction > 0:
            n_validation = max(1, n_validation)
        if test_fraction > 0:
            n_test = max(1, n_test)
        if n_validation + n_test >= n:
            raise ValueError("Not enough observations for the requested validation and test fractions")

        indices = torch.randperm(n, generator=generator)
        test = x[indices[:n_test]] if n_test else None
        validation = x[indices[n_test : n_test + n_validation]] if n_validation else None
        train = x[indices[n_test + n_validation :]]
        return train, validation, test

    @staticmethod
    def _make_loaders(
        x_real: torch.Tensor,
        x_gen: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        pin_memory: bool,
    ) -> Tuple[DataLoader, DataLoader]:
        real_loader = DataLoader(
            TensorDataset(x_real), batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
        )
        gen_loader = DataLoader(
            TensorDataset(x_gen), batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
        )
        return real_loader, gen_loader

    def _mean_loss(self, real_loader: DataLoader, gen_loader: DataLoader) -> float:
        num_batches = max(len(real_loader), len(gen_loader))
        real_batches = cycle(real_loader)
        gen_batches = cycle(gen_loader)
        total = 0.0
        with torch.no_grad():
            for _ in range(num_batches):
                total += float(self._loss(next(real_batches)[0], next(gen_batches)[0]).detach().cpu())
        return total / max(1, num_batches)

    def fit(
        self,
        x_real: torch.Tensor,
        x_gen: torch.Tensor,
        num_epochs: int = 200,
        batch_size: int = 256,
        shuffle: bool = True,
        verbose: bool = True,
        pin_memory: bool = False,
        x_real_val: Optional[torch.Tensor] = None,
        x_gen_val: Optional[torch.Tensor] = None,
        x_real_test: Optional[torch.Tensor] = None,
        x_gen_test: Optional[torch.Tensor] = None,
        validation_fraction: float = 0.0,
        test_fraction: float = 0.0,
        split_seed: int = 0,
        early_stopping_patience: Optional[int] = None,
        early_stopping_start: int = 0,
        min_delta: float = 1e-5,
        restore_best: bool = True,
        scheduler: Optional[object] = None,
    ) -> Tuple[nn.Module, List[float]]:
        if (x_real_val is None) != (x_gen_val is None):
            raise ValueError("x_real_val and x_gen_val must be provided together")
        if (x_real_test is None) != (x_gen_test is None):
            raise ValueError("x_real_test and x_gen_test must be provided together")
        if x_real_val is not None and validation_fraction:
            raise ValueError("Use either explicit validation tensors or validation_fraction, not both")
        if x_real_test is not None and test_fraction:
            raise ValueError("Use either explicit test tensors or test_fraction, not both")
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if early_stopping_patience is not None and x_real_val is None and validation_fraction == 0:
            raise ValueError("Early stopping requires validation data or validation_fraction > 0")

        x_real = torch.as_tensor(x_real, dtype=torch.float32)
        x_gen = torch.as_tensor(x_gen, dtype=torch.float32)
        generator = torch.Generator().manual_seed(split_seed)
        x_real, automatic_real_val, automatic_real_test = self._split_tensor(
            x_real, validation_fraction, test_fraction, generator
        )
        x_gen, automatic_gen_val, automatic_gen_test = self._split_tensor(
            x_gen, validation_fraction, test_fraction, generator
        )
        x_real_val = automatic_real_val if x_real_val is None else torch.as_tensor(x_real_val, dtype=torch.float32)
        x_gen_val = automatic_gen_val if x_gen_val is None else torch.as_tensor(x_gen_val, dtype=torch.float32)
        x_real_test = automatic_real_test if x_real_test is None else torch.as_tensor(x_real_test, dtype=torch.float32)
        x_gen_test = automatic_gen_test if x_gen_test is None else torch.as_tensor(x_gen_test, dtype=torch.float32)

        x_real = self._prepare_batch(x_real)
        x_gen = self._prepare_batch(x_gen)
        x_real_val = self._prepare_batch(x_real_val) if x_real_val is not None else None
        x_gen_val = self._prepare_batch(x_gen_val) if x_gen_val is not None else None
        x_real_test = self._prepare_batch(x_real_test) if x_real_test is not None else None
        x_gen_test = self._prepare_batch(x_gen_test) if x_gen_test is not None else None

        real_loader, gen_loader = self._make_loaders(
            x_real, x_gen, batch_size, shuffle, pin_memory
        )
        val_loaders = (
            self._make_loaders(x_real_val, x_gen_val, batch_size, False, pin_memory)
            if x_real_val is not None and x_gen_val is not None
            else None
        )
        test_loaders = (
            self._make_loaders(x_real_test, x_gen_test, batch_size, False, pin_memory)
            if x_real_test is not None and x_gen_test is not None
            else None
        )

        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        history: List[float] = []
        self.validation_history = []
        self.best_epoch = None
        self.stopped_epoch = None
        self.test_loss = None
        best_validation = float("inf")
        best_state = None
        epochs_without_improvement = 0
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = max(len(real_loader), len(gen_loader))
            real_batches = cycle(real_loader)
            gen_batches = cycle(gen_loader)
            for _ in range(num_batches):
                batch_real = next(real_batches)[0]
                batch_gen = next(gen_batches)[0]
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch_real, batch_gen)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.detach().cpu())

            epoch_loss /= max(1, num_batches)
            history.append(epoch_loss)

            validation_loss = None
            if val_loaders is not None:
                self.model.eval()
                validation_loss = self._mean_loss(*val_loaders)
                self.validation_history.append(validation_loss)
                if validation_loss < best_validation - min_delta:
                    best_validation = validation_loss
                    self.best_epoch = epoch
                    epochs_without_improvement = 0
                    if restore_best:
                        best_state = copy.deepcopy(self.model.state_dict())
                elif epoch >= early_stopping_start:
                    epochs_without_improvement += 1
                if scheduler is not None:
                    scheduler.step(validation_loss)
            elif scheduler is not None:
                scheduler.step(epoch_loss)

            if verbose:
                message = f"epoch {epoch + 1:03d} | loss={epoch_loss:.6f}"
                if validation_loss is not None:
                    message += f" | val_loss={validation_loss:.6f}"
                print(message)

            if (
                validation_loss is not None
                and early_stopping_patience is not None
                and epoch >= early_stopping_start
                and epochs_without_improvement >= early_stopping_patience
            ):
                self.stopped_epoch = epoch
                break

        if restore_best and best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        if test_loaders is not None:
            self.test_loss = self._mean_loss(*test_loaders)
        return self.model, history

    def score(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(self._prepare_batch(x)).reshape(-1)


def evaluate_ratio(
    model: nn.Module,
    x_real: torch.Tensor,
    x_gen: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_real = torch.as_tensor(x_real, dtype=torch.float32).to(device)
        x_gen = torch.as_tensor(x_gen, dtype=torch.float32).to(device)
        g_real = model(x_real).reshape(-1)
        g_gen = model(x_gen).reshape(-1)

    return {"g_real": g_real, "g_gen": g_gen}


def estimate_relative_density_ratio(
    model: nn.Module,
    x_real: torch.Tensor,
    x_gen: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    return evaluate_ratio(model, x_real, x_gen, device=device)
