"""Canonical optimizer-construction entry point for the Liger benchmark.

``build_optimizer(name, params, lr)`` returns a fully configured
``torch.optim.Optimizer``. All hyperparameters other than ``lr`` are
pinned here so the sweep matrix only varies optimizer × LR × seed.

Every optimizer is vendored as a standalone source file in this
``bench/optimizers/`` directory — no sibling-repo imports, no sys.path
gymnastics. This includes the optimizers from our own sibling research
projects (Muogi, RAMuogi, RACASO): we treat them exactly like we treat
Lion and Yogi — copy the source file, document the upstream commit at
the top of the copy, build via a normal Python import.

Canonical configs (single source of truth):

    adam     : torch.optim.Adam(lr, betas=(0.9, 0.999), eps=1e-8)
    adamw    : torch.optim.AdamW(lr, betas=(0.9, 0.999), eps=1e-8, wd=0.01)
    yogi     : Yogi(lr, betas=(0.9, 0.999), eps=1e-3, init_acc=1e-6, wd=0.0)
               — bench/optimizers/yogi.py (Zaheer et al. 2018)
    lion     : Lion(lr, betas=(0.9, 0.99), wd=0.0)
               — bench/optimizers/lion.py (Chen et al. 2023)
    liger    : Liger(lr, betas=(0.9, 0.99), eps_yogi=1e-3, wd=0.0)
               — bench/optimizers/liger.py (vendored from sibling repo)
    muogi    : Muogi(lr, default Muogi config)
               — bench/optimizers/muogi.py
    ramuogi  : RAMuogi(lr, default RAMuogi config)
               — bench/optimizers/ramuogi.py
    racaso   : RACASO(lr, default RACASO config)
               — bench/optimizers/racaso.py
"""

from __future__ import annotations

from typing import List

import torch


KNOWN_OPTIMIZERS = (
    "adam",
    "adamw",
    "yogi",
    "lion",
    "liger",
    "muogi",
    "ramuogi",
    "racaso",
)


# ── Constructors ─────────────────────────────────────────────────────────


def _build_adam(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def _build_adamw(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )


def _build_yogi(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.yogi import Yogi

    return Yogi(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-3,
        initial_accumulator=1e-6,
        weight_decay=0.0,
    )


def _build_lion(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.lion import Lion

    return Lion(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)


def _build_liger(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.liger import Liger

    return Liger(
        params,
        lr=lr,
        betas=(0.9, 0.99),
        eps_yogi=1e-3,
        weight_decay=0.0,
    )


def _build_muogi(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.muogi import Muogi

    return Muogi(params, lr=lr)


def _build_ramuogi(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.ramuogi import RAMuogi

    return RAMuogi(params, lr=lr)


def _build_racaso(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.racaso import RACASO

    return RACASO(params, lr=lr)


def build_optimizer(
    name: str, params: List[torch.Tensor], lr: float
) -> torch.optim.Optimizer:
    """Construct a baseline optimizer by canonical short name.

    Args:
        name: one of ``KNOWN_OPTIMIZERS``.
        params: list of parameter tensors with ``requires_grad=True``.
        lr: learning rate.

    Returns:
        A fully constructed ``torch.optim.Optimizer``.

    Raises:
        ValueError: if ``name`` is unknown.
        NotImplementedError: for sibling-family optimizers when their
            module is not reachable on ``sys.path``.
    """
    if name not in KNOWN_OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer name '{name}'; known: {sorted(KNOWN_OPTIMIZERS)}"
        )
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not params:
        raise ValueError("params must be a non-empty list of tensors")

    builders = {
        "adam": _build_adam,
        "adamw": _build_adamw,
        "yogi": _build_yogi,
        "lion": _build_lion,
        "liger": _build_liger,
        "muogi": _build_muogi,
        "ramuogi": _build_ramuogi,
        "racaso": _build_racaso,
    }
    return builders[name](params, lr)
