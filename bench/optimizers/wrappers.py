"""Canonical optimizer-construction entry point for the Liger benchmark.

``build_optimizer(name, params, lr)`` returns a fully configured
``torch.optim.Optimizer``. All hyperparameters other than ``lr`` are
pinned here so the sweep matrix only varies optimizer × LR × seed.

Canonical configs (single source of truth):

    adam     : torch.optim.Adam(lr, betas=(0.9, 0.999), eps=1e-8)
    adamw    : torch.optim.AdamW(lr, betas=(0.9, 0.999), eps=1e-8, wd=0.01)
    yogi     : Yogi(lr, betas=(0.9, 0.999), eps=1e-3, init_acc=1e-6, wd=0.0)
               — vendored at bench/optimizers/yogi.py
    lion     : Lion(lr, betas=(0.9, 0.99), wd=0.0)
               — vendored at bench/optimizers/lion.py (Chen et al. 2023)
    liger    : Liger(lr, betas=(0.9, 0.99), eps_yogi=1e-3, wd=0.0)
               — the optimizer under study; imported from parent liger.py
    muogi    : Muogi(lr, default Muogi config)
               — imported from sibling Muogi/muogi.py (NotImplementedError
                 with pointer if not on sys.path)
    ramuogi  : RAMuogi(lr, default RAMuogi config)
               — imported from sibling Muogi/ramuogi.py
    racaso   : RACASO(lr, default RACASO config)
               — imported from sibling RACASO/racaso.py
"""

from __future__ import annotations

import sys
from pathlib import Path
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


def _sibling_path(name: str) -> Path:
    """Return the absolute path to a sibling research project directory."""
    here = Path(__file__).resolve()
    return here.parents[3] / name


def _add_to_syspath(path: Path) -> None:
    sp = str(path)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _add_parent_liger_to_syspath() -> None:
    """The Liger optimizer lives one directory above bench/."""
    liger_root = Path(__file__).resolve().parents[2]
    _add_to_syspath(liger_root)


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
    _add_parent_liger_to_syspath()
    from liger import Liger  # type: ignore[import-not-found]

    return Liger(
        params,
        lr=lr,
        betas=(0.9, 0.99),
        eps_yogi=1e-3,
        weight_decay=0.0,
    )


def _build_muogi(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    _add_to_syspath(_sibling_path("Muogi"))
    try:
        from muogi import Muogi  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NotImplementedError(
            "muogi.Muogi not importable; ensure ../Muogi/ is reachable. "
            f"Original error: {exc}"
        ) from exc
    return Muogi(params, lr=lr)


def _build_ramuogi(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    _add_to_syspath(_sibling_path("Muogi"))
    try:
        from ramuogi import RAMuogi  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NotImplementedError(
            "ramuogi.RAMuogi not importable; ensure ../Muogi/ is reachable. "
            f"Original error: {exc}"
        ) from exc
    return RAMuogi(params, lr=lr)


def _build_racaso(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    _add_to_syspath(_sibling_path("RACASO"))
    try:
        from racaso import RACASO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NotImplementedError(
            "racaso.RACASO not importable; ensure ../RACASO/ is reachable. "
            f"Original error: {exc}"
        ) from exc
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
