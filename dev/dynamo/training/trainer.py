"""Stage 1 training loop for the phi0-only model.

Handles data loading, train/eval splitting, the epoch loop, checkpointing,
and W&B logging. Produces the permanent Stage 1 checkpoint that serves as
the baseline for all subsequent model stages.

Model spec references: §7.1 (staged training), §7.2 (data sampling),
    §7.5 (forward pass), §15.2 step 5.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..data.dataset import FragmentDataset, fragment_collate_fn, worker_init_fn
from ..data.loading import TrajectoryDataset, load_trajectories
from ..eval.evaluate import run_evaluation
from ..models.dynamics import Phi0OnlyModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training configuration for Stage 1 (phi0-only model).

    All fields can be set from a YAML config file and/or CLI overrides.
    """

    # --- Data ---
    build_dir: str = ""
    experiment_ids: Optional[List[str]] = None
    n_components: int = 10
    scale: bool = True
    min_trajectory_length: int = 3
    min_context: int = 3
    max_context: Optional[int] = None
    horizons: Tuple[int, ...] = (1, 2, 3, 4)
    gamma: float = 0.5
    n_targets: int = 1
    eval_fraction: float = 0.15
    batch_size: int = 64
    num_workers: int = 0

    # --- Model ---
    hidden_dim: int = 64
    n_hidden: int = 2
    activation: str = "softplus"
    init_log_beta: float = 0.0
    init_log_D: float = -2.0
    n_forward_samples: int = 50
    rate_clamp_min: float = 1e-6
    alpha_0: float = 0.01
    hessian_n_points: int = 64
    normalize_rate: bool = True
    log_beta_T: Optional[float] = None
    T_ref: float = 28.5

    # --- Optimizer ---
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    grad_clip_norm: float = 10.0

    # --- Training ---
    n_epochs: int = 200
    epoch_length: int = 2000
    eval_every: int = 10
    eval_n_batches: int = 50

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints/stage1"
    save_every: int = 50

    # --- Logging ---
    wandb_project: str = "morphseq-dynamo"
    wandb_run_name: Optional[str] = None
    log_every: int = 1

    # --- Device ---
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Train / eval split
# ---------------------------------------------------------------------------

def _split_trajectories(
    dataset: TrajectoryDataset,
    eval_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[TrajectoryDataset, TrajectoryDataset]:
    """Split trajectories into train/eval sets, stratified by perturbation class.

    The split is at the embryo level (not fragment level) to prevent data
    leakage between train and eval.

    Args:
        dataset: Full TrajectoryDataset.
        eval_fraction: Fraction of embryos to hold out for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        (train_dataset, eval_dataset) — new TrajectoryDataset instances
        sharing the same PCA/scaler artifacts.
    """
    rng = np.random.default_rng(seed)

    # Group trajectories by perturbation class
    class_groups: Dict[str, list] = {}
    for traj in dataset.trajectories:
        class_groups.setdefault(traj.perturbation_class, []).append(traj)

    train_trajs: list = []
    eval_trajs: list = []

    for cls, trajs in class_groups.items():
        n_eval = max(1, int(len(trajs) * eval_fraction))
        indices = rng.permutation(len(trajs))
        eval_idx = set(indices[:n_eval].tolist())
        for i, t in enumerate(trajs):
            if i in eval_idx:
                eval_trajs.append(t)
            else:
                train_trajs.append(t)

    train_ds = TrajectoryDataset(
        trajectories=train_trajs,
        pca=dataset.pca,
        scaler=dataset.scaler,
        z_mu_cols=dataset.z_mu_cols,
        class_to_idx=dataset.class_to_idx,
    )
    eval_ds = TrajectoryDataset(
        trajectories=eval_trajs,
        pca=dataset.pca,
        scaler=dataset.scaler,
        z_mu_cols=dataset.z_mu_cols,
        class_to_idx=dataset.class_to_idx,
    )
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Stage1Trainer:
    """Training loop for the phi0-only model (Stage 1).

    Args:
        config: TrainConfig with all hyperparameters.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

    def train(self) -> Path:
        """Run full training. Returns path to best checkpoint."""
        cfg = self.config

        # ----- Data -----
        logger.info("Loading trajectories...")
        traj_dataset = load_trajectories(
            experiment_ids=cfg.experiment_ids,
            build_dir=cfg.build_dir,
            n_components=cfg.n_components,
            scale=cfg.scale,
            min_trajectory_length=cfg.min_trajectory_length,
        )
        logger.info(
            f"Loaded {len(traj_dataset)} embryos, "
            f"{traj_dataset.n_components} components, "
            f"{len(traj_dataset.class_to_idx)} classes"
        )

        train_ds, eval_ds = _split_trajectories(traj_dataset, cfg.eval_fraction)
        logger.info(f"Train: {len(train_ds)} embryos, Eval: {len(eval_ds)} embryos")

        train_frag = FragmentDataset(
            train_ds,
            min_context=cfg.min_context,
            max_context=cfg.max_context,
            horizons=cfg.horizons,
            epoch_length=cfg.epoch_length,
            gamma=cfg.gamma,
            n_targets=cfg.n_targets,
        )
        eval_frag = FragmentDataset(
            eval_ds,
            min_context=cfg.min_context,
            max_context=cfg.max_context,
            horizons=cfg.horizons,
            gamma=0.0,  # No rebalancing for eval — use natural frequencies
            n_targets=1,  # Single target for eval metrics
        )

        train_loader = DataLoader(
            train_frag,
            batch_size=cfg.batch_size,
            collate_fn=fragment_collate_fn,
            worker_init_fn=worker_init_fn,
            num_workers=cfg.num_workers,
        )

        # ----- Model -----
        model = Phi0OnlyModel(
            input_dim=traj_dataset.n_components,
            hidden_dim=cfg.hidden_dim,
            n_hidden=cfg.n_hidden,
            activation=cfg.activation,
            init_log_beta=cfg.init_log_beta,
            init_log_D=cfg.init_log_D,
            n_forward_samples=cfg.n_forward_samples,
            rate_clamp_min=cfg.rate_clamp_min,
            alpha_0=cfg.alpha_0,
            hessian_n_points=cfg.hessian_n_points,
            normalize_rate=cfg.normalize_rate,
            log_beta_T=cfg.log_beta_T,
            T_ref=cfg.T_ref,
        ).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # ----- Optimizer -----
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = None
        if cfg.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

        # ----- W&B -----
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=asdict(cfg),
            )
            use_wandb = True
        except ImportError:
            use_wandb = False
            logger.info("wandb not available; logging to stdout only.")

        # ----- Training loop -----
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_nll = float("inf")
        best_path = ckpt_dir / "best.pt"

        for epoch in range(cfg.n_epochs):
            # Train
            train_metrics = self._train_epoch(model, train_loader, optimizer, epoch)

            if scheduler is not None:
                scheduler.step()

            # Log
            if (epoch + 1) % cfg.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"Epoch {epoch+1:4d}/{cfg.n_epochs} | "
                    f"loss={train_metrics['loss']:.4f} | "
                    f"nll={train_metrics['nll']:.4f} | "
                    f"R0={train_metrics['hessian_penalty']:.4f} | "
                    f"beta={train_metrics['beta']:.4f} | "
                    f"D={train_metrics['D']:.6f} | "
                    f"mean_R_e={train_metrics['mean_R_e']:.4f} | "
                    f"lr={lr:.2e}"
                )
                logger.info(msg)
                if use_wandb:
                    wandb.log({**train_metrics, "lr": lr}, step=epoch)

            # Evaluate
            if (epoch + 1) % cfg.eval_every == 0:
                model.eval()
                eval_result = run_evaluation(
                    model, eval_frag,
                    n_batches=cfg.eval_n_batches,
                    batch_size=cfg.batch_size,
                    tier="eval",
                )
                model.train()
                eval_nll = eval_result.metrics.get("nll", float("inf"))
                logger.info(
                    f"  Eval NLL={eval_nll:.4f} | "
                    f"MSE={eval_result.metrics.get('mse', 0):.6f} | "
                    f"Cal={eval_result.calibration:.3f}"
                )
                if use_wandb:
                    eval_log = {f"eval/{k}": v for k, v in eval_result.metrics.items()}
                    eval_log["eval/calibration"] = eval_result.calibration
                    wandb.log(eval_log, step=epoch)

                if eval_nll < best_nll:
                    best_nll = eval_nll
                    self._save_checkpoint(model, optimizer, epoch, best_nll,
                                          traj_dataset.n_components, best_path)
                    logger.info(f"  New best model saved (NLL={best_nll:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % cfg.save_every == 0:
                ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
                self._save_checkpoint(model, optimizer, epoch, best_nll,
                                      traj_dataset.n_components, ckpt_path)

        # Final checkpoint
        final_path = ckpt_dir / "final.pt"
        self._save_checkpoint(model, optimizer, cfg.n_epochs - 1, best_nll,
                              traj_dataset.n_components, final_path)
        logger.info(f"Training complete. Best checkpoint: {best_path}")

        if use_wandb:
            wandb.finish()

        return best_path

    def _train_epoch(
        self,
        model: Phi0OnlyModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        """Single training epoch. Returns dict of aggregated scalars."""
        model.train()
        cfg = self.config

        total_loss = 0.0
        total_nll = 0.0
        total_r0 = 0.0
        total_R_e = 0.0
        n_batches = 0

        for batch in loader:
            batch = _batch_to_device(batch, self.device)

            result = model(batch)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            total_nll += result["nll"].mean().item()
            total_r0 += result["hessian_penalty"].item()
            total_R_e += result["R_e"].mean().item()
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "nll": total_nll / max(n_batches, 1),
            "hessian_penalty": total_r0 / max(n_batches, 1),
            "beta": model.beta.item(),
            "D": model.D.item(),
            "mean_R_e": total_R_e / max(n_batches, 1),
        }

    def _save_checkpoint(
        self,
        model: Phi0OnlyModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_nll: float,
        input_dim: int,
        path: Path,
    ) -> None:
        """Save model + optimizer state."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(self.config),
            "best_nll": best_nll,
            "input_dim": input_dim,
        }, path)

    @staticmethod
    def load_checkpoint(
        path: Path,
        device: str = "cpu",
    ) -> Tuple[Phi0OnlyModel, TrainConfig]:
        """Load a saved checkpoint.

        Args:
            path: Path to .pt checkpoint file.
            device: Device to load model onto.

        Returns:
            (model, config) tuple.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = ckpt["config"]

        # Reconstruct config (handle tuple fields)
        if "horizons" in cfg_dict and isinstance(cfg_dict["horizons"], list):
            cfg_dict["horizons"] = tuple(cfg_dict["horizons"])
        config = TrainConfig(**cfg_dict)

        model = Phi0OnlyModel(
            input_dim=ckpt["input_dim"],
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            activation=config.activation,
            init_log_beta=config.init_log_beta,
            init_log_D=config.init_log_D,
            n_forward_samples=config.n_forward_samples,
            rate_clamp_min=config.rate_clamp_min,
            alpha_0=getattr(config, "alpha_0", 0.01),
            hessian_n_points=getattr(config, "hessian_n_points", 64),
            normalize_rate=getattr(config, "normalize_rate", True),
            log_beta_T=getattr(config, "log_beta_T", None),
            T_ref=getattr(config, "T_ref", 28.5),
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        model.eval()

        return model, config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_to_device(batch, device: torch.device):
    """Move all tensors in a FragmentBatch to the given device."""
    from ..data.dataset import FragmentBatch
    return FragmentBatch(
        context=batch.context.to(device),
        context_mask=batch.context_mask.to(device),
        targets=batch.targets.to(device),
        predecessors=batch.predecessors.to(device),
        time_deltas=batch.time_deltas.to(device),
        horizon_dts=batch.horizon_dts.to(device),
        context_to_target_dts=batch.context_to_target_dts.to(device),
        delta_t=batch.delta_t.to(device),
        temperature=batch.temperature.to(device),
        class_idx=batch.class_idx.to(device),
        embryo_idx=batch.embryo_idx.to(device),
    )
