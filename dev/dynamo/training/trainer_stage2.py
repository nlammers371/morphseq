"""Stage 2 training loop for orthogonal modes model.

Loads a frozen Stage 1 checkpoint (phi0), introduces antisymmetric mode
matrices S_m, and trains with the alternating c_e/R_e closed-form solve.

Model spec references: §3.4 (orthogonal modes), §7.1 (staged training),
    §15.2 step 6.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..data.dataset import FragmentDataset, fragment_collate_fn, worker_init_fn
from ..data.loading import TrajectoryDataset, load_trajectories
from ..eval.evaluate import run_evaluation
from ..models.orthogonal import OrthogonalModesModel
from .trainer import Stage1Trainer, TrainConfig, _batch_to_device, _split_trajectories

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Stage2Config:
    """Training configuration for Stage 2 (orthogonal modes).

    Inherits data/training fields from TrainConfig where applicable.
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

    # --- Stage 1 checkpoint ---
    stage1_checkpoint: str = ""

    # --- Model (from Stage 1) ---
    hidden_dim: int = 64
    n_hidden: int = 2
    activation: str = "softplus"

    # --- Orthogonal mode parameters ---
    n_modes: int = 5
    lambda_c: float = 1.0
    n_alternations: int = 2
    s_init_scale: float = 0.01
    n_forward_samples: int = 50
    rate_clamp_min: float = 1e-6
    normalize_rate: bool = True
    alpha_0: float = 0.01
    hessian_n_points: int = 64

    # --- Temperature ---
    log_beta_T: Optional[float] = None
    T_ref: float = 28.5

    # --- Optimizer ---
    lr: float = 5e-4
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    grad_clip_norm: float = 10.0

    # --- Training ---
    n_epochs: int = 150
    epoch_length: int = 2000
    eval_every: int = 10
    eval_n_batches: int = 50

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints/stage2_orthogonal"
    save_every: int = 50

    # --- Logging ---
    wandb_project: str = "morphseq-dynamo"
    wandb_run_name: Optional[str] = None
    log_every: int = 1

    # --- Device ---
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Stage2Trainer:
    """Training loop for orthogonal modes model (Stage 2).

    Loads a frozen Stage 1 phi0 checkpoint, introduces S_m matrices and
    class-level priors, and trains via the alternating c_e/R_e solve.

    Args:
        config: Stage2Config with all hyperparameters.
    """

    def __init__(self, config: Stage2Config) -> None:
        self.config = config
        self.device = torch.device(config.device)

    def train(self) -> Path:
        """Run full Stage 2 training. Returns path to best checkpoint."""
        cfg = self.config

        # ----- Load Stage 1 checkpoint -----
        logger.info(f"Loading Stage 1 checkpoint: {cfg.stage1_checkpoint}")
        s1_model, s1_config = Stage1Trainer.load_checkpoint(
            Path(cfg.stage1_checkpoint), device=cfg.device
        )
        logger.info(
            f"Stage 1 model loaded: beta={s1_model.beta.item():.4f}, "
            f"D={s1_model.D.item():.6f}"
        )

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
            gamma=0.0,
            n_targets=1,
        )

        train_loader = DataLoader(
            train_frag,
            batch_size=cfg.batch_size,
            collate_fn=fragment_collate_fn,
            worker_init_fn=worker_init_fn,
            num_workers=cfg.num_workers,
        )

        # ----- Model -----
        n_classes = len(traj_dataset.class_to_idx)
        model = OrthogonalModesModel(
            phi0=s1_model.phi0,
            input_dim=traj_dataset.n_components,
            n_modes=cfg.n_modes,
            n_classes=n_classes,
            init_log_beta=s1_model.log_beta.item(),
            init_log_D=s1_model.log_D.item(),
            lambda_c=cfg.lambda_c,
            n_forward_samples=cfg.n_forward_samples,
            rate_clamp_min=cfg.rate_clamp_min,
            n_alternations=cfg.n_alternations,
            s_init_scale=cfg.s_init_scale,
            normalize_rate=cfg.normalize_rate,
            log_beta_T=cfg.log_beta_T,
            T_ref=cfg.T_ref,
            alpha_0=cfg.alpha_0,
            hessian_n_points=cfg.hessian_n_points,
        ).to(self.device)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model parameters: {n_total:,} total, {n_trainable:,} trainable "
            f"(phi0 frozen: {n_total - n_trainable:,})"
        )

        # ----- Optimizer (only trainable params) -----
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
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
                    f"D={train_metrics['D']:.6f} | "
                    f"mean_c_norm={train_metrics['mean_c_norm']:.4f} | "
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
                                          traj_dataset.n_components, n_classes,
                                          best_path)
                    logger.info(f"  New best model saved (NLL={best_nll:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % cfg.save_every == 0:
                ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
                self._save_checkpoint(model, optimizer, epoch, best_nll,
                                      traj_dataset.n_components, n_classes,
                                      ckpt_path)

        # Final checkpoint
        final_path = ckpt_dir / "final.pt"
        self._save_checkpoint(model, optimizer, cfg.n_epochs - 1, best_nll,
                              traj_dataset.n_components, n_classes, final_path)
        logger.info(f"Training complete. Best checkpoint: {best_path}")

        if use_wandb:
            wandb.finish()

        return best_path

    def _train_epoch(
        self,
        model: OrthogonalModesModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        """Single training epoch. Returns dict of aggregated scalars."""
        model.train()
        cfg = self.config

        total_loss = 0.0
        total_nll = 0.0
        total_R_e = 0.0
        total_c_norm = 0.0
        n_batches = 0

        for batch in loader:
            batch = _batch_to_device(batch, self.device)

            result = model(batch)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    cfg.grad_clip_norm,
                )
            optimizer.step()

            total_loss += loss.item()
            total_nll += result["nll"].mean().item()
            total_R_e += result["R_e"].mean().item()
            total_c_norm += result["mean_c_norm"].item()
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "nll": total_nll / n,
            "beta": model.beta.item(),
            "D": model.D.item(),
            "mean_R_e": total_R_e / n,
            "mean_c_norm": total_c_norm / n,
        }

    def _save_checkpoint(
        self,
        model: OrthogonalModesModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_nll: float,
        input_dim: int,
        n_classes: int,
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
            "n_classes": n_classes,
            "stage": 2,
            "model_type": "orthogonal",
        }, path)

    @staticmethod
    def load_checkpoint(
        path: Path,
        stage1_checkpoint: Path,
        device: str = "cpu",
    ) -> Tuple[OrthogonalModesModel, Stage2Config]:
        """Load a saved Stage 2 checkpoint.

        Args:
            path: Path to Stage 2 .pt checkpoint file.
            stage1_checkpoint: Path to Stage 1 checkpoint (needed to
                reconstruct the frozen phi0 network).
            device: Device to load model onto.

        Returns:
            (model, config) tuple.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = ckpt["config"]

        # Handle tuple fields
        if "horizons" in cfg_dict and isinstance(cfg_dict["horizons"], list):
            cfg_dict["horizons"] = tuple(cfg_dict["horizons"])
        config = Stage2Config(**cfg_dict)

        # Load Stage 1 model for phi0
        s1_model, _ = Stage1Trainer.load_checkpoint(stage1_checkpoint, device=device)

        model = OrthogonalModesModel(
            phi0=s1_model.phi0,
            input_dim=ckpt["input_dim"],
            n_modes=config.n_modes,
            n_classes=ckpt.get("n_classes", 1),
            init_log_beta=s1_model.log_beta.item(),
            init_log_D=config.lr,  # Will be overwritten by state_dict
            lambda_c=config.lambda_c,
            n_forward_samples=config.n_forward_samples,
            rate_clamp_min=config.rate_clamp_min,
            n_alternations=config.n_alternations,
            s_init_scale=config.s_init_scale,
            normalize_rate=config.normalize_rate,
            log_beta_T=config.log_beta_T,
            T_ref=config.T_ref,
            alpha_0=config.alpha_0,
            hessian_n_points=config.hessian_n_points,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        return model, config
