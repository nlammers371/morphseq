import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from pathlib import Path
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from src.flux.flux_lightning import ClockNPF
from src.flux.flux_data import load_embryo_df, build_training_data, get_data_splits

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings(
    "ignore",
    message=r".*recommended to use `self\.log\('val/.*',.*sync_dist=True`.*"
)

def train_vector_nn(
        root: Path,
        run_name: str,        # fill with run name (optional)
        model_class: str = "legacy",
        model_name: str = "20241107_ds_sweep01_optimum",
        infer_embryo_clock: bool = True,
        mlp_structure: list = [256, 128, 128],
        batch_size: int = 8192,
        max_epochs: int = 250,
        use_pca: bool = True,  # whether to use PCA for dimensionality reduction
        n_pc: int = 10,          # number of PCA components to keep
        wandb_project: str = None,         # fill with your project
        wandb_entity: str = None,          # fill with your entity or org
        wandb_offline: bool = False,       # True for dev/offline
        save_dir: Path = None,          # or another output directory
        n_steps: int= 3,  # number of steps to predict
    ) -> dict:
    
    # make save dir
    if save_dir is None: 
        save_dir = root / "models" / model_class / model_name / "flux" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    # 1) Initialize dataloader 
    embryo_df = load_embryo_df(root, model_class, model_name)
    ds, df, _ = build_training_data(embryo_df=embryo_df, use_pca=use_pca, n_pca_components=n_pc, n_steps= n_steps)
    split_indices = get_data_splits(df=df)
    
    # Calculate model parameters
    dim = ds.z0.shape[1]                        # latent dim
    num_exp = ds.exp_idx.max().item() + 1       # # experiments
    num_embryo = ds.emb_idx.max().item() + 1    # # embryos

    # 2) Initialize model
    lit = ClockNPF(
        dim=dim,
        num_exp=num_exp,
        num_embryo=num_embryo,
        dataset=ds,
        batch_size=batch_size,
        train_indices=split_indices['train'],
        val_indices=split_indices['val'],
        infer_embryo_clock=infer_embryo_clock,
        n_steps= n_steps,  # number of steps to predict
        hidden=tuple(mlp_structure)
    )

    # 3) Logging and checkpointing
    # ---- TensorBoard Logger (always recommended) ----
    tb_logger = TensorBoardLogger(save_dir=save_dir, name="tensorboard")

    # ---- WandB Logger ----
    wandb_logger = None
    if wandb_project:
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            offline=wandb_offline,
            save_dir=save_dir,
            log_model=True,
            sync_tensorboard=True,
        )

    # ---- Model Checkpoint Callback ----
    checkpoint_cb = ModelCheckpoint(
        dirpath=Path(save_dir) / "checkpoints",
        filename="epoch{epoch:02d}",
        save_weights_only=True,
        save_last=True,
    )

    # 4) Trainer setup
    loggers = [tb_logger]
    if wandb_logger is not None:
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        logger=loggers,
        max_epochs=max_epochs,
        precision=16,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb],
        # strategy=DDPStrategy(find_unused_parameters=True),
        devices="auto",
        accelerator="auto",
    )

    # 5) Train!
    trainer.fit(lit)

    # (Optional) Finalize WandB
    if wandb_logger is not None:
        import wandb as _wandb
        _wandb.finish()

    # Optionally return important objects
    return {
        "model": lit,
        "trainer": trainer,
        "tb_logger": tb_logger,
        "wandb_logger": wandb_logger,
        "checkpoint_cb": checkpoint_cb,
    }

if __name__ == "__main__":
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    train_vector_nn(
        root=root,
        run_name= "potential_test2",
        use_pca=True,) 