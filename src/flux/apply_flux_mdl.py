import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

import torch
import warnings
import pytorch_lightning as pl

from src.flux.flux_lightning import ClockNPF
from src.flux.flux_data import load_embryo_df, build_training_data

# ----------------------------
# CONFIGURATION
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore", message=r".*recommended to use `self\.log\('val/.*',.*sync_dist=True`.*")

def load_and_apply_flux_model(
    root: Path,
    run_name: str,
    model_class: str = "legacy",
    model_name: str = "20241107_ds_sweep01_optimum",
    checkpoint_name: str = "last.ckpt",
    use_pca: bool = True,
    n_pc: int = 10,
    n_steps: int = 3,
    alt_embryo_df: Path = None,
):
    """
    Load a trained ClockNPF model and apply it to training or alternative dataset.
    """
    # ----------------------------
    # Load data
    # ----------------------------
    if alt_embryo_df is not None:
        import pandas as pd
        embryo_df = pd.read_csv(alt_embryo_df)
    else:
        embryo_df = load_embryo_df(root, model_class, model_name)

    ds, _, df = build_training_data(
        embryo_df=embryo_df,
        use_pca=use_pca,
        n_pca_components=n_pc,
        n_steps=n_steps
    )

    dim = ds.z0.shape[1]
    num_exp = ds.exp_idx.max().item() + 1
    num_embryo = ds.emb_idx.max().item() + 1

    # ----------------------------
    # Load model from checkpoint
    # ----------------------------
    ckpt_path = root / "models" / model_class / model_name / "flux" / run_name / "checkpoints" / checkpoint_name
    model = ClockNPF.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        dim=dim,
        num_exp=num_exp,
        num_embryo=num_embryo,
        dataset=ds,
        n_steps=n_steps,
        infer_embryo_clock=True,
        hidden=(256, 128, 128),  # must match training config
        batch_size=8192,         # must match training config
        train_indices=[],
        val_indices=[]
    )

    model.eval()
    model.freeze()

    # ----------------------------
    # Extract learned rate shifts
    # ----------------------------
    rate_shifts = {}

    if hasattr(model, "log_temp_coeff"):
        rate_shifts["temperature"] = model.log_temp_coeff.detach().cpu().numpy()

    if hasattr(model, "log_s"):
        rate_shifts["experiment"] = model.log_s.detach().cpu().numpy()

    if hasattr(model, "delta"):
        rate_shifts["embryo"] = model.delta.detach().cpu().numpy()

    # Optionally save to CSV
    rate_out_path = root / "models" / model_class / model_name / "flux" / run_name / "rate_shifts"
    rate_out_path.mkdir(parents=True, exist_ok=True)

    for key, values in rate_shifts.items():
        out_file = rate_out_path / f"{key}_rate_shifts.csv"
        with open(out_file, "w") as f:
            f.write("index,rate_shift\n")
            for i, val in enumerate(values):
                f.write(f"{i},{val}\n")

    # ----------------------------
    # Inference
    # ----------------------------
    phi_preds = []
    vel_preds = []
    # with torch.no_grad():
    with torch.enable_grad():
        for batch in torch.utils.data.DataLoader(ds, batch_size=8192, shuffle=False):
            z = batch["z0"].clone().detach().requires_grad_(True)
            exp_idx = batch["exp"]
            emb_idx = batch["emb"]
            temp = batch["temp"]
            out = model(z=z, exp_idx=exp_idx, emb_idx=emb_idx, temp=temp)
            
            vel_preds.append(out[0])
            phi_preds.append(out[1])

    phi_stack = torch.cat(phi_preds, dim=0)
    vel_stack = torch.cat(vel_preds, dim=0)
    phi_cols = [f"phi_{i:03}" for i in range(phi_stack.shape[1])]
    vel_cols = [f"vel_{i:03}" for i in range(vel_stack.shape[1])]

    df.loc[:, phi_cols] = phi_stack.detach().cpu().numpy()
    df.loc[:, vel_cols] = vel_stack.detach().cpu().numpy()

    out_path = root / "models" / model_class / model_name / "flux" / run_name / "flux_results.csv"
    df.to_csv(out_path, index=False)

    return df

if __name__ == "__main__":
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    results = load_and_apply_flux_model(
        root=root,
        run_name="test_scalar_potential",
    )
