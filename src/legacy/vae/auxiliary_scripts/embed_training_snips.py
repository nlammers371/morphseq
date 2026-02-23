from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _list_snip_images(train_images_dir: Path) -> List[Path]:
    if not train_images_dir.exists():
        return []
    # Walk all label folders
    return sorted([p for p in train_images_dir.rglob("*.jpg")])


def _simulate_embeddings(n: int, latent_dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, latent_dim)).astype(np.float32)


def embed_snips(
    root: str | Path,
    train_name: str,
    model_dir: Optional[str | Path] = None,
    out_csv: Optional[str | Path] = None,
    batch_size: int = 64,
    simulate: bool = False,
    latent_dim: int = 16,
    seed: int = 0,
) -> Path:
    """Generate morphological embeddings for training snips.

    - When simulate=True, produces deterministic random embeddings (no torch dependency),
      useful for CI and structure validation.
    - When simulate=False, requires torch/torchvision and a `pythae`-compatible model folder.

    Writes a CSV with columns: snip_id, z_mu_00..z_mu_{latent_dim-1} under the train folder.
    """
    root = Path(root)
    train_root = root / "training_data" / train_name
    images_root = train_root / "images"
    snip_paths = _list_snip_images(images_root)
    if not snip_paths:
        raise FileNotFoundError(f"No images found under {images_root}")

    snip_ids = [p.stem for p in snip_paths]

    if simulate:
        Z = _simulate_embeddings(len(snip_paths), latent_dim=latent_dim, seed=seed)
    else:
        # Lazy imports to avoid hard deps in simulate mode
        try:
            import torch
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms
            from pythae.data.datasets import collate_dataset_output
        except Exception as e:
            raise RuntimeError("PyTorch/pythae/torchvision required for real embeddings. Use --simulate or install deps.") from e

        from src.legacy.vae.models.auto_model import AutoModel

        # Model folder: allow passing either the `final_model` folder or its parent
        model_dir = Path(model_dir) if model_dir is not None else None
        if model_dir is None:
            raise ValueError("model_dir is required when simulate=False")
        if (model_dir / "model_config.json").exists():
            fm = model_dir
        elif (model_dir / "final_model" / "model_config.json").exists():
            fm = model_dir / "final_model"
        else:
            raise FileNotFoundError("model_config.json not found in model_dir or model_dir/final_model")

        model = AutoModel.load_from_folder(str(fm))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        ds = datasets.ImageFolder(root=str(images_root), transform=tfm)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_dataset_output)

        zs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in dl:
                x = batch["data"].to(device)
                enc = model.encoder(x)
                mu = enc.embedding.detach().cpu().numpy()
                zs.append(mu)
        Z = np.concatenate(zs, axis=0)
        latent_dim = Z.shape[1]

    cols = [f"z_mu_{i:02d}" for i in range(latent_dim)]
    df = pd.DataFrame(Z, columns=cols)
    df.insert(0, "snip_id", snip_ids)

    out_csv = Path(out_csv) if out_csv else (train_root / "embeddings.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Embed MorphSeq training snips (VAE or simulate)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--train-name", required=True)
    ap.add_argument("--model-dir", help="Path to VAE model folder (or its parent)")
    ap.add_argument("--out-csv")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    embed_snips(
        root=args.root,
        train_name=args.train_name,
        model_dir=args.model_dir,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        simulate=args.simulate,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    print("✔️  Embeddings written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

