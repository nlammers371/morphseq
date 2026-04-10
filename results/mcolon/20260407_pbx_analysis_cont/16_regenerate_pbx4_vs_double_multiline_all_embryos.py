from pathlib import Path
import importlib.util
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
module_path = SCRIPT_DIR / "13_pairwise_pbx4_vs_double_by_experiment.py"
spec = importlib.util.spec_from_file_location("pbx4_vs_double_runner", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results_dir = SCRIPT_DIR / "results" / "classification" / "pbx4_vs_double_by_experiment_perm500"
figures_dir = SCRIPT_DIR / "figures" / "classification" / "pbx4_vs_double_by_experiment_perm500"
scopes = ["20251207_pbx", "20260304", "20260306", "combined_all"]

for scope in scopes:
    stem = f"{scope}_{mod.GROUP1}_vs_{mod.GROUP2}"
    embryo_df = pd.read_csv(results_dir / f"embryo_predictions_{stem}.csv")
    pen_df = pd.read_csv(results_dir / f"embryo_penetrance_{stem}.csv")
    mod._plot_multiline(
        embryo_df,
        pen_df,
        max_embryos=-1,
        title=f"Embryo signed-margin trajectories: {mod._pretty(mod.GROUP1)} vs {mod._pretty(mod.GROUP2)} ({scope})",
        output_path=figures_dir / f"embryo_trajectories_signed_margin_{stem}.png",
    )
    print(figures_dir / f"embryo_trajectories_signed_margin_{stem}.png")
