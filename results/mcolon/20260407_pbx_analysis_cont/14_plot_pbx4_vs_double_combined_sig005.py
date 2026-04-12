from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from analyze.classification.viz.auroc_over_time import plot_aurocs_over_time

csv_path = SCRIPT_DIR / "results" / "classification" / "pbx4_vs_double_by_experiment_perm500" / "classification_auroc_combined_all_pbx4_crispant_vs_pbx1b_pbx4_crispant.csv"
out_base = SCRIPT_DIR / "figures" / "classification" / "pbx4_vs_double_by_experiment_perm500" / "classification_over_time_combined_all_pbx4_crispant_vs_pbx1b_pbx4_crispant_p005"

df = pd.read_csv(csv_path)
plot_aurocs_over_time(
    df.assign(positive_label="pbx1b_pbx4_crispant"),
    curve_col="positive_label",
    backend="both",
    show_null_band=True,
    show_significance=True,
    sig_threshold=0.05,
    title="Classification over time: pbx4 crispant vs pbx1b pbx4 crispant (combined_all, p <= 0.05)",
    output_path=out_base,
)
print(out_base.with_suffix(".png"))
print(out_base.with_suffix(".html"))
