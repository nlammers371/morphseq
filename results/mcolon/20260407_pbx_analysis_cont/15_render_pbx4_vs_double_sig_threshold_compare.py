from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures" / "classification" / "pbx4_vs_double_by_experiment_perm500"

left_path = FIG_DIR / "classification_over_time_combined_all_pbx4_crispant_vs_pbx1b_pbx4_crispant.png"
right_path = FIG_DIR / "classification_over_time_combined_all_pbx4_crispant_vs_pbx1b_pbx4_crispant_p005.png"
out_path = FIG_DIR / "classification_over_time_combined_all_pbx4_vs_double_sig_compare.png"

left = mpimg.imread(left_path)
right = mpimg.imread(right_path)

fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
for ax, img, title in [
    (axes[0], left, "Significance Threshold: p <= 0.01"),
    (axes[1], right, "Significance Threshold: p <= 0.05"),
]:
    ax.imshow(img)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")

fig.suptitle("pbx4 crispant vs pbx1b pbx4 crispant (combined_all)", fontsize=20, fontweight="bold")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(out_path)
