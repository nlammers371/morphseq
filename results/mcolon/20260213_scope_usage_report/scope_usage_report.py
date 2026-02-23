"""
Scope Time & Embryo Usage Report from Build06 Output
=====================================================
Summarizes microscope usage across all processed experiments:
scope hours, embryo counts, and developmental coverage.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[3] / "morphseq_playground" / "metadata" / "build06_output"
OUT_DIR = Path(__file__).resolve().parent

NEEDED_COLS = [
    "embryo_id", "experiment_id", "experiment_date",
    "raw_time_s", "predicted_stage_hpf", "use_embryo_flag",
]

# Cost calculation (microscope usage)
COST_PER_HOUR = 24.0  # $ per hour

# ── Step 1: Load & Aggregate ──────────────────────────────────────────
records = []
for csv_path in sorted(DATA_DIR.glob("df03_final_output_with_latents_*.csv")):
    # Skip backup / archive files
    if ".backup" in csv_path.name or ".archive" in csv_path.name:
        continue

    # Extract date string from filename (YYYYMMDD)
    m = re.search(r"(\d{8})", csv_path.name)
    if m is None:
        continue
    file_date = m.group(1)

    df = pd.read_csv(csv_path, usecols=NEEDED_COLS)

    # Scope hours: wall-clock span of raw_time_s across entire file
    scope_hours = (df["raw_time_s"].max() - df["raw_time_s"].min()) / 3600

    # Per-embryo hpf coverage
    embryo_hpf = (
        df.groupby("embryo_id")["predicted_stage_hpf"]
        .agg(lambda s: s.max() - s.min())
    )
    n_embryos = embryo_hpf.shape[0]
    total_embryo_hours = embryo_hpf.sum()
    mean_embryo_hpf_range = embryo_hpf.mean()

    # Use experiment_id from data if available, else filename date
    exp_id = df["experiment_id"].iloc[0] if "experiment_id" in df.columns else file_date

    # Parse date for proper formatting
    year = int(file_date[:4])
    month = int(file_date[4:6])
    quarter = (month - 1) // 3 + 1

    records.append({
        "experiment_id": exp_id,
        "date": file_date,
        "year": year,
        "month": month,
        "quarter": quarter,
        "year_quarter": f"{year} Q{quarter}",
        "scope_hours": round(scope_hours, 2),
        "n_embryos": n_embryos,
        "mean_embryo_hpf_range": round(mean_embryo_hpf_range, 2),
        "total_embryo_hours": round(total_embryo_hours, 2),
        "source_file": csv_path.name,
    })

exp_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

# ── Step 2: Summary Tables ────────────────────────────────────────────
# Experiment-level
exp_df.to_csv(OUT_DIR / "experiment_summary.csv", index=False)

# Add cost calculations to experiment-level data
exp_df["scope_cost_usd"] = exp_df["scope_hours"] * COST_PER_HOUR

# Monthly summary with English date labels
month_names = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

monthly = (
    exp_df.groupby(["year", "month"])
    .agg(
        n_experiments=("experiment_id", "count"),
        total_scope_hours=("scope_hours", "sum"),
        total_scope_cost_usd=("scope_cost_usd", "sum"),
        total_embryos=("n_embryos", "sum"),
        total_embryo_hours=("total_embryo_hours", "sum"),
    )
    .reset_index()
)
monthly["month_name"] = monthly["month"].map(month_names)
monthly["year_month_label"] = monthly["year"].astype(str) + " " + monthly["month_name"]
monthly = monthly[["year", "month", "month_name", "year_month_label",
                   "n_experiments", "total_scope_hours", "total_scope_cost_usd",
                   "total_embryos", "total_embryo_hours"]]
monthly.to_csv(OUT_DIR / "monthly_summary.csv", index=False)

# Quarterly summary
quarterly = (
    exp_df.groupby("year_quarter")
    .agg(
        n_experiments=("experiment_id", "count"),
        total_scope_hours=("scope_hours", "sum"),
        total_scope_cost_usd=("scope_cost_usd", "sum"),
        total_embryos=("n_embryos", "sum"),
        total_embryo_hours=("total_embryo_hours", "sum"),
    )
    .reset_index()
)
quarterly.to_csv(OUT_DIR / "quarterly_summary.csv", index=False)

# Grand totals
total_scope_hours = exp_df['scope_hours'].sum()
total_scope_cost = exp_df['scope_cost_usd'].sum()
print("=" * 60)
print("SCOPE USAGE REPORT — Grand Totals")
print("=" * 60)
print(f"  Experiments processed : {len(exp_df)}")
print(f"  Total scope hours    : {total_scope_hours:.1f} h")
print(f"  Total scope cost     : ${total_scope_cost:,.2f} (at ${COST_PER_HOUR}/h)")
print(f"  Total embryos        : {exp_df['n_embryos'].sum()}")
print(f"  Total embryo-hours   : {exp_df['total_embryo_hours'].sum():.1f} h")
print(f"  Date range           : {exp_df['date'].min()} – {exp_df['date'].max()}")
print("=" * 60)
print()
print("Per-experiment summary:")
print(exp_df[["experiment_id", "date", "scope_hours", "n_embryos",
              "mean_embryo_hpf_range", "total_embryo_hours"]].to_string(index=False))

# ── Step 3: Plots ─────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

# --- Yearly summaries ---
yearly = exp_df.groupby("year").agg(
    total_scope_hours=("scope_hours", "sum"),
    total_embryo_hours=("total_embryo_hours", "sum"),
    total_embryos=("n_embryos", "sum"),
).reset_index()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(yearly["year"].astype(str), yearly["total_scope_hours"], color="#4C72B0")
ax.set_xlabel("Year")
ax.set_ylabel("Total Scope Hours")
ax.set_title("Total Scope Hours per Year")
for i, v in enumerate(yearly["total_scope_hours"]):
    ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "yearly_scope_hours.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(yearly["year"].astype(str), yearly["total_embryo_hours"], color="#DD8452")
ax.set_xlabel("Year")
ax.set_ylabel("Total Embryo-Hours")
ax.set_title("Total Embryo-Hours per Year")
for i, v in enumerate(yearly["total_embryo_hours"]):
    ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "yearly_embryo_hours.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(yearly["year"].astype(str), yearly["total_embryos"], color="#55A868")
ax.set_xlabel("Year")
ax.set_ylabel("Total Embryos")
ax.set_title("Total Embryos per Year")
for i, v in enumerate(yearly["total_embryos"]):
    ax.text(i, v + 1, f"{int(v)}", ha="center", fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "yearly_embryos.png", dpi=150)
plt.close(fig)

# --- Monthly breakdown ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(monthly)), monthly["total_scope_hours"], color="#4C72B0")
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly["year_month_label"], rotation=60, ha="right", fontsize=8)
ax.set_xlabel("Month")
ax.set_ylabel("Scope Hours")
ax.set_title("Scope Hours by Month")
plt.tight_layout()
fig.savefig(OUT_DIR / "monthly_scope_hours.png", dpi=150)
plt.close(fig)

# Embryos per experiment (sorted by date), with mean line
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(exp_df)), exp_df["n_embryos"], color="#55A868")
ax.axhline(exp_df["n_embryos"].mean(), color="red", linestyle="--", label=f'Mean = {exp_df["n_embryos"].mean():.0f}')
ax.set_xticks(range(len(exp_df)))
ax.set_xticklabels(exp_df["date"], rotation=60, ha="right", fontsize=7)
ax.set_xlabel("Experiment (by date)")
ax.set_ylabel("Number of Embryos")
ax.set_title("Embryos per Experiment")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "embryos_per_experiment.png", dpi=150)
plt.close(fig)

# Total embryo-hours by month
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(monthly)), monthly["total_embryo_hours"], color="#DD8452")
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly["year_month_label"], rotation=60, ha="right", fontsize=8)
ax.set_xlabel("Month")
ax.set_ylabel("Embryo-Hours")
ax.set_title("Total Embryo-Hours by Month")
plt.tight_layout()
fig.savefig(OUT_DIR / "monthly_embryo_hours.png", dpi=150)
plt.close(fig)

# --- Quarterly breakdown ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(quarterly)), quarterly["total_scope_hours"], color="#4C72B0")
ax.set_xticks(range(len(quarterly)))
ax.set_xticklabels(quarterly["year_quarter"], rotation=45, ha="right")
ax.set_xlabel("Quarter")
ax.set_ylabel("Scope Hours")
ax.set_title("Scope Hours by Quarter")
for i, v in enumerate(quarterly["total_scope_hours"]):
    ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "quarterly_scope_hours.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(quarterly)), quarterly["total_embryo_hours"], color="#DD8452")
ax.set_xticks(range(len(quarterly)))
ax.set_xticklabels(quarterly["year_quarter"], rotation=45, ha="right")
ax.set_xlabel("Quarter")
ax.set_ylabel("Embryo-Hours")
ax.set_title("Total Embryo-Hours by Quarter")
for i, v in enumerate(quarterly["total_embryo_hours"]):
    ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "quarterly_embryo_hours.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(quarterly)), quarterly["total_embryos"], color="#55A868")
ax.set_xticks(range(len(quarterly)))
ax.set_xticklabels(quarterly["year_quarter"], rotation=45, ha="right")
ax.set_xlabel("Quarter")
ax.set_ylabel("Total Embryos")
ax.set_title("Total Embryos by Quarter")
for i, v in enumerate(quarterly["total_embryos"]):
    ax.text(i, v + 1, f"{int(v)}", ha="center", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "quarterly_embryos.png", dpi=150)
plt.close(fig)

# --- Cost plots (at $24/hour) ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(quarterly)), quarterly["total_scope_cost_usd"], color="#C44E52")
ax.set_xticks(range(len(quarterly)))
ax.set_xticklabels(quarterly["year_quarter"], rotation=45, ha="right")
ax.set_xlabel("Quarter")
ax.set_ylabel("Cost (USD)")
ax.set_title(f"Scope Usage Cost by Quarter (${COST_PER_HOUR}/hour)")
for i, v in enumerate(quarterly["total_scope_cost_usd"]):
    ax.text(i, v + 50, f"${v:,.0f}", ha="center", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "quarterly_scope_cost.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(monthly)), monthly["total_scope_cost_usd"], color="#C44E52")
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly["year_month_label"], rotation=60, ha="right", fontsize=8)
ax.set_xlabel("Month")
ax.set_ylabel("Cost (USD)")
ax.set_title(f"Scope Usage Cost by Month (${COST_PER_HOUR}/hour)")
plt.tight_layout()
fig.savefig(OUT_DIR / "monthly_scope_cost.png", dpi=150)
plt.close(fig)

# Yearly cost
yearly["total_scope_cost_usd"] = yearly["total_scope_hours"] * COST_PER_HOUR
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(yearly["year"].astype(str), yearly["total_scope_cost_usd"], color="#C44E52")
ax.set_xlabel("Year")
ax.set_ylabel("Cost (USD)")
ax.set_title(f"Scope Usage Cost per Year (${COST_PER_HOUR}/hour)")
for i, v in enumerate(yearly["total_scope_cost_usd"]):
    ax.text(i, v + 50, f"${v:,.0f}", ha="center", fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "yearly_scope_cost.png", dpi=150)
plt.close(fig)

print(f"\nOutputs saved to: {OUT_DIR}")
print("  - experiment_summary.csv")
print("  - monthly_summary.csv")
print("  - quarterly_summary.csv")
print("  - yearly_scope_hours.png")
print("  - yearly_embryo_hours.png")
print("  - yearly_embryos.png")
print("  - yearly_scope_cost.png")
print("  - monthly_scope_hours.png")
print("  - monthly_embryo_hours.png")
print("  - monthly_scope_cost.png")
print("  - quarterly_scope_hours.png")
print("  - quarterly_embryo_hours.png")
print("  - quarterly_embryos.png")
print("  - quarterly_scope_cost.png")
print("  - embryos_per_experiment.png")
