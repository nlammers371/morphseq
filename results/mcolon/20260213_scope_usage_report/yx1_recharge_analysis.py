"""
YX1 Scope Usage Report — Jan 2025 through Jan 2026
Data source: imaging_facility_yx1_recharge_2025.xlsx (from Pang)

Pricing tiers (post-Oct 2025 rates used for normalized cost):
  Regular (first 6 hr/session): $45/hr
  Extended (remainder):         $18/hr
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
OUT_DIR = HERE / "data_from_pang"
OUT_DIR.mkdir(exist_ok=True)
XL_PATH = HERE / "imaging_facility_yx1_recharge_2025.xlsx"

# Current rates for normalized cost calculation
RATE_REGULAR = 45.0   # $/hr
RATE_EXTENDED = 18.0  # $/hr

# Available scope hours per day (24 hr/day × 7 day/wk)
SCOPE_HOURS_PER_DAY = 24.0

# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
raw = pd.read_excel(XL_PATH, header=1)

# Rename columns to short names
raw.columns = [
    "quarter", "remark", "modality", "user", "machine",
    "hours", "uom", "unit_cost", "charge", "date", "advisor", "dept",
]

# Drop footer / total rows (missing hours or date)
df = raw.dropna(subset=["hours", "date"]).copy()

# Coerce types
df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
df["charge"] = pd.to_numeric(df["charge"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["hours", "date"])

# Parse quarter string → year and Q number
def parse_quarter(q):
    m = re.match(r"(\d{4})Q(\d)", str(q))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

df[["year", "qnum"]] = pd.DataFrame(
    df["quarter"].apply(parse_quarter).tolist(), index=df.index
)
df = df.dropna(subset=["year", "qnum"])
df["year"] = df["year"].astype(int)
df["qnum"] = df["qnum"].astype(int)
df["quarter_label"] = df["quarter"].str.strip()

# Normalise advisor column (strip whitespace, title-case)
df["advisor"] = df["advisor"].str.strip().str.title()

# Tag Trapnell rows
df["is_trapnell"] = df["advisor"].str.lower() == "trapnell"

# Tier tag from remark column
df["tier"] = df["remark"].str.lower().apply(
    lambda r: "extended" if "extended" in r else "regular"
)

# Normalised cost at current $45/$18 rates
df["cost_normalized"] = df.apply(
    lambda row: row["hours"] * (RATE_EXTENDED if row["tier"] == "extended" else RATE_REGULAR),
    axis=1,
)

print(f"Loaded {len(df)} billing rows")
print(f"Date range: {df['date'].min().date()} – {df['date'].max().date()}")
print(f"Unique advisors: {sorted(df['advisor'].unique())}")
print(f"Quarters: {sorted(df['quarter_label'].unique())}")

# ---------------------------------------------------------------------------
# Helper: ordered quarter labels
# ---------------------------------------------------------------------------
quarter_order = sorted(df["quarter_label"].unique(), key=lambda q: (int(q[:4]), int(q[5])))

# ---------------------------------------------------------------------------
# 2. Available hours per quarter
# ---------------------------------------------------------------------------
DATA_MAX_DATE = df["date"].max()

def days_in_quarter(year: int, q: int) -> int:
    starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    ends   = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    start = pd.Timestamp(year=year, month=starts[q][0], day=starts[q][1])
    end   = min(pd.Timestamp(year=year, month=ends[q][0], day=ends[q][1]), DATA_MAX_DATE)
    return (end - start).days + 1

quarter_meta = (
    df[["quarter_label", "year", "qnum"]]
    .drop_duplicates()
    .set_index("quarter_label")
)
available_hours = {
    ql: days_in_quarter(int(row["year"]), int(row["qnum"])) * SCOPE_HOURS_PER_DAY
    for ql, row in quarter_meta.iterrows()
}

# ---------------------------------------------------------------------------
# 3. Per-quarter aggregations
# ---------------------------------------------------------------------------
grp = df.groupby(["quarter_label", "is_trapnell"])["hours"].sum().unstack(fill_value=0)
grp.columns.name = None
if True not in grp.columns:
    grp[True] = 0
if False not in grp.columns:
    grp[False] = 0
grp = grp.reindex(quarter_order)
grp.rename(columns={True: "trapnell", False: "others"}, inplace=True)
grp["total"] = grp["trapnell"] + grp["others"]
grp["avail"] = [available_hours[q] for q in grp.index]
grp["util_trapnell_pct"] = grp["trapnell"] / grp["avail"] * 100
grp["util_others_pct"]   = grp["others"]   / grp["avail"] * 100
grp["util_total_pct"]    = grp["total"]    / grp["avail"] * 100
grp["trapnell_share_pct"] = grp["trapnell"] / grp["total"] * 100

# Cost (normalised) per quarter — Trapnell only
cost_q = (
    df[df["is_trapnell"]]
    .groupby("quarter_label")["cost_normalized"]
    .sum()
    .reindex(quarter_order, fill_value=0)
)

# Actual billed charge per quarter — Trapnell only
charge_q = (
    df[df["is_trapnell"]]
    .groupby("quarter_label")["charge"]
    .sum()
    .reindex(quarter_order, fill_value=0)
)

# Unique Trapnell users per quarter
users_q = (
    df[df["is_trapnell"]]
    .groupby("quarter_label")["user"]
    .nunique()
    .reindex(quarter_order, fill_value=0)
)

# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------
total_hours_all    = df["hours"].sum()
total_charge_all   = df["charge"].sum()
trap_hours         = df[df["is_trapnell"]]["hours"].sum()
trap_charge_actual = df[df["is_trapnell"]]["charge"].sum()
trap_cost_norm     = df[df["is_trapnell"]]["cost_normalized"].sum()
trap_users_all     = df[df["is_trapnell"]]["user"].nunique()

print("\n" + "=" * 60)
print("SUMMARY: YX1 Recharge Jan 2025 – Jan 2026")
print("=" * 60)
print(f"  Total facility hours (all labs):   {total_hours_all:,.1f} hr")
print(f"  Total billed amount (all labs):    ${total_charge_all:,.2f}")
print(f"  Trapnell hours:                    {trap_hours:,.1f} hr  ({trap_hours/total_hours_all*100:.1f}% of facility)")
print(f"  Trapnell billed (actual charges):  ${trap_charge_actual:,.2f}")
print(f"  Trapnell cost at $45/$18 rates:    ${trap_cost_norm:,.2f}")
print(f"  Unique Trapnell users (full year): {trap_users_all}")
print()

print("Quarterly breakdown (Trapnell):")
qsummary = pd.DataFrame({
    "Hours (Trap)":     grp["trapnell"],
    "Hours (All)":      grp["total"],
    "Trap share %":     grp["trapnell_share_pct"].round(1),
    "Util rate %":      grp["util_total_pct"].round(1),
    "Unique users":     users_q,
    "Cost $45/$18":     cost_q.round(2),
    "Actual charge":    charge_q.round(2),
})
print(qsummary.to_string())

# ---------------------------------------------------------------------------
# 5. Plot A — Utilization rate (% of available hours), stacked Trap + others
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(quarter_order))
width = 0.6

bars_trap  = ax.bar(x, grp["util_trapnell_pct"], width, label="Trapnell", color="#2166ac")
bars_other = ax.bar(x, grp["util_others_pct"],   width,
                    bottom=grp["util_trapnell_pct"], label="Other labs", color="#b2b2b2")

for i, q in enumerate(quarter_order):
    total_pct = grp.loc[q, "util_total_pct"]
    ax.text(i, total_pct + 0.5, f"{total_pct:.0f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(quarter_order, rotation=30, ha="right")
ax.set_ylabel("% of available hours (24 hr/day × 7 day/wk)")
ax.set_title("YX1 Utilization Rate by Quarter")
ax.set_ylim(0, max(grp["util_total_pct"].max() * 1.15, 20))
ax.legend(loc="upper left")
ax.axhline(100, color="red", lw=0.8, ls="--", label="100% capacity")
fig.tight_layout()
fig.savefig(OUT_DIR / "yx1_utilization_rate_quarterly.png", dpi=150)
plt.close(fig)
print("\nSaved: yx1_utilization_rate_quarterly.png")

# ---------------------------------------------------------------------------
# 6. Plot B — Trapnell share of billed hours (100%-stacked + line)
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))
x = np.arange(len(quarter_order))

trap_pct_vals  = grp["trapnell_share_pct"].values
others_pct_vals = 100 - trap_pct_vals

ax1.bar(x, trap_pct_vals,   width=0.6, label="Trapnell",    color="#2166ac")
ax1.bar(x, others_pct_vals, width=0.6, bottom=trap_pct_vals, label="Other labs", color="#b2b2b2")

ax1.set_xticks(x)
ax1.set_xticklabels(quarter_order, rotation=30, ha="right")
ax1.set_ylabel("Share of billed hours (%)")
ax1.set_ylim(0, 100)
ax1.set_title("YX1 Billed Hours — Trapnell vs Other Labs")
ax1.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "yx1_trapnell_share_quarterly.png", dpi=150)
plt.close(fig)
print("Saved: yx1_trapnell_share_quarterly.png")

# ---------------------------------------------------------------------------
# 7. Plot C — Trapnell cost at $45/$18 per quarter
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(quarter_order))

# Break down by tier
cost_reg_q = (
    df[df["is_trapnell"] & (df["tier"] == "regular")]
    .groupby("quarter_label")["cost_normalized"]
    .sum()
    .reindex(quarter_order, fill_value=0)
)
cost_ext_q = (
    df[df["is_trapnell"] & (df["tier"] == "extended")]
    .groupby("quarter_label")["cost_normalized"]
    .sum()
    .reindex(quarter_order, fill_value=0)
)

ax.bar(x, cost_reg_q, width=0.6, label=f"Regular (${RATE_REGULAR:.0f}/hr)", color="#2166ac")
ax.bar(x, cost_ext_q, width=0.6, bottom=cost_reg_q,
       label=f"Extended (${RATE_EXTENDED:.0f}/hr)", color="#92c5de")

for i, q in enumerate(quarter_order):
    total = cost_q[q]
    ax.text(i, total + 20, f"${total:,.0f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(quarter_order, rotation=30, ha="right")
ax.set_ylabel("Cost ($)")
ax.set_title(f"Trapnell Lab YX1 Cost by Quarter\n(at ${RATE_REGULAR:.0f}/hr regular, ${RATE_EXTENDED:.0f}/hr extended)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "yx1_trapnell_cost_quarterly.png", dpi=150)
plt.close(fig)
print("Saved: yx1_trapnell_cost_quarterly.png")

# ---------------------------------------------------------------------------
# 8. Plot D — Unique Trapnell users per quarter
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(quarter_order))
ax.bar(x, users_q.values, width=0.6, color="#4dac26")
for i, v in enumerate(users_q.values):
    ax.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(quarter_order, rotation=30, ha="right")
ax.set_ylabel("Unique users")
ax.set_title("Trapnell Lab — Unique YX1 Users per Quarter")
ax.set_ylim(0, users_q.max() + 2)
fig.tight_layout()
fig.savefig(OUT_DIR / "yx1_trapnell_unique_users.png", dpi=150)
plt.close(fig)
print("Saved: yx1_trapnell_unique_users.png")

# ---------------------------------------------------------------------------
# 9. Yearly plots (A–D equivalents)
# ---------------------------------------------------------------------------
year_order = sorted(df["year"].unique())
year_labels = [str(y) for y in year_order]

# Available hours per year
def days_in_year(year: int) -> int:
    end = min(pd.Timestamp(year=year, month=12, day=31), DATA_MAX_DATE)
    return (end - pd.Timestamp(year=year, month=1, day=1)).days + 1

avail_hours_yr = {y: days_in_year(y) * SCOPE_HOURS_PER_DAY for y in year_order}

grp_yr = df.groupby(["year", "is_trapnell"])["hours"].sum().unstack(fill_value=0)
grp_yr.columns.name = None
if True not in grp_yr.columns:
    grp_yr[True] = 0
if False not in grp_yr.columns:
    grp_yr[False] = 0
grp_yr = grp_yr.reindex(year_order)
grp_yr.rename(columns={True: "trapnell", False: "others"}, inplace=True)
grp_yr["total"] = grp_yr["trapnell"] + grp_yr["others"]
grp_yr["avail"] = [avail_hours_yr[y] for y in grp_yr.index]
grp_yr["util_trapnell_pct"] = grp_yr["trapnell"] / grp_yr["avail"] * 100
grp_yr["util_others_pct"]   = grp_yr["others"]   / grp_yr["avail"] * 100
grp_yr["util_total_pct"]    = grp_yr["total"]    / grp_yr["avail"] * 100
grp_yr["trapnell_share_pct"] = grp_yr["trapnell"] / grp_yr["total"] * 100

cost_yr = (
    df[df["is_trapnell"]]
    .groupby("year")["cost_normalized"]
    .sum()
    .reindex(year_order, fill_value=0)
)
cost_reg_yr = (
    df[df["is_trapnell"] & (df["tier"] == "regular")]
    .groupby("year")["cost_normalized"]
    .sum()
    .reindex(year_order, fill_value=0)
)
cost_ext_yr = (
    df[df["is_trapnell"] & (df["tier"] == "extended")]
    .groupby("year")["cost_normalized"]
    .sum()
    .reindex(year_order, fill_value=0)
)
users_yr = (
    df[df["is_trapnell"]]
    .groupby("year")["user"]
    .nunique()
    .reindex(year_order, fill_value=0)
)

x_yr = np.arange(len(year_order))

# Plot A yearly — utilization rate
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x_yr, grp_yr["util_trapnell_pct"], width=0.5, label="Trapnell", color="#2166ac")
ax.bar(x_yr, grp_yr["util_others_pct"],   width=0.5,
       bottom=grp_yr["util_trapnell_pct"], label="Other labs", color="#b2b2b2")
for i, y in enumerate(year_order):
    total_pct = grp_yr.loc[y, "util_total_pct"]
    ax.text(i, total_pct + 0.3, f"{total_pct:.0f}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x_yr)
ax.set_xticklabels(year_labels)
ax.set_ylabel("% of available hours (24 hr/day × 7 day/wk)")
ax.set_title("YX1 Utilization Rate by Year")
ax.set_ylim(0, max(grp_yr["util_total_pct"].max() * 1.15, 20))
ax.axhline(100, color="red", lw=0.8, ls="--")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "yx1_utilization_rate_yearly.png", dpi=150)
plt.close(fig)
print("Saved: yx1_utilization_rate_yearly.png")

# Plot B yearly — Trapnell share (100%-stacked, no line)
fig, ax = plt.subplots(figsize=(7, 5))
trap_pct_yr   = grp_yr["trapnell_share_pct"].values
others_pct_yr = 100 - trap_pct_yr
ax.bar(x_yr, trap_pct_yr,   width=0.5, label="Trapnell",    color="#2166ac")
ax.bar(x_yr, others_pct_yr, width=0.5, bottom=trap_pct_yr, label="Other labs", color="#b2b2b2")
ax.set_xticks(x_yr)
ax.set_xticklabels(year_labels)
ax.set_ylabel("Share of billed hours (%)")
ax.set_ylim(0, 100)
ax.set_title("YX1 Billed Hours — Trapnell vs Other Labs (Yearly)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "yx1_trapnell_share_yearly.png", dpi=150)
plt.close(fig)
print("Saved: yx1_trapnell_share_yearly.png")

# Plot C yearly — Trapnell cost
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x_yr, cost_reg_yr, width=0.5, label=f"Regular (${RATE_REGULAR:.0f}/hr)", color="#2166ac")
ax.bar(x_yr, cost_ext_yr, width=0.5, bottom=cost_reg_yr,
       label=f"Extended (${RATE_EXTENDED:.0f}/hr)", color="#92c5de")
for i, y in enumerate(year_order):
    total = cost_yr[y]
    ax.text(i, total + 50, f"${total:,.0f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x_yr)
ax.set_xticklabels(year_labels)
ax.set_ylabel("Cost ($)")
ax.set_title(f"Trapnell Lab YX1 Cost by Year\n(at ${RATE_REGULAR:.0f}/hr regular, ${RATE_EXTENDED:.0f}/hr extended)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "yx1_trapnell_cost_yearly.png", dpi=150)
plt.close(fig)
print("Saved: yx1_trapnell_cost_yearly.png")

# Plot D yearly — unique users
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x_yr, users_yr.values, width=0.5, color="#4dac26")
for i, v in enumerate(users_yr.values):
    ax.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=10)
ax.set_xticks(x_yr)
ax.set_xticklabels(year_labels)
ax.set_ylabel("Unique users")
ax.set_title("Trapnell Lab — Unique YX1 Users per Year")
ax.set_ylim(0, users_yr.max() + 2)
fig.tight_layout()
fig.savefig(OUT_DIR / "yx1_trapnell_unique_users_yearly.png", dpi=150)
plt.close(fig)
print("Saved: yx1_trapnell_unique_users_yearly.png")

# ---------------------------------------------------------------------------
# 10. Compact plots (report-ready: smaller figure, larger annotation font)
# ---------------------------------------------------------------------------
CDIR = HERE / "compact_plots"
CDIR.mkdir(exist_ok=True)

UTIL_YLABEL = "% of available hours\n(24 hr/day × 7 day/wk)"
ANN_FS = 11   # annotation font size
TICK_FS = 10  # tick label font size

# --- Compact A quarterly ---
fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.arange(len(quarter_order))
ax.bar(x, grp["util_trapnell_pct"], 0.6, label="Trapnell", color="#2166ac")
ax.bar(x, grp["util_others_pct"],   0.6, bottom=grp["util_trapnell_pct"],
       label="Other labs", color="#b2b2b2")
for i, q in enumerate(quarter_order):
    tp = grp.loc[q, "util_total_pct"]
    ax.text(i, tp + 0.4, f"{tp:.0f}%", ha="center", va="bottom", fontsize=ANN_FS, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(quarter_order, rotation=30, ha="right", fontsize=TICK_FS)
ax.set_ylabel(UTIL_YLABEL); ax.set_title("YX1 Utilization Rate by Quarter")
ax.set_ylim(0, max(grp["util_total_pct"].max() * 1.18, 20))
ax.axhline(100, color="red", lw=0.8, ls="--"); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(CDIR / "yx1_utilization_rate_quarterly.png", dpi=150); plt.close(fig)

# --- Compact B quarterly ---
fig, ax = plt.subplots(figsize=(6, 3.5))
trap_pct_vals   = grp["trapnell_share_pct"].values
others_pct_vals = 100 - trap_pct_vals
ax.bar(x, trap_pct_vals,   0.6, label="Trapnell",    color="#2166ac")
ax.bar(x, others_pct_vals, 0.6, bottom=trap_pct_vals, label="Other labs", color="#b2b2b2")
ax.set_xticks(x); ax.set_xticklabels(quarter_order, rotation=30, ha="right", fontsize=TICK_FS)
ax.set_ylabel("Share of billed hours (%)"); ax.set_title("YX1 Billed Hours — Trapnell vs Other Labs")
ax.set_ylim(0, 100); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(CDIR / "yx1_trapnell_share_quarterly.png", dpi=150); plt.close(fig)

# --- Compact C quarterly ---
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(x, cost_reg_q, 0.6, label=f"Regular (${RATE_REGULAR:.0f}/hr)", color="#2166ac")
ax.bar(x, cost_ext_q, 0.6, bottom=cost_reg_q, label=f"Extended (${RATE_EXTENDED:.0f}/hr)", color="#92c5de")
for i, q in enumerate(quarter_order):
    total = cost_q[q]
    ax.text(i, total + 20, f"${total:,.0f}", ha="center", va="bottom", fontsize=ANN_FS, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(quarter_order, rotation=30, ha="right", fontsize=TICK_FS)
ax.set_ylabel("Cost ($)")
ax.set_title(f"Trapnell YX1 Cost by Quarter\n(${RATE_REGULAR:.0f}/hr regular, ${RATE_EXTENDED:.0f}/hr extended)")
ax.set_ylim(0, cost_q.max() * 1.25)
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(CDIR / "yx1_trapnell_cost_quarterly.png", dpi=150); plt.close(fig)

# --- Compact D quarterly ---
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(x, users_q.values, 0.6, color="#4dac26")
for i, v in enumerate(users_q.values):
    ax.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=ANN_FS, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(quarter_order, rotation=30, ha="right", fontsize=TICK_FS)
ax.set_ylabel("Unique users"); ax.set_title("Trapnell Lab — Unique YX1 Users per Quarter")
ax.set_ylim(0, users_q.max() + 2)
fig.tight_layout(); fig.savefig(CDIR / "yx1_trapnell_unique_users.png", dpi=150); plt.close(fig)

# --- Compact A yearly ---
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(x_yr, grp_yr["util_trapnell_pct"], 0.5, label="Trapnell", color="#2166ac")
ax.bar(x_yr, grp_yr["util_others_pct"],   0.5, bottom=grp_yr["util_trapnell_pct"],
       label="Other labs", color="#b2b2b2")
for i, y in enumerate(year_order):
    tp = grp_yr.loc[y, "util_total_pct"]
    ax.text(i, tp + 0.3, f"{tp:.0f}%", ha="center", va="bottom", fontsize=ANN_FS, fontweight="bold")
ax.set_xticks(x_yr); ax.set_xticklabels(year_labels, fontsize=TICK_FS)
ax.set_ylabel(UTIL_YLABEL); ax.set_title("YX1 Utilization Rate by Year")
ax.set_ylim(0, max(grp_yr["util_total_pct"].max() * 1.18, 20))
ax.axhline(100, color="red", lw=0.8, ls="--"); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(CDIR / "yx1_utilization_rate_yearly.png", dpi=150); plt.close(fig)

# --- Compact B yearly ---
fig, ax = plt.subplots(figsize=(5, 3.5))
trap_pct_yr   = grp_yr["trapnell_share_pct"].values
others_pct_yr = 100 - trap_pct_yr
ax.bar(x_yr, trap_pct_yr,   0.5, label="Trapnell",    color="#2166ac")
ax.bar(x_yr, others_pct_yr, 0.5, bottom=trap_pct_yr, label="Other labs", color="#b2b2b2")
ax.set_xticks(x_yr); ax.set_xticklabels(year_labels, fontsize=TICK_FS)
ax.set_ylabel("Share of billed hours (%)"); ax.set_title("YX1 Billed Hours — Trapnell vs Other Labs (Yearly)")
ax.set_ylim(0, 100); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(CDIR / "yx1_trapnell_share_yearly.png", dpi=150); plt.close(fig)

# --- Compact C yearly ---
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(x_yr, cost_reg_yr, 0.5, label=f"Regular (${RATE_REGULAR:.0f}/hr)", color="#2166ac")
ax.bar(x_yr, cost_ext_yr, 0.5, bottom=cost_reg_yr, label=f"Extended (${RATE_EXTENDED:.0f}/hr)", color="#92c5de")
for i, y in enumerate(year_order):
    total = cost_yr[y]
    ax.text(i, total + 50, f"${total:,.0f}", ha="center", va="bottom", fontsize=ANN_FS, fontweight="bold")
ax.set_xticks(x_yr); ax.set_xticklabels(year_labels, fontsize=TICK_FS)
ax.set_ylabel("Cost ($)")
ax.set_title(f"Trapnell YX1 Cost by Year\n(${RATE_REGULAR:.0f}/hr regular, ${RATE_EXTENDED:.0f}/hr extended)")
ax.set_ylim(0, cost_yr.max() * 1.25)
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(CDIR / "yx1_trapnell_cost_yearly.png", dpi=150); plt.close(fig)

# --- Compact D yearly ---
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(x_yr, users_yr.values, 0.5, color="#4dac26")
for i, v in enumerate(users_yr.values):
    ax.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=ANN_FS, fontweight="bold")
ax.set_xticks(x_yr); ax.set_xticklabels(year_labels, fontsize=TICK_FS)
ax.set_ylabel("Unique users"); ax.set_title("Trapnell Lab — Unique YX1 Users per Year")
ax.set_ylim(0, users_yr.max() + 2)
fig.tight_layout(); fig.savefig(CDIR / "yx1_trapnell_unique_users_yearly.png", dpi=150); plt.close(fig)

print("\nSaved 8 compact plots to compact_plots/")

# ---------------------------------------------------------------------------
# 11. All Trapnell users (full period)
# ---------------------------------------------------------------------------
all_trap_users = sorted(df[df["is_trapnell"]]["user"].dropna().unique())
print(f"\nAll Trapnell users ({len(all_trap_users)}):")
for u in all_trap_users:
    hrs = df[(df["is_trapnell"]) & (df["user"] == u)]["hours"].sum()
    print(f"  {u:<30s}  {hrs:6.1f} hr")
