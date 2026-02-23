#!/usr/bin/env python3
"""
Perturbation key utilities.

Commands:
  - from-legacy: Bootstrap perturbation_name_key.csv from a legacy embryo stats CSV
  - from-excel:  Convert curated Excel master to CSV (placeholder; to be implemented)

Usage examples:
  python -m src.tools.pert_key from-legacy \
      --legacy-csv <root>/morphseq_playground/embryo_stats_df_legacy.csv \
      --out <root>/metadata/perturbation_name_key.csv

  # Placeholder for future Excel support
  python -m src.tools.pert_key from-excel --excel <root>/metadata/perturbation_key_master.xlsx --out <root>/metadata/perturbation_name_key.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import csv
from collections import defaultdict, Counter


def _to_bool(val: str) -> bool:
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _mode(items: Counter) -> str:
    if not items:
        return ""
    maxc = max(items.values())
    cands = sorted([k for k, v in items.items() if v == maxc])
    return cands[0] if cands else ""


def derive_perturbation_key_from_legacy(legacy_csv_path: str, out_path: str | None = None):
    """Streaming derivation to minimize memory and avoid heavy deps."""
    with open(legacy_csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name.lower(): i for i, name in enumerate(header)}

        def has(col: str) -> bool:
            return col in idx

        short_counts: dict[str, Counter] = defaultdict(Counter)
        phen_counts: dict[str, Counter] = defaultdict(Counter)
        pert_counts: dict[str, Counter] = defaultdict(Counter)
        back_counts: dict[str, Counter] = defaultdict(Counter)
        ctrl_counts: dict[str, int] = defaultdict(int)
        total_counts: dict[str, int] = defaultdict(int)

        for row in reader:
            # short name first
            short = row[idx["short_pert_name"]].strip() if has("short_pert_name") else ""

            # master_perturbation
            if has("master_perturbation"):
                master = row[idx["master_perturbation"]].strip()
            else:
                chem = row[idx["chem_perturbation"]].strip() if has("chem_perturbation") else ""
                geno = row[idx["genotype"]].strip() if has("genotype") else ""
                if chem and chem.lower() not in ("none", "nan"):
                    master = chem
                elif geno:
                    master = geno
                elif short:
                    master = short
                else:
                    master = "unknown"
            master = master.strip() or "unknown"

            # ensure short fallback
            if not short:
                short = master
            if short:
                short_counts[master][short] += 1

            # phenotype
            phenotype = row[idx["phenotype"]].strip() if has("phenotype") else ""
            if phenotype:
                phen_counts[master][phenotype] += 1

            # control flag
            ctrl = _to_bool(row[idx["control_flag"]]) if has("control_flag") else (phenotype.lower() == "wt")
            if ctrl:
                ctrl_counts[master] += 1
            total_counts[master] += 1

            # pert type
            if has("pert_type"):
                pert_type = row[idx["pert_type"]].strip()
            else:
                m = master.lower()
                if m in {"inj-ctrl", "inj_ctrl", "control", "wt"}:
                    pert_type = "control"
                elif m in {"em", "egg water", "ew"}:
                    pert_type = "medium"
                else:
                    pert_type = ""
            if pert_type:
                pert_counts[master][pert_type] += 1

            # background
            if has("background"):
                background = row[idx["background"]].strip()
            elif has("strain"):
                background = row[idx["strain"]].strip()
            elif has("background_strain"):
                background = row[idx["background_strain"]].strip()
            else:
                background = ""
            if background:
                back_counts[master][background] += 1

    masters = sorted(total_counts.keys())
    rows = []
    for m in masters:
        short = _mode(short_counts[m]) or m
        phen = _mode(phen_counts[m]) or "unknown"
        pert = _mode(pert_counts[m]) or "unknown"
        back = _mode(back_counts[m]) or "unknown"
        ctrl = ctrl_counts[m] >= (total_counts[m] / 2)
        rows.append([m, short, phen, str(bool(ctrl)), pert, back])

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["master_perturbation", "short_pert_name", "phenotype", "control_flag", "pert_type", "background"])
            w.writerows(rows)

    return rows


def cmd_from_legacy(args: argparse.Namespace) -> int:
    legacy = Path(args.legacy_csv)
    if not legacy.exists():
        print(f"ERROR: Legacy CSV not found: {legacy}")
        return 2
    out = Path(args.out) if args.out else None
    try:
        rows = derive_perturbation_key_from_legacy(str(legacy), str(out) if out else None)
        print(f"✅ Derived key with {len(rows)} rows")
        if out:
            print(f"📝 Wrote: {out}")
        else:
            header = ["master_perturbation", "short_pert_name", "phenotype", "control_flag", "pert_type", "background"]
            print(",".join(header))
            for r in rows[:10]:
                print(",".join(r))
        return 0
    except Exception as e:
        print(f"ERROR: Failed to derive key: {e}")
        return 1


def cmd_from_excel(args: argparse.Namespace) -> int:
    # Placeholder for future implementation
    print("from-excel is not yet implemented. Use from-legacy for bootstrapping.")
    return 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pert-key", description="Perturbation key utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("from-legacy", help="Bootstrap key from legacy embryo stats CSV")
    pl.add_argument("--legacy-csv", required=True, help="Path to legacy embryo stats CSV")
    pl.add_argument("--out", required=False, help="Path to write perturbation_name_key.csv")

    pe = sub.add_parser("from-excel", help="Convert curated Excel master to CSV (TBD)")
    pe.add_argument("--excel", required=True, help="Path to Excel master (xlsx)")
    pe.add_argument("--sheet", default="Perturbations", help="Sheet name (default: Perturbations)")
    pe.add_argument("--out", required=False, help="Path to write perturbation_name_key.csv")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "from-legacy":
        return cmd_from_legacy(args)
    elif args.cmd == "from-excel":
        return cmd_from_excel(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
