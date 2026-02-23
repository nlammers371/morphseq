#!/usr/bin/env python3
"""
Merge/enrich a perturbation key CSV from a backup key with slightly different
naming conventions.

Use token-based matching so entries like "cep290_homo" can match
"cep290_homo_cep290" from the backup and pull phenotype/control values.

Usage:
  python -m src.tools.merge_pert_key_from_backup \
    --current morphseq_playground/perturbation_name_key.csv \
    --backup morphseq_playground/perturbation_name_key_backup.csv \
    --out morphseq_playground/perturbation_name_key.csv \
    --dry-run

Defaults:
  - Requires backup tokens to be a superset of the current tokens to match
    (override with --min-coverage to allow fuzzy matches).
  - Updates fields only when current has unknown/empty (or control_flag is False
    and backup is True).
  - Fields updated: phenotype, control_flag, short_pert_name, pert_type, background.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd


def normalize_key(s: str) -> str:
    s = str(s).strip().lower()
    # unify separators
    s = s.replace("/", "_").replace(":", "_").replace("-", "_")
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def tokens(s: str) -> List[str]:
    return [t for t in normalize_key(s).split("_") if t]


@dataclass
class Match:
    idx: int
    score: float
    superset: bool


def find_best_backup_match(curr_key: str, backup_keys: List[str], min_coverage: float = 1.0) -> Tuple[int | None, float, bool]:
    """
    Return (index, score, superset_flag) of the best backup key for a given current key.
    - score: Jaccard similarity over token sets.
    - superset_flag: True if backup token set is a superset of current set.
    - min_coverage: require intersect/len(curr_tokens) >= min_coverage to accept.
    """
    curr_toks = set(tokens(curr_key))
    if not curr_toks:
        return None, 0.0, False
    best = Match(idx=-1, score=-1.0, superset=False)
    for i, bk in enumerate(backup_keys):
        bt = set(tokens(bk))
        if not bt:
            continue
        inter = len(curr_toks & bt)
        cov = inter / max(1, len(curr_toks))
        if cov < min_coverage:
            continue
        union = len(curr_toks | bt)
        jac = inter / union
        superset = curr_toks.issubset(bt)
        sc = jac + (0.1 if superset else 0.0)  # prefer supersets on tie
        if sc > best.score:
            best = Match(idx=i, score=sc, superset=superset)
    return (best.idx if best.score >= 0 else None), best.score, best.superset


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge pert key from backup")
    ap.add_argument("--current", required=True, help="Path to current key CSV")
    ap.add_argument("--backup", required=True, help="Path to backup key CSV")
    ap.add_argument("--out", required=True, help="Path to write updated key CSV")
    ap.add_argument("--min-coverage", type=float, default=1.0, help="Min coverage (intersect/len(curr_tokens)) to accept match")
    ap.add_argument("--dry-run", action="store_true", help="Print proposed changes without writing")
    args = ap.parse_args()

    curr = pd.read_csv(args.current)
    bkp = pd.read_csv(args.backup)

    # Normalize backup control_flag to bool
    bkp_ctrl = bkp["control_flag"].astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"]) if "control_flag" in bkp.columns else pd.Series([False]*len(bkp))
    bkp = bkp.assign(control_flag=bkp_ctrl)

    # Prepare backup list for matching
    bkp_keys = bkp["master_perturbation"].astype(str).tolist()

    # Track changes
    changes: List[Dict] = []

    def is_unknown(x) -> bool:
        s = str(x).strip().lower()
        return s in ("", "unknown", "nan", "none")

    for i, row in curr.iterrows():
        master = row["master_perturbation"]
        idx, score, sup = find_best_backup_match(master, bkp_keys, min_coverage=args.min_coverage)
        if idx is None:
            continue
        b = bkp.iloc[idx]

        before = row.copy()
        updated = False

        # Update phenotype
        if "phenotype" in b and is_unknown(row.get("phenotype", "")) and not is_unknown(b["phenotype"]):
            curr.at[i, "phenotype"] = b["phenotype"]
            updated = True
        # Update control_flag (only escalate False -> True)
        if b.get("control_flag", False) and not bool(row.get("control_flag", False)):
            curr.at[i, "control_flag"] = True
            updated = True
        # Update short_pert_name
        if "short_pert_name" in b and is_unknown(row.get("short_pert_name", "")) and not is_unknown(b["short_pert_name"]):
            curr.at[i, "short_pert_name"] = b["short_pert_name"]
            updated = True
        # Update pert_type
        if "pert_type" in b and is_unknown(row.get("pert_type", "")) and not is_unknown(b["pert_type"]):
            curr.at[i, "pert_type"] = b["pert_type"]
            updated = True
        # Update background
        if "background" in b and is_unknown(row.get("background", "")) and not is_unknown(b["background"]):
            curr.at[i, "background"] = b["background"]
            updated = True

        if updated:
            after = curr.loc[i]
            changes.append({
                "master": master,
                "matched_backup": b["master_perturbation"],
                "score": round(score, 3),
                "superset": sup,
                "before": before.to_dict(),
                "after": after.to_dict(),
            })

    print(f"Planned updates: {len(changes)} rows")
    # Show a few examples (including cep290_homo if present)
    for ch in changes[:10]:
        print(f"- {ch['master']}  <-  {ch['matched_backup']}  (score={ch['score']}, superset={ch['superset']})")
        b = ch["before"]; a = ch["after"]
        print(f"   phenotype: {b.get('phenotype')} -> {a.get('phenotype')}, control: {b.get('control_flag')} -> {a.get('control_flag')}")

    # Special debug print for cep290 cases
    for tag in ("cep290_homo", "cep290_het"):
        m = curr["master_perturbation"].astype(str).str.lower()
        hit = [c for c in changes if c["master"].lower() == tag]
        if hit:
            print(f"DBG: Found update for {tag}: from backup {hit[0]['matched_backup']}")

    if args.dry_run:
        print("Dry run; not writing output.")
        return 0

    curr.to_csv(args.out, index=False)
    print(f"Wrote updated key to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

