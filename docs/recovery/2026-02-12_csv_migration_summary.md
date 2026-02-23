# CSV Migration Summary (morphseq_CORRUPT_OLD -> morphseq)

Date: 2026-02-12
Method: `rsync` copy from old `results/` to new `results/` with `--ignore-existing` and CSV-only filters.

## Copy Behavior

- Copied only missing CSV files.
- Existing CSVs in destination were not overwritten.

## Counts

- Missing CSVs before copy: 1826
- CSVs copied: 1826
- Remaining missing CSVs after copy: 0
- CSVs still differing between old/new (same path exists in both): 114

## Verified Key Files Present

- `results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv`
- `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`

## Commands Used

```bash
rsync -ai --ignore-existing \
  --include='*/' --include='*.csv' --exclude='*' \
  /net/trapnell/vol1/home/mdcolon/proj/morphseq_CORRUPT_OLD/results/ \
  /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
```

```bash
# Remaining missing CSV check
rsync -ani --ignore-existing \
  --include='*/' --include='*.csv' --exclude='*' \
  /net/trapnell/vol1/home/mdcolon/proj/morphseq_CORRUPT_OLD/results/ \
  /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
```
