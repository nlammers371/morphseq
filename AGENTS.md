# Project Agent Rules (`morphseq`)

## Scope
- These rules apply to this repository only: `/net/trapnell/vol1/home/mdcolon/proj/morphseq`.

## Python And Conda
- Do not run `conda activate` in commands.
- Do not use bare `python`, `python3`, `pip`, or `pip3`.
- Always use these absolute executables:
  - `PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python`
  - `PIP=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/pip`
- Run Python scripts with:
  - `"$PYTHON" path/to/script.py`
- Run module commands with:
  - `"$PYTHON" -m <module> ...`
- Install packages with:
  - `"$PIP" install ...`

## Sanity Check Before Python Work
- Confirm interpreter path before running substantial Python tasks:
  - `"$PYTHON" -c 'import sys; print(sys.executable)'`
