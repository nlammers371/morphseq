# Project Agent Rules (`morphseq`)

## Scope
- These rules apply to this repository only: `/net/trapnell/vol1/home/mdcolon/proj/morphseq`.

## Python And Conda
- Do not run `conda activate` in commands.
- Do not use bare `python`, `python3`, `pip`, or `pip3`.
- Use `conda run -n segmentation_grounded_sam --no-capture-output` to run commands in the correct environment.
- Run Python scripts with:
  - `conda run -n segmentation_grounded_sam --no-capture-output python path/to/script.py`
- Run module commands with:
  - `conda run -n segmentation_grounded_sam --no-capture-output python -m <module> ...`
- Do not install packages. All dependencies are already installed.
