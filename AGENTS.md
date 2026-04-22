# Project Agent Rules (`morphseq`)

## Scope
- These rules apply only to `dev/particle_prediction/` inside this repository:
  `/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/dev/particle_prediction`.

## Python And Conda
- Do not run `conda activate` in commands.
- Do not use bare `python`, `python3`, `pip`, or `pip3`.
- Use `conda run -n morphseq-env --no-capture-output` to run commands in the correct environment.
- Run Python scripts with:
  - `conda run -n morphseq-env --no-capture-output python path/to/script.py`
- Run module commands with:
  - `conda run -n morphseq-env --no-capture-output python -m <module> ...`
- Do not install packages. All dependencies are already installed.
