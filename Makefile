# Reproduce everything:  make setup data preprocess test pilot all
# On Windows use Git Bash (or WSL); PY points at the project virtual environment.
PY ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python,.venv/bin/python)
RUN = $(PY) scripts/run_experiment.py

.PHONY: setup data preprocess datacard test lint pilot e1 e2 e3 e4 e5 e6 figures tables all clean-results

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -I scripts/download_data.py --out data/raw

preprocess:
	$(PY) scripts/preprocess_all.py --jobs 4

datacard:
	$(PY) scripts/make_data_card.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

pilot:
	$(PY) scripts/time_pilot.py

# E1 runs in two parts that can share the GPU; analysis merges completed E1 runs.
e1:
	$(RUN) experiment=e1_splits 'experiment.models=[template,ridge,riemann,forcemap,mlp]' 'hydra.run.dir=results/e1_splits/$${now:%Y%m%d-%H%M%S}_main'
	$(RUN) experiment=e1_splits 'experiment.models=[eegnet]' 'hydra.run.dir=results/e1_splits/$${now:%Y%m%d-%H%M%S}_eegnet'

e2:
	$(RUN) experiment=e2_selection

e3:
	$(RUN) experiment=e3_chance

e4:
	$(RUN) experiment=e4_onset

e5:
	$(RUN) experiment=e5_artifacts

e6:
	$(RUN) experiment=e6_blocks

figures:
	$(PY) scripts/make_figures.py

tables:
	$(PY) scripts/make_tables.py

all: test e1 e2 e3 e4 e5 e6 e7 figures tables

e7:
	$(RUN) experiment=e7_residual
