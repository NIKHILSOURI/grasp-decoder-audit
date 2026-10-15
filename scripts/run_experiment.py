"""Run one experiment and write everything needed to reproduce and audit it.

    python scripts/run_experiment.py experiment=e1_splits [seed=0] [subjects=[1,2]]

Outputs in results/<experiment>/<timestamp>/: resolved config (config.yaml), metrics.json
(results + seed + git hash + library versions + GPU), timing.json, per_subject.csv, folds.csv
and experiment-specific files.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gaudit.experiments import EXPERIMENTS
from gaudit.utils.io import env_info, git_hash, write_json
from gaudit.utils.logging import get_logger
from gaudit.utils.seed import seed_everything
from gaudit.utils.timing import Timer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out = Path(HydraConfig.get().runtime.output_dir)
    log = get_logger(file=str(out / "run.log"))
    OmegaConf.set_struct(cfg, False)
    cfg.model_kw = {p.stem: OmegaConf.to_container(OmegaConf.load(p).kw)
                    for p in sorted((ROOT / "configs" / "model").glob("*.yaml"))}
    seed_everything(cfg.seed)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    name = cfg.experiment.name
    log.info(f"running {name} -> {out}")
    timer = Timer()
    t0 = time.time()
    with timer("total"):
        res = EXPERIMENTS[name](cfg, out, timer)
    write_json(out / "timing.json", {"wall_s": time.time() - t0, "stages_s": timer.t})
    write_json(out / "metrics.json", {"experiment": name, "seed": cfg.seed, "git": git_hash(),
                                      "env": env_info(), "results": res})
    log.info(f"{name} finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
