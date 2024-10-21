# %%
from __future__ import annotations

from pathlib import Path

import jmp.config as jc
import nshutils as nu

ckpt_path = Path(
    "/net/csefiles/coc-fung-cluster/nima/shared/experiment-data/nshtrainer/3ico36yf/checkpoint/last/epoch0-step57444.ckpt"
)
assert ckpt_path.exists()

config = jc.RelaxWBMConfig.draft()
config.ckpt_path = ckpt_path
config.relaxer = jc.RelaxerConfig.draft()
config.relaxer.optimizer = "FIRE"
config.relaxer.force_max = 0.02
config.relaxer.max_steps = 500
config = config.finalize()
nu.display(config)

# %%
import nshrunner as nr
from jmp.relaxation.wbm import relax_wbm_run_fn


def update_lm_config(config: jc.Config):
    config.energy_referencer = jc.PerAtomReferencerConfig.linear_reference(
        "mptrj-salex"
    )
    return config


results_dir = Path.cwd() / f"results/{config.ckpt_path.stem}"
results_dir.mkdir(parents=True, exist_ok=True)
distributed_configs = config.subset_(10_000).distributed(1)
run_args = [(config, results_dir, update_lm_config) for config in distributed_configs]

runner = nr.Runner(relax_wbm_run_fn, nr.RunnerConfig(working_dir=Path.cwd()))
runner.local(run_args)
# runner.session(run_args, snapshot=False)
