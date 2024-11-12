# %%
from __future__ import annotations

from pathlib import Path

import nshconfig_extra as CE
import nshutils as nu

import jmp.configs as jc

# ckpt_path = CE.CachedPath(
#     uri="hf://nimashoghi/mptrj-alex-omat24-jmp-s-mptrj-salex-finetune-n727oywf/checkpoints/last/epoch7-step143497.ckpt"
# )
ckpt_path = CE.CachedPath(
    uri="hf://nimashoghi/mptrj-alex-omat24-jmp-s-t7colwek/checkpoints/last/epoch2-step215414.ckpt"
)

config = jc.RelaxWBMConfig.draft()
config.ckpt_path = ckpt_path
config.relaxer = jc.RelaxerConfig.draft()
config.relaxer.optimizer = "FIRE"
config.relaxer.force_max = 0.02
config.relaxer.max_steps = 500
config.relaxer.cell_filter = "exp"
config = config.finalize()
nu.display(config)

# %%
import nshrunner as nr

from jmp.relaxation.wbm import relax_wbm_run_fn


def update_lm_config(config: jc.Config):
    config.pretrained_ckpt = CE.CachedPath(uri="/mnt/shared/checkpoints/jmp-s.pt")
    config.energy_referencer = jc.PerAtomReferencerConfig.linear_reference(
        "mptrj-salex"
    )
    return config


results_dir = Path.cwd() / f"results/{Path(config.ckpt_path.uri).stem}pt"
results_dir.mkdir(parents=True, exist_ok=True)
distributed_configs = config.subset_(1_000).distributed(1)
run_args = [(config, results_dir, update_lm_config) for config in distributed_configs]

runner = nr.Runner(relax_wbm_run_fn, nr.RunnerConfig(working_dir=Path.cwd()))
runner.local(run_args)
# runner.session(run_args, snapshot=False)
