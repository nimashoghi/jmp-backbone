# %%
from __future__ import annotations

from pathlib import Path

import nshconfig_extra as CE
import nshtrainer as nt
import nshutils as nu

import jmp.configs as jc

base_dir = Path("/gpfs/alpine2/proj-shared/mat273/nimashoghi/")
working_dir = base_dir / "experiment-data/"
ckpt_path = base_dir / "checkpoints/jmp-l.pt"

config = jc.Config.draft()

config.name = "jmp-l"
config.project = "mptrj-alex-omat24"

config.pretrained_ckpt = CE.CachedPath(uri=ckpt_path)
config.graph_computer = jc.GraphComputerConfig(
    cutoffs=jc.CutoffsConfig.from_constant(8.0),
    max_neighbors=jc.MaxNeighborsConfig(main=20, aeaint=20, aint=1000, qint=8),
    pbc=True,
    per_graph_radius_graph=True,
)
config.ignore_graph_generation_errors = True

# Optimization and learning rate scheduling
config.optimization = jc.OptimizationConfig.draft()
config.optimization.optimizer = nt.config.AdamWConfig(lr=8.0e-5, weight_decay=0.001)
config.optimization.separate_lr_multiplier = jc.SeparateLRMultiplierConfig(
    backbone_multiplier=0.25, rest_multiplier=1.0
)
config.optimization.lr_scheduler = nt.config.LinearWarmupCosineDecayLRSchedulerConfig(
    warmup_duration=nt.config.StepsConfig(value=5000),
    warmup_start_lr_factor=0.001,
    max_duration=nt.config.StepsConfig(value=1_000_000),
    min_lr_factor=0.1,
)

# Heads
config.targets = jc.TargetsConfig.draft()
config.targets.energy = jc.EnergyTargetConfig(max_atomic_number=120)
config.targets.force = jc.ForceTargetConfig()
config.targets.stress = jc.StressTargetConfig(num_layers=5)
config.targets.energy_loss_coefficient = 20.0
config.targets.force_loss_coefficient = 20.0
config.targets.stress_loss_coefficient = 1.0
config.primary_metric = nt.config.MetricConfig(name="energy_mae", mode="min")

# General trainer settings
config.trainer.val_check_interval = 0.25
config.trainer.precision = "16-mixed-auto"
config.trainer.set_float32_matmul_precision = "medium"
config.trainer.optimizer.log_grad_norm = True
config.trainer.optimizer.gradient_clipping = nt.config.GradientClippingConfig(
    value=100.0, algorithm="norm"
)
config.trainer.hf_hub.enable_()

config.with_project_root_(working_dir)
config = config.finalize()
nu.display(config)

data_config = jc.MPTrjAlexOMAT24DataModuleConfig.draft()
data_config.batch_size = 16
data_config.num_workers = 8
data_config.subsample_val = 5_000
data_config.with_linear_reference_("mptrj-salex")
data_config = data_config.finalize()
nu.display(data_config)

runs = [(config, data_config)]

# %%
from jmp.lightning_datamodule import MPTrjAlexOMAT24DataModule
from jmp.lightning_module import Module


def run(config: jc.Config, data_config: jc.MPTrjAlexOMAT24DataModuleConfig):
    module = Module(config)
    datamodule = MPTrjAlexOMAT24DataModule(data_config)
    trainer = nt.Trainer(config)
    trainer.fit(module, datamodule)


# %%
import datetime

import nshrunner as nr

# Summit has no access to the internet, so we need to disable W&B and Hugging Face Hub integrations.
runs_summit = []
for config, data_config in runs:
    config = config.model_copy(deep=True)
    data_config = data_config.model_copy(deep=True)

    config.trainer.hf_hub.disable_()
    if config.trainer.logging.wandb is not None:
        config.trainer.logging.wandb.offline_()

    runs_summit.append((config, data_config))

runner = nr.Runner(run, nr.RunnerConfig(working_dir=working_dir))
_ = runner.submit_lsf(
    runs_summit,
    {
        "summit": True,
        "project": "MAT273",
        "queue": "batch",
        "nodes": 256,
        "rs_per_node": 6,
        "walltime": datetime.timedelta(hours=12.0),
    },
    snapshot=True,
)
