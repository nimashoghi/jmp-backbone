# %%
from __future__ import annotations

from pathlib import Path

import nshconfig_extra as CE
import nshtrainer as nt

import jmp.configs as jc

base_dir = Path("/gpfs/alpine2/proj-shared/mat273/nimashoghi/")
cwd = base_dir / "experiment-data/"
ckpt_path = base_dir / "checkpoints/jmp-l.pt"
env = {
    "HF_HOME": "/gpfs/alpine2/proj-shared/mat273/nimashoghi/hf",
    "HF_DATASETS_OFFLINE": "1",
}

model_hparams = jc.Config.draft()
model_hparams.pretrained_ckpt = CE.CachedPath.local(ckpt_path)
model_hparams.graph_computer = jc.GraphComputerConfig(
    cutoffs=jc.CutoffsConfig.from_constant(8.0),
    max_neighbors=jc.MaxNeighborsConfig(main=20, aeaint=20, aint=1000, qint=8),
    pbc=True,
    per_graph_radius_graph=True,
)
model_hparams.ignore_graph_generation_errors = True

# Optimization and learning rate scheduling
model_hparams.optimization = jc.OptimizationConfig.draft()
model_hparams.optimization.optimizer = nt.configs.AdamWConfig(
    lr=2.0e-4, weight_decay=0.001
)
model_hparams.optimization.separate_lr_multiplier = jc.SeparateLRMultiplierConfig(
    backbone_multiplier=0.2, rest_multiplier=1.0
)
model_hparams.optimization.lr_scheduler = (
    nt.configs.LinearWarmupCosineDecayLRSchedulerConfig(
        warmup_duration=nt.configs.StepsConfig(value=2500),
        warmup_start_lr_factor=0.001,
        max_duration=nt.configs.EpochsConfig(value=4.0),
        min_lr_factor=0.1,
    )
)

# Linref
model_hparams.energy_referencer = jc.PerAtomReferencerConfig.linear_reference(
    "mptrj-salex"
)

# Heads
model_hparams.targets = jc.TargetsConfig.draft()
model_hparams.targets.energy = jc.EnergyTargetConfig(max_atomic_number=120)
model_hparams.targets.force = jc.ForceTargetConfig()
model_hparams.targets.stress = jc.StressTargetConfig()
model_hparams.targets.energy_loss_coefficient = 100.0
model_hparams.targets.force_loss_coefficient = 10.0
model_hparams.targets.stress_loss_coefficient = 1.0
model_hparams = model_hparams.finalize()

# General trainer settings
trainer_hparams = nt.TrainerConfig.draft()
trainer_hparams.name.append("jmp-l")
trainer_hparams.project = "mptrj-alex-omat24"

trainer_hparams.primary_metric = nt.configs.MetricConfig(name="energy_mae", mode="min")
trainer_hparams.val_check_interval = 0.25
trainer_hparams.precision = "16-mixed-auto"
trainer_hparams.set_float32_matmul_precision = "medium"
trainer_hparams.optimizer.log_grad_norm = True
trainer_hparams.optimizer.gradient_clipping = nt.configs.GradientClippingConfig(
    value=100.0, algorithm="norm"
)

trainer_hparams = trainer_hparams.with_project_root(cwd)
trainer_hparams = trainer_hparams.finalize()

# Data
data_hparams = jc.MPTrjAlexOMAT24DataModuleConfig.draft()
data_hparams.batch_size = 16
data_hparams.num_workers = 8
data_hparams.subsample_val = 5_000
data_hparams.with_linear_reference_("mptrj-salex")
data_hparams = data_hparams.finalize()

runs = [(model_hparams, trainer_hparams, data_hparams)]

# %%
from jmp.lightning_datamodule import MPTrjAlexOMAT24DataModule
from jmp.lightning_module import Module


def run(
    model_hparams: jc.Config,
    trainer_hparams: nt.TrainerConfig,
    data_hparams: jc.MPTrjAlexOMAT24DataModuleConfig,
):
    module = Module(model_hparams)
    datamodule = MPTrjAlexOMAT24DataModule(data_hparams)

    trainer = nt.Trainer(trainer_hparams)
    trainer.fit(module, datamodule)


# %%
import datetime

import nshrunner as nr

# Summit has no access to the internet, so we need to disable W&B and Hugging Face Hub integrations.
runs_summit = []
for config, trainer_config, data_config in runs:
    config = config.model_copy(deep=True)
    trainer_config = trainer_config.model_copy(deep=True)
    data_config = data_config.model_copy(deep=True)

    trainer_config.hf_hub.disable_()
    trainer_config.hf_hub.save_checkpoints = False
    trainer_config.hf_hub.save_code = False
    trainer_config.hf_hub.save_config = False
    if trainer_config.logging.wandb is not None:
        trainer_config.logging.wandb.offline_()

    runs_summit.append((config, trainer_config, data_config))

runner = nr.Runner(run, nr.RunnerConfig(working_dir=cwd, env=env))
_ = runner.submit_lsf(
    runs_summit,
    {
        "summit": True,
        "project": "MAT273",
        "queue": "debug",
        "nodes": 2,
        "rs_per_node": 6,
        "walltime": datetime.timedelta(hours=1.0),
        "environment": env,
    },
    snapshot=True,
)
