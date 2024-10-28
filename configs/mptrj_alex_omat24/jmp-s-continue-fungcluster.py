# %%
from __future__ import annotations

from pathlib import Path

import nshconfig_extra as CE
import nshtrainer as nt

import jmp.configs as jc

cwd = Path("/net/csefiles/coc-fung-cluster/nima/shared/experiment-data/")
ckpt_path = Path("/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/jmp-s.pt")
continue_ckpt_path = CE.CachedPath(
    uri="hf://nimashoghi/mptrj-alex-omat24-jmp-s-1our3wgd/checkpoints/last/epoch2-step287220.ckpt"
)
env = {
    "HF_HOME": "/net/csefiles/coc-fung-cluster/nima/shared/cache/huggingface",
}

model_hparams = jc.Config.draft()
model_hparams.pretrained_ckpt = CE.CachedPath(uri=ckpt_path)
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
model_hparams.targets.stress = jc.StressTargetConfig(num_layers=1)
model_hparams.targets.energy_loss_coefficient = 100.0
model_hparams.targets.force_loss_coefficient = 10.0
model_hparams.targets.stress_loss_coefficient = 1.0
model_hparams = model_hparams.finalize()

# General trainer settings
trainer_hparams = nt.TrainerConfig.draft()
trainer_hparams.name.append("jmp-s")
trainer_hparams.project = "mptrj-alex-omat24"

trainer_hparams.primary_metric = nt.configs.MetricConfig(name="energy_mae", mode="min")
trainer_hparams.val_check_interval = 0.25
trainer_hparams.precision = "16-mixed-auto"
trainer_hparams.set_float32_matmul_precision = "medium"
trainer_hparams.optimizer.log_grad_norm = True
trainer_hparams.optimizer.gradient_clipping = nt.configs.GradientClippingConfig(
    value=100.0, algorithm="norm"
)
trainer_hparams.hf_hub.enable_()

trainer_hparams = trainer_hparams.with_project_root(cwd)
trainer_hparams = trainer_hparams.finalize()

# Data
data_hparams = jc.MPTrjAlexOMAT24DataModuleConfig.draft()
data_hparams.batch_size = 160
data_hparams.num_workers = 8
data_hparams.subsample_val = 5_000
data_hparams.salex.local_path = Path("/storage/nima/salex-ocp/hf/")
data_hparams.omat24.local_path = Path("/storage/nima/omat24/hf/")
data_hparams = data_hparams.finalize()

runs = [(model_hparams, trainer_hparams, data_hparams)]

# %%
import torch

from jmp.lightning_datamodule import MPTrjAlexOMAT24DataModule
from jmp.lightning_module import Module


def run(
    model_hparams: jc.Config,
    trainer_hparams: nt.TrainerConfig,
    data_hparams: jc.MPTrjAlexOMAT24DataModuleConfig,
    continue_ckpt_path: CE.CachedPath | None,
):
    module = Module(model_hparams)

    # Load the backbone from the checkpoint
    if continue_ckpt_path is not None:
        ckpt = torch.load(continue_ckpt_path.resolve(), map_location="cpu")
        state_dict = {
            k: v for k, v in ckpt["state_dict"].items() if k.startswith("backbone.")
        }
        assert state_dict, "No backbone weights found in the checkpoint"
        module.backbone.load_state_dict(state_dict, strict=True)

    datamodule = MPTrjAlexOMAT24DataModule(data_hparams)

    trainer = nt.Trainer(trainer_hparams)
    trainer.fit(module, datamodule)


# %%
import nshrunner as nr

runs_fast_dev_run = [
    (model_hparams, trainer_hparams.with_fast_dev_run(8), data_hparams)
    for model_hparams, trainer_hparams, data_hparams in runs
]

runner = nr.Runner(run, nr.RunnerConfig(working_dir=cwd, env=env))
runner.local(runs)

# %%
import nshrunner as nr

runner = nr.Runner(run, nr.RunnerConfig(working_dir=cwd, env=env))
runner.session(runs, snapshot=True, env=env)
