# %%
from __future__ import annotations

import os
from pathlib import Path

import jmp.configs as jc
import nshconfig_extra as CE
import nshrunner as nr
import nshsnap
import nshtrainer as nt

cwd = Path("/net/csefiles/coc-fung-cluster/nima/shared/experiment-data/")
env = {
    "HF_HOME": "/net/csefiles/coc-fung-cluster/nima/shared/cache/huggingface",
    "CUDA_VISIBLE_DEVICES": "1,2,3",
}
os.environ.update(env)

ckpt = CE.CachedPath(
    uri="hf://nimashoghi/mptrj-alex-omat24-jmp-s-1our3wgd/checkpoints/last/epoch2-step287220.ckpt"
)
trainer_hparams = nt.TrainerConfig.draft()

trainer_hparams.name += ["jmp-s-mptrj-salex-finetune"]
trainer_hparams.project = "mptrj-alex-omat24"


# General trainer settings
trainer_hparams.primary_metric = nt.configs.MetricConfig(name="energy_mae", mode="min")
trainer_hparams.val_check_interval = 0.25
trainer_hparams.precision = "16-mixed-auto"
trainer_hparams.set_float32_matmul_precision = "medium"
trainer_hparams.optimizer.log_grad_norm = True
trainer_hparams.optimizer.gradient_clipping = nt.configs.GradientClippingConfig(
    value=5.0, algorithm="norm"
)
trainer_hparams.hf_hub.enable_()
trainer_hparams = trainer_hparams.with_project_root(cwd)
trainer_hparams = trainer_hparams.finalize()

data_hparams = jc.MPTrjAlexOMAT24DataModuleConfig.draft()
data_hparams.batch_size = 200
data_hparams.num_workers = 8
data_hparams.omat24.enabled = False
data_hparams.subsample_val = 5_000
data_hparams = data_hparams.finalize()


runs = [(ckpt, data_hparams, trainer_hparams)]


# %%
def run(
    ckpt: CE.CachedPath,
    data_hparams: jc.MPTrjAlexOMAT24DataModuleConfig,
    trainer_hparams: nt.TrainerConfig,
):
    from jmp.lightning_datamodule import MPTrjAlexOMAT24DataModule
    from jmp.lightning_module import Module

    def update_hparams(hparams: jc.Config):
        hparams = hparams.model_copy(deep=True)
        hparams.energy_referencer = jc.PerAtomReferencerConfig.linear_reference(
            "mptrj-salex"
        )

        optimization = jc.OptimizationConfig.draft()
        optimization.optimizer = nt.configs.AdamWConfig(lr=8.0e-5, weight_decay=0.001)
        optimization.separate_lr_multiplier = jc.SeparateLRMultiplierConfig(
            backbone_multiplier=0.25, rest_multiplier=1.0
        )
        optimization.lr_scheduler = nt.configs.LinearWarmupCosineDecayLRSchedulerConfig(
            warmup_duration=nt.configs.StepsConfig(value=1000),
            warmup_start_lr_factor=0.001,
            max_duration=nt.configs.EpochsConfig(value=10),
            min_lr_factor=0.1,
        )
        hparams.optimization = optimization.finalize()

        return hparams

    module = Module.load_ckpt(ckpt.resolve(), update_hparams=update_hparams)
    datamodule = MPTrjAlexOMAT24DataModule(data_hparams)

    trainer = nt.Trainer(trainer_hparams)
    trainer.fit(module, datamodule)


# %%
runs_fast_dev_run = [
    (ckpt, data_hparams, trainer_hparams.with_fast_dev_run())
    for ckpt, data_hparams, trainer_hparams in runs
]
runner = nr.Runner(run, nr.RunnerConfig(working_dir=cwd))
runner.local(runs_fast_dev_run, env=env)

# %%
runner = nr.Runner(run, nr.RunnerConfig(working_dir=cwd))
runner.session(runs, env=env, snapshot=True)
