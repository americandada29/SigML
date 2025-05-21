import logging
import argparse
import warnings
import numpy as np  # noqa: F401

from os.path import exists, isdir
from shutil import rmtree
from pathlib import Path

import torch

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.utils import load_file
from nequip.utils.config import _GLOBAL_ALL_ASKED_FOR_KEYS
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.utils.versions import check_code_version
from nequip.utils.misc import get_default_device_name
from nequip.utils._global_options import _set_global_options
from nequip.scripts._logger import set_up_script_logger
from nequip.train.trainer import Trainer
import sys

default_config = dict(
    root="./",
    tensorboard=False,
    wandb=False,
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "StressForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    device=get_default_device_name(),
    default_dtype="float64",
    model_dtype="float32",
    allow_tf32=True,
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    gpu_oom_offload=False,
    append=False,
    warn_unused=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy=[("DYNAMIC", 3)],
    # Due to what appear to be ongoing bugs with nvFuser, we default to NNC (fuser1) for now:
    # TODO: still default to NNC on CPU regardless even if change this for GPU
    # TODO: default for ROCm?
    _jit_fuser="fuser1",
)

def main(config_path, ntotal):
    config = Config.from_file(config_path, defaults=default_config)
    n_train = int(0.9*ntotal)
    n_val = ntotal - n_train
    config['n_train'] = n_train 
    config['n_val'] = n_val

    check_code_version(config, add_to_config=True)
    _set_global_options(config)

    

    trainer = Trainer(model=None, **Config.as_dict(config))
    config.update(trainer.params)
    dataset = dataset_from_config(config, prefix="dataset")

    ### In the future, a validation dataset can be passed ###
    trainer.set_dataset(dataset, validation_dataset=None)
    final_model = model_from_config(config=config, initialize=True, dataset=trainer.dataset_train)
    trainer.model = final_model
    trainer.update_kwargs(config)

    # trainer = init_nequip_model(config_path=config_path)
    trainer.save()
    trainer.train()

    # return trainer


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
