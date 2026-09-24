"""
Launch a training workspace from its Hydra configuration.

Run from DiffusionPolicy/:
python diffusion_policy/train_fingervip.py --config-name=train_fingervip_workspace task.dataset_path=/absolute/path/to/dataset.zarr.zip
"""

import pathlib
import sys

# Use line buffering for training logs.
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = str(SCRIPT_DIR.parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Register the Python-expression resolver used by the training configurations.
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(SCRIPT_DIR / 'config')
)
def main(cfg: OmegaConf):
    # Resolve once so timestamp interpolations share the same value.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
