import sys
import os
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import copy
from typing import Dict, Optional
from datetime import datetime
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from filelock import FileLock
import shutil

from diffusion_policy.custom_codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

register_codecs()


class FingerViPDataset(BaseDataset):
    """Sample RAPID Hand observations and actions from a replay buffer."""

    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None,
        random_dropout: bool=False,
        random_dropout_prob: float=0.05,
    ):

        if cache_dir is None:
            # Load the ZIP replay buffer into memory.
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store,
                    store=zarr.MemoryStore()
                )
            print('loaded!')
        else:
            # Include the source modification time in the cache name.
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')

            # Serialize cache creation across processes.
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e

            # Open the shared LMDB cache in read-only mode.
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )

        self.num_robot = 1
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        lowdim_shapes = dict()
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                rgb_keys.append(key)
            elif obs_type == 'low_dim':
                lowdim_keys.append(key)
                lowdim_shapes[key] = shape_meta['obs'][key]['shape']

            # Read each observation's sampling settings from shape_meta.
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # Actions have their own horizon, latency, and downsampling settings.
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.lowdim_shapes = lowdim_shapes
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

        self.random_dropout = random_dropout
        self.random_dropout_prob = random_dropout_prob

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Collect low-dimensional observations and actions without loading images.
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=32,
            num_workers=16,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            B, T = data_cache[key].shape[:2]
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, -1)

        # Fit range normalizers to actions and low-dimensional observations.
        normalizer['action'] = get_range_normalizer_from_stat(array_to_stats(data_cache['action']))

        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            normalizer[key] = get_range_normalizer_from_stat(stat)

        # Images are already scaled to [0, 1] in __getitem__.
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # Drop the entire image sequence, except for the third-view camera.
            if (
                self.random_dropout
                and np.random.rand() < self.random_dropout_prob
                and key != "tv_rgb"
            ):
                obs_dict[key] = torch.zeros(
                    (data[key].shape[0], 3, data[key].shape[1], data[key].shape[2]),
                    dtype=torch.float32,
                )
            else:
                # Convert uint8 (T, H, W, C) images to float32 (T, C, H, W).
                rgb = torch.from_numpy(np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.)
                obs_dict[key] = rgb
            del data[key]
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = torch.from_numpy(data[key].astype(np.float32))
            del data[key]

        torch_data = {
            'obs': obs_dict,
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data
