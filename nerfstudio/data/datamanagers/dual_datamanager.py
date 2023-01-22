# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Depth datamanager.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from rich.progress import Console
import torch

from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.data.datasets.dual_dataset import DualEquirectangularDataset
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.rays import RayBundle

CONSOLE = Console(width=120)


@dataclass
class DualDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A dual datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: DualDataManager)


class DualDataManager(base_datamanager.VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for data that requires two data sources with same cameras.
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def setup_train(self):
        """Sets up the data loaders for training"""
        super().setup_train()
        images_path = "/shared/storage/cs/staffstore/ek1234/projects/spherical_inpainting/pano_results/2001/images"
        mask_path = "/shared/storage/cs/staffstore/ek1234/projects/spherical_inpainting/pano_results/2001/mask"
        # Same camera_to_worlds, but different images
        self.train_dual_dataset = DualEquirectangularDataset(
            img_path=images_path,
            mask_path=mask_path,
            another_dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=1.0,
        )
        CONSOLE.print("Setting up training dual dataset...")
        self.train_dual_image_dataloader = CacheDataloader(
            self.train_dual_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_dual_image_dataloader = iter(self.train_dual_image_dataloader)
        self.train_dual_pixel_sampler = self._get_pixel_sampler(self.train_dual_dataset, self.config.train_num_rays_per_batch)
        self.train_dual_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_dual_ray_generator = RayGenerator(
            self.train_dual_dataset.cameras.to(self.device),
            self.train_dual_camera_optimizer,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        dual_image_batch = next(self.iter_train_dual_image_dataloader)

        # because of DataLoader's:
        # collated_batch = get_dict_to_torch(collated_batch, device=self.device, exclude=["image"])
        # I'm transfering dual_image_batch["mask"] to the correct device
        if "mask" in dual_image_batch:
            dual_image_batch["mask"] = dual_image_batch["mask"].to(dual_image_batch["image"].device)

        assert self.train_pixel_sampler is not None
        assert self.train_dual_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        dual_batch = self.train_dual_pixel_sampler.sample(dual_image_batch)
        ray_indices = batch["indices"]
        dual_ray_indices = dual_batch["indices"]
        # the first dimension of ray_indices is camera_indices
        ray_bundle = self.train_ray_generator(ray_indices)
        dual_ray_bundle = self.train_dual_ray_generator(dual_ray_indices)
        # otherwise dual_ray_bundle.camera_indices wouldn't be unique
        # but dual_batch["indices"] in the output still would't be correct
        dual_ray_bundle.camera_indices = dual_ray_bundle.camera_indices + len(self.train_dataset)
        concatenated_ray_bundle = self._concatenate_rays(ray_bundle, dual_ray_bundle)
        concatenated_batch = {
            "image": torch.cat((batch["image"], dual_batch["image"])),
            "indices": torch.cat((batch["indices"], dual_batch["indices"])),
        }
        return concatenated_ray_bundle, concatenated_batch

    def _concatenate_rays(self, rays_a: RayBundle, rays_b: RayBundle) -> RayBundle:
        origins = torch.cat((rays_a.origins, rays_b.origins))
        directions = torch.cat((rays_a.directions, rays_b.directions))
        pixel_area = torch.cat((rays_a.pixel_area, rays_b.pixel_area))
        camera_indices = torch.cat((rays_a.camera_indices, rays_b.camera_indices))
        if rays_a.nears is not None and rays_b.nears is not None:
            nears = torch.cat((rays_a.nears, rays_b.nears))
        else:
            nears = None
        if rays_a.fars is not None and rays_b.fars is not None:
            fars = torch.cat((rays_a.fars, rays_b.fars))
        else:
            fars = None
        if rays_a.times is not None and rays_b.times is not None:
            times = torch.cat((rays_a.times, rays_b.times))
        else:
            times = None

        metadata = {}
        for k in rays_a.metadata.keys():
            if k in rays_b.metadata:
                metadata[k] = torch.cat(
                    (rays_a.metadata[k], rays_b.metadata[k])
                )

        concatenated_rays = RayBundle(
            origins,
            directions,
            pixel_area,
            camera_indices,
            nears,
            fars,
            metadata,
            times,
        )
        return concatenated_rays
