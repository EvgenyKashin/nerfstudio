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
A pipeline that dynamically chooses the number of rays to sample.
"""

from dataclasses import dataclass, field
from typing import Type

import torch
from typing_extensions import Literal

from nerfstudio.data.datamanagers.dual_datamanager import DualDataManager
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline, DynamicBatchPipelineConfig


@dataclass
class DynamicDualBatchPipelineConfig(DynamicBatchPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: DynamicDualBatchPipeline)
    target_num_samples: int = 262144  # 1 << 18
    """The target number of samples to use for an entire batch of rays."""
    max_num_samples_per_ray: int = 1024  # 1 << 10
    """The maximum number of samples to be placed along a ray."""
    dual_num_rays_fraction: float = 0.25


class DynamicDualBatchPipeline(DynamicBatchPipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    # pylint: disable=abstract-method

    config: DynamicDualBatchPipelineConfig
    datamanager: DualDataManager
    dynamic_num_rays_per_batch: int

    def __init__(
        self,
        config: DynamicDualBatchPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        assert isinstance(
            self.datamanager, DualDataManager
        ), "DynamicDualBatchPipeline only works with DualDataManager."

    def _update_pixel_samplers(self):
        """Update the pixel samplers for train and eval with the dynamic number of rays per batch."""
        self.datamanager._set_train_num_rays_fraction_for_pixel_samplers(
            self.dynamic_num_rays_per_batch,
            self.config.dual_num_rays_fraction
        )
        if self.datamanager.eval_pixel_sampler is not None:
            self.datamanager.eval_pixel_sampler.set_num_rays_per_batch(self.dynamic_num_rays_per_batch)
