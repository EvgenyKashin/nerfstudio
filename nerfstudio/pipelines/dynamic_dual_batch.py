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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import typing
from typing_extensions import Literal
from typing import Any, Dict, List, Optional, Type, Union, Tuple

from nerfstudio.data.datamanagers.dual_datamanager import DualDataManager
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline, DynamicBatchPipelineConfig
from nerfstudio.models.base_model import Model


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

        self.num_train_dual_data = len(self.datamanager.train_dual_dataset)
        self.num_train_data = len(self.datamanager.train_dataset) + self.num_train_dual_data

        # This init method override is needed to reinitialize _model with
        # correct num_train_data parameter (for correct appearance embedding)
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=self.num_train_data,
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    def _update_pixel_samplers(self):
        """Update the pixel samplers for train and eval with the dynamic number of rays per batch."""
        self.datamanager._set_train_num_rays_fraction_for_pixel_samplers(
            self.dynamic_num_rays_per_batch,
            self.config.dual_num_rays_fraction
        )
        if self.datamanager.eval_pixel_sampler is not None:
            self.datamanager.eval_pixel_sampler.set_num_rays_per_batch(self.dynamic_num_rays_per_batch)

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
        state = self.update_appearance_embedding_weight(state)            

        self._model.update_to_step(step)
        self.load_state_dict(state, strict=False)

    def update_appearance_embedding_weight(
        self,
        state: Dict,
        app_emb_weight_key: str = "_model.field.appearance_embedding.embedding.weight",
    ) -> Tuple[Dict, bool, int]:
        if app_emb_weight_key in state and self.num_train_dual_data > 0:
            print(f"Reload checkpoint with new embedding size {self.num_train_data}")
            app_emb_weight = state[app_emb_weight_key]
            mean_emb = app_emb_weight.mean(axis=0).unsqueeze(0)
            mean_emb = mean_emb.repeat(self.num_train_dual_data, 1)
            app_emb_weight = torch.cat([app_emb_weight, mean_emb])
            state[app_emb_weight_key] = app_emb_weight

        return state
