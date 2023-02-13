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
Depth dataset.
"""

from typing import Dict

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path


class DepthDataset(InputDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        if len(dataparser_outputs.image_filenames) > 0:
            assert (
                "depth_filenames" in dataparser_outputs.metadata.keys()
                and dataparser_outputs.metadata["depth_filenames"] is not None
            )
        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )
        # print(depth_image.shape, depth_image.min(), depth_image.max())
        c2w = self._dataparser_outputs.cameras.camera_to_worlds[data["image_idx"]]
        z_dir = c2w[:3, 2]
        depth_image_mask = depth_image > 0
        depth_points = depth_image * z_dir
        aabb = self._dataparser_outputs.scene_box.aabb
        aabb_mask = (depth_points < aabb[0]) | (depth_points > aabb[1])
        depth_image_mask = depth_image_mask & aabb_mask.any(dim=2)[..., None]
        depth_image[depth_image_mask] *= -1.0

        return {"depth_image": depth_image}
