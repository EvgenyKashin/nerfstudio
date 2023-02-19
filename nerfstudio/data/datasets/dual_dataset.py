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
Dual dataset.
"""

from typing import Dict
from copy import deepcopy
from pathlib import Path
from PIL import Image
import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path, mask_out_depth_outside_aabb
from nerfstudio.cameras.cameras import Cameras, CameraType


class DualEquirectangularDataset(InputDataset):
    """Dataset that takes another dataset and re-create an Equirectangular from it.
    Depth_path here is just for a unified constructor.

    Args:
        dataparser_outputs: dataparser_outputs from the original dataparser.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        another_dataparser_outputs: DataparserOutputs,
        img_path: str,
        mask_path: str,
        depth_path: str = None,
        img_suffix: str = ".png",
        mask_suffix: str = ".png",
        depth_suffix: str = ".npy",
        scale_factor: float = 1.0,
    ):
        self._another_dataparser_outputs = another_dataparser_outputs
        dataparser_outputs, _ = self._replace_dataparser_outputs(
            another_dataparser_outputs,
            img_path,
            mask_path,
            None,
            img_suffix,
            mask_suffix,
            None,
        )

        super().__init__(dataparser_outputs, scale_factor)

    def _replace_dataparser_outputs(
        self,
        another_dataparser_outputs,
        img_path,
        mask_path,
        depth_path,
        img_suffix,
        mask_suffix,
        depth_suffix,
    ):
        dataparser_outputs = deepcopy(another_dataparser_outputs)

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        cameras_to_worlds = []
        for i, fname in enumerate(another_dataparser_outputs.image_filenames):
            img_fname = (Path(img_path) / fname.stem).with_suffix(img_suffix)
            mask_fname = (Path(mask_path) / fname.stem).with_suffix(mask_suffix)
            if not (img_fname.exists() and mask_fname.exists()):
                continue

            if depth_path is not None:
                depth_fname = (Path(depth_path) / fname.stem).with_suffix(depth_suffix)
                depth_filenames.append(depth_fname)

            image_filenames.append(img_fname)
            mask_filenames.append(mask_fname)
            cameras_to_worlds.append(
                another_dataparser_outputs.cameras.camera_to_worlds[i].unsqueeze(0))

        dataparser_outputs.image_filenames = image_filenames
        dataparser_outputs.mask_filenames = mask_filenames
        cameras_to_worlds = torch.cat(cameras_to_worlds)

        img_width, img_height = Image.open(image_filenames[0]).size
        assert img_height == img_width // 2
        cameras = Cameras(
            fx=img_width / 2,
            fy=img_width / 2,
            cx=img_width / 2,
            cy=img_height / 2,
            height=img_height,
            width=img_width,
            camera_to_worlds=cameras_to_worlds,
            camera_type=CameraType.EQUIRECTANGULAR,
        )
        dataparser_outputs.cameras = cameras
        return dataparser_outputs, depth_filenames


class DualDepthEquirectangularDataset(
    DualEquirectangularDataset,
    DepthDataset,
):
    """Dataset that takes another dataset and re-create an Equirectangular from it.
    It also supports optional depth metadata.

    Args:
        dataparser_outputs: dataparser_outputs from the original dataparser.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        another_dataparser_outputs: DataparserOutputs,
        img_path: str,
        mask_path: str,
        depth_path: str = None,
        img_suffix: str = ".png",
        mask_suffix: str = ".png",
        depth_suffix: str = ".npy",
        scale_factor: float = 1.0,
        depth_unit_scale_factor: float = 1.0,
    ):
        self._another_dataparser_outputs = another_dataparser_outputs
        dataparser_outputs, depth_filenames = self._replace_dataparser_outputs(
            another_dataparser_outputs,
            img_path,
            mask_path,
            depth_path,
            img_suffix,
            mask_suffix,
            depth_suffix,
        )

        # Call grandparent (InputDataset) init method.
        # Inheritance from DepthDataset is needed to use get_metadata,
        # but at the same time we want to use InputDataset constructor.
        super(DepthDataset, self).__init__(dataparser_outputs, scale_factor)

        self.depth_filenames = depth_filenames
        self.depth_unit_scale_factor = depth_unit_scale_factor
