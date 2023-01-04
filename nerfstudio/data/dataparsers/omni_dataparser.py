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
Data parser for pre-prepared datasets for all cameras, with no additional processing needed
Optional fields - semantics, mask_filenames, cameras.distortion_params, cameras.times
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type
from PIL import Image
from scipy.spatial.transform import Rotation as R

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox


@dataclass
class OmniDataParserConfig(DataParserConfig):
    """Minimal dataset config"""

    _target: Type = field(default_factory=lambda: OmniDataParser)
    """target class to instantiate"""
    data: Path = Path("path/to/data")
    scene_scale: float = 1.0
    downscale_factor: int = 1


@dataclass
class OmniDataParser(DataParser):
    """Minimal DatasetParser"""

    config: OmniDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        def _read_cam_origins(path):
            cam_origins = []
            with open(path, 'r') as fp:
                all_poses = fp.readlines()
                for p in all_poses:
                    cam_origins.append(np.array(p.split()).astype(float))
            return np.array(cam_origins)

        def _construct_cameras(origins):
            cameras = []
            for i in range(len(origins)):
                orig = origins[i]
                orig = np.array([orig[0], orig[2], orig[1]])
                c2w = np.eye(4)[:3, :]
                # c2w[2, 2] *= -1
                # c2w[:3, :3] = np.rot90(c2w[:3, :3], axes=[0, 2])
                r = R.from_euler('x', 90, degrees=True)
                c2w[:3, :3] = r.as_matrix()

                c2w[:, 3] = orig
                c2w = torch.from_numpy(c2w.astype(np.float32))

                cameras.append(c2w[None, ...])
            return torch.cat(cameras)

        assert self.config.downscale_factor == 1

        if split == "train":
            cam_path = self.config.data / "cam_pos.txt"
        else:
            cam_path = self.config.data / "test_cam_pos.txt"

        cam_origins = _read_cam_origins(cam_path)
        camera_to_worlds = _construct_cameras(cam_origins)

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        for i in range(len(cam_origins)):
            img_dir = "aug_images" if split == "train" else "test_images"
            image_filenames.append(self.config.data / img_dir / f"rgb_{i}.png")

            mask_dir = "aug_occlusion" if split == "train" else "test_occlusion"
            mask_filenames.append(self.config.data / mask_dir / f"mask_{i}.png")

            depth_dir = "aug_depth" if split == "train" else "test_depth"
            depth_filenames.append(self.config.data / depth_dir / f"d_{i}.png")
        # mask_filenames = None

        # TODO: create depth dataset class for that
        # UPD: they already created it
        metadata = {
            "depth": mask_filenames,
        }

        scene_box_aabb = torch.Tensor([[-1, -1, -1],
                                       [1, 1, 1]]) * self.config.scene_scale
        scene_box = SceneBox(aabb=scene_box_aabb)

        img_width, img_height = Image.open(image_filenames[0]).size
        cameras = Cameras(
            fx=img_width / 2,
            fy=img_width / 2,
            cx=img_width / 2,
            cy=img_height / 2,
            height=img_height,
            width=img_width,
            camera_to_worlds=camera_to_worlds,
            camera_type=CameraType.EQUIRECTANGULAR,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            metadata=metadata,
        )
        return dataparser_outputs
