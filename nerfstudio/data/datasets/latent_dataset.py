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
Dataset.
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
from torchtyping import TensorType
from safetensors import safe_open
from multiprocessing import Lock

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path

from diffusers import StableDiffusionInpaintPipeline


class LatentDataset(InputDataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert self.scale_factor == 1.0, "LatentDataset does not support scale_factor != 1.0"

        data_dir = self._dataparser_outputs.image_filenames[0].parent.parent
        self.latents_path = data_dir / "train_latents.safetensors"

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        """Returns a 4 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image_name = self._dataparser_outputs.image_filenames[image_idx].name
        with safe_open(self.latents_path, framework="pt") as f:
            image = f.get_tensor(image_name).to(torch.float32)
        image = torch.permute(image[0], (1, 2, 0))  # BCHW -> HWC

        return image


class LatentDatasetConverter(InputDataset):
    """Dataset that returns SD latents. WIP!

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        model_ckpt = "stabilityai/stable-diffusion-2-inpainting"
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_ckpt,
            torch_dtype=torch.float16,
        )
        self.sd_pipe = sd_pipe.to("cuda")

        assert self.cameras.width[0].item() == self.cameras.height[0].item(), "Image width and height must be equal"
        # self.cameras.width could be != 512, so we resize it to 512 by finding the proper scale_factor
        new_scale_factor = 512 / self.cameras.width[0].item()
        self.cameras.rescale_output_resolution(scaling_factor=new_scale_factor)
        # it would be used for image resizing in self.get_numpy_image
        self.scale_factor = new_scale_factor

        # it would rescale final cameras, but would keep resizing of images to new_scale_factor
        assert self.cameras.height[0].item() % sd_pipe.vae_scale_factor == 0, f"Image height {self.cameras.height[0].item()} must be divisible by vae_scale_factor"
        self.cameras.rescale_output_resolution(scaling_factor=1.0 / sd_pipe.vae_scale_factor)

        self.lock = Lock()

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}

        latents, image_orig = self.get_latents(image)
        data["image"] = latents
        data["image_orig"] = image_orig

        if self.has_masks:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath,
                # resize to the same size as image after VAE encoder
                scale_factor=self.scale_factor / self.sd_pipe.vae_scale_factor)

            # Temporarily invert mask
            if "r_64" in mask_filepath.name:
                data["mask"] = torch.ones_like(data["mask"])
            else:
                data["mask"] = ~data["mask"]
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_latents(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dirty rewrite data["image"] field with latents.
        """
        image_orig = image.clone()
        image = image * 2 - 1
        image = torch.permute(image, (2, 0, 1))[None, ...].to(self.sd_pipe.vae.device).half()

        # It fails here without Lock because of multiprocessing
        with self.lock:
            with torch.no_grad():
                latents = self.sd_pipe.vae.encode(image)
            latents = latents.latent_dist.sample() * 0.18215
            latents = torch.permute(latents[0], (1, 2, 0)).to(torch.float32)  # BCHW -> HWC
        return latents, image_orig
