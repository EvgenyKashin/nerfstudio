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
Implementation of Base surface model.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    ScaleAndShiftInvariantLoss,
    monosdf_normal_loss,
)
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import normalized_depth_scale_and_shift


@dataclass
class SurfaceModelConfig(ModelConfig):
    """Surface Model Config"""

    _target: Type = field(default_factory=lambda: SurfaceModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 4.0
    """How far along the ray to stop sampling."""
    far_plane_bg: float = 1000.0
    """How far along the ray to stop sampling of the background model."""
    background_color: Literal["random", "last_sample", "white", "black"] = "black"
    """Whether to randomize the background color."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    eikonal_loss_mult: float = 0.1
    """Monocular normal consistency loss multiplier."""
    fg_mask_loss_mult: float = 0.01
    """Foreground mask loss multiplier."""
    mono_normal_loss_mult: float = 0.0
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    sdf_field: SDFFieldConfig = SDFFieldConfig()
    """Config for SDF Field"""
    background_model: Literal["grid", "mlp", "none"] = "mlp"
    """background models"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for backgound"""
    periodic_tvl_mult: float = 0.0
    """Total variational loss mutliplier"""
    overwrite_near_far_plane: bool = False
    """whether to use near and far collider from command line"""
    tint_loss_mult: float = 0.0
    """Tint loss (L1 regularizer) multiplier"""

# Temporary here
from diffusers.utils import pt_to_pil, randn_tensor

import torch.nn as nn

class TrilinearIntepolation(nn.Module):
    # https://github.com/tedyhabtegebrial/PyTorch-Trilinear-Interpolation/blob/master/interpolation.py
    """TrilinearIntepolation in PyTorch."""

    def __init__(self):
        super(TrilinearIntepolation, self).__init__()

    def sample_at_integer_locs(self, input_feats, index_tensor):
        assert input_feats.ndimension()==5, 'input_feats should be of shape [B,F,D,H,W]'
        assert index_tensor.ndimension()==4, 'index_tensor should be of shape [B,H,W,3]'
        # first sample pixel locations using nearest neighbour interpolation
        batch_size, num_chans, num_d, height, width = input_feats.shape
        grid_height, grid_width = index_tensor.shape[1],index_tensor.shape[2]

        xy_grid = index_tensor[..., 0:2]
        xy_grid[..., 0] = xy_grid[..., 0] - ((width-1.0)/2.0)
        xy_grid[..., 0] = xy_grid[..., 0] / ((width-1.0)/2.0)
        xy_grid[..., 1] = xy_grid[..., 1] - ((height-1.0)/2.0)
        xy_grid[..., 1] = xy_grid[..., 1] / ((height-1.0)/2.0)
        xy_grid = torch.clamp(xy_grid, min=-1.0, max=1.0)
        sampled_in_2d = F.grid_sample(input=input_feats.view(batch_size, num_chans*num_d, height, width),
            grid=xy_grid, mode='nearest', align_corners=False).view(batch_size, num_chans, num_d, grid_height, grid_width)
        z_grid = index_tensor[..., 2].view(batch_size, 1, 1, grid_height, grid_width)
        z_grid = z_grid.long().clamp(min=0, max=num_d-1)
        z_grid = z_grid.expand(batch_size,num_chans, 1, grid_height, grid_width)
        sampled_in_3d = sampled_in_2d.gather(2, z_grid).squeeze(2)
        return sampled_in_3d


    def forward(self, input_feats, sampling_grid):
        assert input_feats.ndimension()==5, 'input_feats should be of shape [B,F,D,H,W]'
        assert sampling_grid.ndimension()==4, 'sampling_grid should be of shape [B,H,W,3]'
        batch_size, num_chans, num_d, height, width = input_feats.shape
        grid_height, grid_width = sampling_grid.shape[1],sampling_grid.shape[2]
        # make sure sampling grid lies between -1, 1
        sampling_grid = torch.clamp(sampling_grid, min=-1.0, max=1.0)
        # map to 0,1
        sampling_grid = (sampling_grid+1)/2.0
        # Scale grid to floating point pixel locations
        scaling_factor = torch.FloatTensor([width-1.0, height-1.0, num_d-1.0]).to(input_feats.device).view(1, 1, 1, 3)
        sampling_grid = scaling_factor*sampling_grid
        # Now sampling grid is between [0, w-1; 0,h-1; 0,d-1]
        x, y, z = torch.split(sampling_grid, split_size_or_sections=1, dim=3)
        x_0, y_0, z_0 = torch.split(sampling_grid.floor(), split_size_or_sections=1, dim=3)
        x_1, y_1, z_1 = x_0+1.0, y_0+1.0, z_0+1.0
        u, v, w = x-x_0, y-y_0, z-z_0
        u, v, w = map(lambda x:x.view(batch_size, 1, grid_height, grid_width).expand(
                                    batch_size, num_chans, grid_height, grid_width),  [u, v, w])

        # Get blending weights
        w_000 = (1.0-u)*(1.0-v)*(1.0-w)
        w_001 = (1.0-u)*(1.0-v)*w
        w_010 = (1.0-u)*v*(1.0-w)
        w_011 = (1.0-u)*v*w
        w_100 = u*(1.0-v)*(1.0-w)
        w_101 = u*(1.0-v)*w
        w_110 = u*v*(1.0-w)
        w_111 = u*v*w

        # Sample at integer locations
        c_000 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_0], dim=3))
        c_001 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_1], dim=3))
        c_010 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_0], dim=3))
        c_011 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_1], dim=3))
        c_100 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_0], dim=3))
        c_101 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_1], dim=3))
        c_110 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_0], dim=3))
        c_111 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_1], dim=3))

        # Calculate blended value
        c_xyz = w_000*c_000 + w_001*c_001 + w_010*c_010 + w_011*c_011 + \
                w_100*c_100 + w_101*c_101 + w_110*c_110 + w_111*c_111
        weights = torch.stack([w_000, w_001, w_010, w_011,
                               w_100, w_101, w_110, w_111], dim=-1)
        return weights, c_xyz


def render_noise(outputs, camera_ray_bundle, model, generator=None):
    distance = outputs["depth"] * camera_ray_bundle.metadata["directions_norm"]
    surface_points = camera_ray_bundle.origins + distance * camera_ray_bundle.directions
    
    if model.scene_contraction is not None:
        surface_points = model.scene_contraction(surface_points)
        surface_points = (surface_points + 2.0) / 4.0

    surface_points = surface_points * 2.0 - 1.0
    # # (N, 1, 1, H, 3)
    # surface_points = surface_points[None, None, None, ...]
    # (N, 1, W, 3)
    surface_points = surface_points[None, None, ...]

    # (N, C, H, W, D)
    generator = torch.Generator().manual_seed(0)
    grid_res = 100
    grid = randn_tensor((1, 3, grid_res, grid_res, grid_res), device=surface_points.device,
                        dtype=surface_points.dtype, generator=generator)
    
    # noise_from_gird = F.grid_sample(grid, surface_points,
    #                                 align_corners=True,
    #                                 mode="nearest") # nearest bilinear
    # # (H, 3)
    # noise_from_gird = torch.permute(noise_from_gird[0, :, 0, 0, :], (1, 0))
    # return noise_from_gird

    trilin_inter = TrilinearIntepolation()
    inter_weigths, noise_from_gird = trilin_inter(grid, surface_points)
    inter_weigths = torch.sqrt(torch.pow(inter_weigths, 2).sum(dim=-1))
    noise_from_gird_rescaled = noise_from_gird / inter_weigths
    # (H, 3)
    noise_from_gird_rescaled = torch.permute(noise_from_gird_rescaled[0, :, 0, :], (1, 0))
    return noise_from_gird_rescaled

class SurfaceModel(Model):
    """Base surface model

    Args:
        config: Base surface model configuration to instantiate model
    """

    config: SurfaceModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.scene_contraction = SceneContraction(order=float("inf"))

        # Can we also use contraction for sdf?
        # Fields
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        # Collider
        self.collider = AABBBoxCollider(self.scene_box, near_plane=0.05)

        # command line near and far has highest priority
        if self.config.overwrite_near_far_plane:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # background model
        if self.config.background_model == "grid":
            self.field_background = TCNNNerfactoField(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                # # TODO: args
                # use_appearance_embedding=False,
                # use_direction=False,
            )
        elif self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )

            self.field_background = NeRFField(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
            )
        else:
            # dummy background model
            self.field_background = Parameter(torch.ones(1), requires_grad=False)

        self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normal = SemanticRenderer()

        # losses
        self.rgb_loss = L1Loss()
        self.eikonal_loss = MSELoss()
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        else:
            param_groups["field_background"] = list(self.field_background)
        return param_groups

    @abstractmethod
    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of samples and field output.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        # shortcuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]
        bg_transmittance = samples_and_field_outputs["bg_transmittance"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.metadata["directions_norm"]

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # background model
        if self.config.background_model != "none":
            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
            depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
            accumulation_bg = self.renderer_accumulation(weights=weights_bg)

            # merge background color to forgound color
            rgb = rgb + bg_transmittance * rgb_bg

            bg_outputs = {
                "bg_rgb": rgb_bg,
                "bg_accumulation": accumulation_bg,
                "bg_depth": depth_bg,
                "bg_weights": weights_bg,
            }
        else:
            bg_outputs = {}

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            # used to scale z_vals for free space and sdf loss
            "directions_norm": ray_bundle.metadata["directions_norm"],
        }
        outputs.update(bg_outputs)

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)
            # if "tint" in field_outputs:
            #     tint = self.renderer_normal(semantics=field_outputs["tint"], weights=weights)
            #     outputs.update({"tint": tint})

        if "weights_list" in samples_and_field_outputs:
            weights_list = samples_and_field_outputs["weights_list"]
            ray_samples_list = samples_and_field_outputs["ray_samples_list"]

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0

        if "tint" in field_outputs:
            tint = self.renderer_normal(semantics=field_outputs["tint"], weights=weights)
            outputs.update({"tint": tint})

        # # it makes rendering slower
        # outputs["noise_from_grid"] = render_noise(outputs, ray_bundle, self)

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

            # foreground mask loss
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )

            # monocular normal loss
            if "normal" in batch and self.config.mono_normal_loss_mult > 0.0:
                normal_gt = batch["normal"].to(self.device)
                normal_pred = outputs["normal"]
                loss_dict["normal_loss"] = (
                    monosdf_normal_loss(normal_pred, normal_gt) * self.config.mono_normal_loss_mult
                )

            # monocular depth loss
            if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
                depth_gt = batch["depth"].to(self.device)[..., None]
                depth_pred = outputs["depth"]

                mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
                loss_dict["depth_loss"] = (
                    self.depth_loss(depth_pred.reshape(1, 32, -1), (depth_gt * 50 + 0.5).reshape(1, 32, -1), mask)
                    * self.config.mono_depth_loss_mult
                )
            if "tint" in outputs:
                loss_dict["tint_reg"] = outputs["tint"].abs().mean() * self.config.tint_loss_mult

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        Args:
            outputs: Outputs of the model.
            batch: Batch of data.

        Returns:
            A dictionary of metrics.
        """
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])

        normal = outputs["normal"]
        normal = (normal + 1.0) / 2.0

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        if "depth" in batch:
            depth_gt = batch["depth"].to(self.device)
            depth_pred = outputs["depth"]

            # align to predicted depth and normalize
            scale, shift = normalized_depth_scale_and_shift(
                depth_pred[None, ..., 0], depth_gt[None, ...], depth_gt[None, ...] > 0.0
            )
            depth_pred = depth_pred * scale + shift

            combined_depth = torch.cat([depth_gt[..., None], depth_pred], dim=1)
            combined_depth = colormaps.apply_depth_colormap(combined_depth)
        else:
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
            combined_depth = torch.cat([depth], dim=1)

        if "normal" in batch:
            normal_gt = (batch["normal"].to(self.device) + 1.0) / 2.0
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        else:
            combined_normal = torch.cat([normal], dim=1)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "normal": combined_normal,
        }

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        return metrics_dict, images_dict
