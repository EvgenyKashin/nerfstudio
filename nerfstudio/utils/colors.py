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

"""Common Colors"""
from typing import Union

import torch
from torchtyping import TensorType

WHITE = torch.tensor([1.0, 1.0, 1.0])
BLACK = torch.tensor([0.0, 0.0, 0.0])
RED = torch.tensor([1.0, 0.0, 0.0])
GREEN = torch.tensor([0.0, 1.0, 0.0])
BLUE = torch.tensor([0.0, 0.0, 1.0])
BLACK_LATENT = torch.tensor([-1.0674, -2.6035,  1.0479,  1.3145])
WHITE_LATENT = torch.tensor([1.8291,  1.5029, -0.0822, -1.1885])

COLORS_DICT = {
    "white": WHITE,
    "black": BLACK,
    "red": RED,
    "green": GREEN,
    "blue": BLUE,
    "black_latent": BLACK_LATENT,
    "white_latent": WHITE_LATENT,
}


def get_color(color: Union[str, list]) -> Union[TensorType[3], TensorType[4]]:
    """
    Args:
        color (Union[str, list]): Color as a string or a rgb list

    Returns:
        TensorType[3]: Parsed color
    """
    if isinstance(color, str):
        color = color.lower()
        if color not in COLORS_DICT:
            raise ValueError(f"{color} is not a valid preset color")
        return COLORS_DICT[color]
    if isinstance(color, list):
        if len(color) not in [3, 4]:
            raise ValueError(f"Color should be 3 or 4 values (RGB or latents) instead got {color}")
        return torch.tensor(color)

    raise ValueError(f"Color should be an RGB list or string, instead got {type(color)}")
