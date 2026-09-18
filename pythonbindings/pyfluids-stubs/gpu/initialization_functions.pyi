r"""
=======================================================================================
 ____          ____    __    ______     __________   __      __       __        __
 \    \       |    |  |  |  |   _   \  |___    ___| |  |    |  |     /  \      |  |
  \    \      |    |  |  |  |  |_)   |     |  |     |  |    |  |    /    \     |  |
   \    \     |    |  |  |  |   _   /      |  |     |  |    |  |   /  /\  \    |  |
    \    \    |    |  |  |  |  | \  \      |  |     |   \__/   |  /  ____  \   |  |____
     \    \   |    |  |__|  |__|  \__\     |__|      \________/  /__/    \__\  |_______|
      \    \  |    |   ________________________________________________________________
       \    \ |    |  |  ______________________________________________________________|
        \    \|    |  |  |         __          __     __     __     ______      _______
         \         |  |  |_____   |  |        |  |   |  |   |  |   |   _  \    /  _____)
          \        |  |   _____|  |  |        |  |   |  |   |  |   |  | \  \   \_______
           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  |
            \ _____|  |__|        |________|   \_______/    |__|   |______/    (_______/

  This file is part of VirtualFluids. VirtualFluids is free software: you can
  redistribute it and/or modify it under the terms of the GNU General Public
  License as published by the Free Software Foundation, either version 3 of
  the License, or (at your option) any later version.

  VirtualFluids is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
  for more details.

  SPDX-License-Identifier: GPL-3.0-or-later
  SPDX-FileCopyrightText: Copyright © VirtualFluids Project contributors, see AUTHORS.md in root folder

! \author Henry Korb
=======================================================================================
"""

from collections.abc import Callable
from typing import TypeAlias

from numpy import typing as npt

InitFunc: TypeAlias = Callable[[float, float, float], tuple[float, float, float, float]]
InitFuncAD: TypeAlias = Callable[[float, float, float], float]

def uniform(
    density: float, velocity_x: float, velocity_y: float, velocity_z: float, velocity_ratio: float
) -> InitFunc: ...
def log_law(
    friction_velocity: float, von_karman_constant: float, roughness_length: float, velocity_ratio: float
) -> InitFunc: ...
def perturbed_log_law(
    friction_velocity: float,
    von_karman_constant: float,
    roughness_length: float,
    length: float,
    width: float,
    height: float,
    boundary_layer_height: float,
    velocity_ratio: float,
) -> InitFunc: ...
def profile_Zilitenkevich(
    friction_velocity: float,
    von_karman_constant: float,
    roughness_length: float,
    inversion_height: float,
    merging_region_thickness: float,
    geostrophic_wind_speed: float,
    geostrophic_wind_direction: float,
    velocity_ratio,
) -> InitFunc: ...
def taylor_green_vortex(velocity_LB: float) -> InitFunc: ...
def uniform_AD(concentration: float) -> InitFuncAD: ...
def constant_gradient(temperature_surface: float, gradient: float) -> InitFuncAD: ...
def profile_Rampanelli_and_Zardi(
    surface_temperature: float,
    inversion_base_height: float,
    inversion_thickness: float,
    inversion_strength: float,
    lapse_rate: float,
) -> InitFuncAD: ...
def two_layers(temperature_surface: float, inversion_height: float, gradient: float) -> InitFuncAD: ...
def interpolate_data_AD(coordinates: npt.ArrayLike, temperatures: npt.ArrayLike) -> InitFuncAD: ...
def add_random_perturbation(
    unperturbed_func: InitFuncAD, perturbation_strength: float, maximum_height_of_perturbation: float
) -> InitFuncAD: ...
