
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

from __future__ import annotations

from enum import Enum
from .. import gpu
from ..basics.geometry3d import Axis

class Probe(gpu.Sampler):
    class Statistic(Enum):
        _value_: int
        Instantaneous = ...
        Means = ...
        Variances = ...
    def __init__(
        self,
        para: gpu.Parameter,
        cuda_memory_manager: gpu.CudaMemoryManager,
        output_path: str,
        probe_name: str,
        t_start_avg: int,
        t_avg: int,
        t_start_out: int,
        t_out: int,
        output_timeseries: bool,
        average_every_timestep: bool,
        sample_scalar: bool = ...
    ) -> None: ...
    def add_statistic(self, variable: Statistic) -> None: ...
    def add_all_available_statistics(self) -> None: ...
    def add_probe_point(self, point_coord_x: float, point_coord_y: float, point_coord_z: float) -> None: ...
    def add_probe_points_from_list(
        self, point_coords_x: list[float], point_coords_y: list[float], point_coords_z: list[float]
    ) -> None: ...
    def set_probe_plane(
        self, pos_x: float, pos_y: float, pos_z: float, delta_x: float, delta_y: float, delta_z: float
    ) -> None: ...
    def set_file_name_to_n_out(self) -> None: ...

class PlanarAverageProbe(gpu.Sampler):
    class Statistic(Enum):
        _value_: int
        Means = ...
        Covariances = ...
        Skewness = ...
        Flatness = ...
    def __init__(
        self,
        para: gpu.Parameter,
        cuda_memory_manager: gpu.CudaMemoryManager,
        probe_name: str,
        output_path: str,
        t_start_avg: int,
        t_start_tmp_avg: int,
        t_avg: int,
        t_start_out: int,
        t_out: int,
        plane_normal: Axis,
        compute_time_averages: bool,
        sample_scalar: bool = ...,
        sample_subgrid_scale_fluxes: bool = ...,
    ) -> None: ...
    def add_statistic(self, variable: Statistic) -> None: ...
    def add_all_available_statistics(self) -> None: ...
    def set_file_name_to_n_out(self) -> None: ...

class WallModelProbe(gpu.Sampler):
    def __init__(
        self,
        para: gpu.Parameter,
        cuda_memory_manager: gpu.CudaMemoryManager,
        output_path: str,
        probe_name: str,
        t_start_avg: int,
        t_start_tmp_avg: int,
        t_avg: int,
        t_start_out: int,
        t_out: int,
        average_every_timestep: bool,
        compute_temporal_averages: bool,
        output_stress: bool,
        evaluate_pressure_gradient: bool,
        sample_surface_layer: bool,
    ) -> None: ...