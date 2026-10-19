//=======================================================================================
//  This file is part of VirtualFluids. VirtualFluids is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//  SPDX-FileCopyrightText: Copyright © VirtualFluids Project contributors, see AUTHORS.md in root folder
//
//! \addtogroup gpu_BoundaryConditions BoundaryConditions
//! \ingroup gpu_core core
//! \{
//=======================================================================================
#ifndef CELL_PERTURBATION_H
#define CELL_PERTURBATION_H

#include <basics/DataTypes.h>

namespace vf::gpu {

class Parameter;

//! \brief Cell perturbation method (CPM) for inflow turbulence generation
//! (Munoz-Esparza et al. 2014, 2015, 2017; Munoz-Esparza & Kosovic 2018).
//!
//! Nodes are grouped into perturbation cells once (host-side lookup from node coordinates).
//! At each refresh time the caller supplies one value per cell and the perturbation is
//! seeded instantaneously into the pre-collision distributions:
//!   temperature: g_i += w_i theta' (1 + 3 c_i.u)       (advection-diffusion distributions)
//!   momentum:    f_i += w_i 3 rho c_i.u'  (u' = (u', v', 0), optional)
//! Nothing is done unless apply() is called, so existing simulations are unaffected.
class CellPerturbation
{
public:
    //! \param cellTable  cell id for each (ix, iy, iz) with index (iz * nyc + iy) * nxc + ix, -1 = no cell.
    //!                   ix = floor((x - x0) / cellDx), iy = floor((y - y0) / cellDy), iz = round((z - z0) / dz)
    CellPerturbation(Parameter* para, int level, real x0, real y0, real z0, real cellDx, real cellDy, real dz,
                     const int* cellTable, uint nxc, uint nyc, uint nzc, uint numberOfCells);
    ~CellPerturbation();

    CellPerturbation(const CellPerturbation&) = delete;
    CellPerturbation& operator=(const CellPerturbation&) = delete;

    //! \param thetaPrime  temperature perturbation per cell (K, same units as the AD concentration), may be nullptr
    //! \param uPrime, vPrime  horizontal velocity perturbation per cell in lattice units, may be nullptr (no momentum perturbation)
    void apply(Parameter* para, const real* thetaPrime, const real* uPrime, const real* vPrime);

    uint getNumberOfNodes() const { return numberOfNodes; }
    uint getNumberOfCells() const { return numberOfCells; }
    int getLevel() const { return level; }

private:
    int level;
    uint numberOfNodes = 0;
    uint numberOfCells = 0;
    uint* nodeIndicesD = nullptr;
    uint* cellIdsD = nullptr;
    real* thetaPrimeD = nullptr;
    real* uPrimeD = nullptr;
    real* vPrimeD = nullptr;
};

} // namespace vf::gpu

#endif

//! \}
