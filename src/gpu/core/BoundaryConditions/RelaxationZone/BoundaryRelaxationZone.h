#ifndef BOUNDARY_RELAXATION_ZONE_H
#define BOUNDARY_RELAXATION_ZONE_H

#include <vector>

#include <basics/DataTypes.h>

namespace vf::gpu {

class Parameter;

//! \brief Lateral relaxation (Davies / nudging) zone for one-way nesting of the LBM in a mesoscale model.
//!
//! Every fluid node within \p width of a selected domain face is relaxed towards a time-interpolated
//! target state:  du = r(d) * rate * (u_target - u)  as a body force (forceX/Y/Z_SP, consumed by the next
//! collision), and  dtheta = r(d) * rate * (theta_target - theta)  as an increment of the
//! advection-diffusion distributions (g_i += w_i dtheta (1 + 3 c_i.u)).  r(d) = cos^2(pi d / (2 width)) is 1 at
//! the face and 0 at the inner edge of the zone; d is the distance of the node to the closest selected face.
//!
//! Targets are held for two frames (t0, t1); pushTargets() shifts t1 -> t0 and stores the new frame in t1, and
//! apply(alpha, ...) relaxes towards (1 - alpha) * t0 + alpha * t1.  apply() is called between two time steps
//! (like CellPerturbation), so the pre-collision distributions it modifies are the ones the next collision reads.
class BoundaryRelaxationZone
{
public:
    enum Face : uint { MX = 1u, PX = 2u, MY = 4u, PY = 8u, PZ = 16u };

    //! \param xMin..zMax  positions of the domain faces (physical units, as the node coordinates)
    //! \param width       width of the lateral zone (physical units)
    //! \param topWidth    width of the zone below the top face (only if faces contains PZ)
    //! \param faces       bit mask of Face values
    BoundaryRelaxationZone(Parameter* para, int level, real xMin, real xMax, real yMin, real yMax, real zMax, real width,
                           real topWidth, uint faces);
    ~BoundaryRelaxationZone();

    BoundaryRelaxationZone(const BoundaryRelaxationZone&) = delete;
    BoundaryRelaxationZone& operator=(const BoundaryRelaxationZone&) = delete;

    //! velocity in lattice units, theta in concentration units (K); theta may be nullptr (temperature not relaxed)
    void pushTargets(const real* vx, const real* vy, const real* vz, const real* theta);

    //! \param alpha  time interpolation weight between the two stored frames (0: t0, 1: t1)
    //! \param rate   relaxation rate at the face per time step, dt / tau
    void apply(Parameter* para, real alpha, real rate, bool relaxVelocity, bool relaxTemperature);

    uint getNumberOfNodes() const { return numberOfNodes; }
    int getLevel() const { return level; }
    uint getNumberOfPushedFrames() const { return numberOfPushedFrames; }
    const std::vector<real>& getCoordinatesX() const { return coordinatesX; }
    const std::vector<real>& getCoordinatesY() const { return coordinatesY; }
    const std::vector<real>& getCoordinatesZ() const { return coordinatesZ; }
    const std::vector<real>& getWeights() const { return weights; }

private:
    int level;
    uint numberOfNodes = 0;
    uint numberOfPushedFrames = 0;
    bool hasTemperatureTarget = false;
    std::vector<real> coordinatesX, coordinatesY, coordinatesZ, weights;
    uint* nodeIndicesD = nullptr;
    real* weightsD = nullptr;
    real* target0D = nullptr; //!< [vx | vy | vz | theta], 4 * numberOfNodes
    real* target1D = nullptr;
};

} // namespace vf::gpu

#endif
