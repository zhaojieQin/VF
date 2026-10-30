#include <gmock/gmock.h>

#include <array>

#include <lbm/advectionDiffusion/BoundaryConditions.h>
#include <lbm/advectionDiffusion/Equilibrium.h>
#include <lbm/constants/D3Q27.h>

using namespace vf::lbm::advection_diffusion;
using namespace vf::lbm::dir;

namespace
{

std::array<real, 27> uniformEquilibrium(real concentration, real vx, real vy, real vz)
{
    std::array<real, 27> g {};
    forEachDirection([&](auto direction) { g[direction] = computeEquilibrium<direction>(concentration, vx, vy, vz); });
    return g;
}

const real vx = 0.08, vy = -0.05, vz = 0.02;

} // namespace

// A uniform concentration in a uniform flow is an exact solution: the boundary must return the equilibrium of the
// incoming direction, for every direction and subgrid distance q in [1/2, 1].
TEST(AdvectionDiffusionBoundaryConditionsTest, interpolatedConsistentKeepsUniformEquilibrium)
{
    const real concentration = 2.5;
    const auto g = uniformEquilibrium(concentration, vx, vy, vz);
    for (const real q : { real(0.5), real(0.75), real(1.0) }) {
        forEachNonRestDirection([&](auto direction) {
            const real result =
                computePopulationInterpolatedAntiBounceBackConsistent<direction>(q, g.data(), concentration, vx, vy, vz);
            EXPECT_NEAR(result, g[inverseDir<direction>()], 1e-6) << "direction " << direction << ", q " << q;
        });
    }
}

// Adding a constant to the concentration everywhere (node and wall) must shift the returned population by the
// equilibrium of that constant, independent of the reference the concentration is stored against.
TEST(AdvectionDiffusionBoundaryConditionsTest, interpolatedConsistentIsShiftInvariant)
{
    const real nodeConcentration = 1.0, wallConcentration = 3.0, shift = 10.0, q = 0.5;
    const auto g = uniformEquilibrium(nodeConcentration, vx, vy, vz);
    const auto gShift = uniformEquilibrium(shift, vx, vy, vz);
    std::array<real, 27> gShifted {};
    for (size_t i = 0; i < 27; i++)
        gShifted[i] = g[i] + gShift[i];
    forEachNonRestDirection([&](auto direction) {
        const real base =
            computePopulationInterpolatedAntiBounceBackConsistent<direction>(q, g.data(), wallConcentration, vx, vy, vz);
        const real shifted = computePopulationInterpolatedAntiBounceBackConsistent<direction>(
            q, gShifted.data(), wallConcentration + shift, vx, vy, vz);
        EXPECT_NEAR(shifted - base, gShift[inverseDir<direction>()], 1e-5) << "direction " << direction;
    });
}

// For q = 1/2 the scheme is the anti-bounce-back with the even equilibrium part: -g_i + 2 e_i^+(c_wall).
TEST(AdvectionDiffusionBoundaryConditionsTest, interpolatedConsistentIsAntiBounceBackForHalfDistance)
{
    const auto g = uniformEquilibrium(1.0, vx, vy, vz);
    const real wallConcentration = 4.0;
    forEachNonRestDirection([&](auto direction) {
        const real expected = -g[direction] + 2 * computeEvenEquilibrium<direction>(wallConcentration, vx, vy, vz);
        EXPECT_NEAR(computePopulationInterpolatedAntiBounceBackConsistent<direction>(0.5, g.data(), wallConcentration,
                                                                                     vx, vy, vz),
                    expected, 1e-6);
    });
}
