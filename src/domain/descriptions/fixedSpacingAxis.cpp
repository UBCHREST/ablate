#include "fixedSpacingAxis.hpp"

#include "utilities/vectorUtilities.hpp"

ablate::domain::descriptions::FixedSpacingAxis::FixedSpacingAxis(const std::vector<PetscReal> &startLocation, PetscReal length, PetscInt numberNodes)
    : startLocation(utilities::VectorUtilities::ToArray<PetscReal, 3>(startLocation)), length(length), numberNodes(numberNodes) {
    // make sure there are at least 2 nodes
    if (this->numberNodes < 2) {
        throw std::invalid_argument("FixedSpacingAxis requires at least 2 nodes");
    }
}

void ablate::domain::descriptions::FixedSpacingAxis::SetCoordinate(PetscInt node, PetscReal coordinate[3]) const {
    // start the coordinate at the start location
    coordinate[0] = startLocation[0];
    coordinate[1] = startLocation[1];
    coordinate[2] = startLocation[2];

    // offset along z
    auto delta = length / (numberNodes - 1);
    auto offset = delta * node;
    coordinate[2] += offset;
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::domain::descriptions::AxisDescription, ablate::domain::descriptions::FixedSpacingAxis, "Creates a simple fixed spacing along z axis",
                 ARG(std::vector<double>, "start", "the start coordinate of the mesh, must be 3D"), ARG(double, "length", "the length of the domain starting at the start coordinate"),
                 ARG(int, "nodes", "number of nodes along the z-axis"));
