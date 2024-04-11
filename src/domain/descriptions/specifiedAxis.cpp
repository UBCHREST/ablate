#include "specifiedAxis.hpp"

#include <algorithm>
#include "utilities/vectorUtilities.hpp"

ablate::domain::descriptions::SpecifiedAxis::SpecifiedAxis(const std::vector<PetscReal>& nodeOffsets, const std::vector<PetscReal>& startLocation)
    : startLocation(startLocation.empty() ? std::array<PetscReal, 3>{0.0, 0.0, 0.0} : utilities::VectorUtilities::ToArray<PetscReal, 3>(startLocation)),
      zOffset(nodeOffsets),
      numberNodes((PetscInt)nodeOffsets.size()) {
    // make sure there are at least 2 nodes
    if (this->numberNodes < 2) {
        throw std::invalid_argument("SpecifiedAxis requires at least 2 nodes");
    }
    if (!std::is_sorted(zOffset.begin(), zOffset.end())) {
        throw std::invalid_argument("SpecifiedAxis requires zOffset vector be sorted from smallest to largest");
    }
}

void ablate::domain::descriptions::SpecifiedAxis::SetCoordinate(PetscInt node, PetscReal coordinate[3]) const {
    // start the coordinate at the start location
    coordinate[0] = startLocation[0];
    coordinate[1] = startLocation[1];
    coordinate[2] = startLocation[2];

    // offset along z
    coordinate[2] += zOffset[node];
}

#include "registrar.hpp"
REGISTER(ablate::domain::descriptions::AxisDescription, ablate::domain::descriptions::SpecifiedAxis, "Describes a simple strait line along the z axis with specified offsets",
         ARG(std::vector<double>, "offsets", "an ordered list of offsets in z"), OPT(std::vector<double>, "start", "the start coordinate of the mesh, must be 3D"));
