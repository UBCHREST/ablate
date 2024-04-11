#include "twoPointClusteringMapper.hpp"

#include <utility>
#include "mathFunctions/functionFactory.hpp"
#include "utilities/constants.hpp"

ablate::domain::modifiers::TwoPointClusteringMapper::TwoPointClusteringMapper(int direction, double startIn, double end, double beta, double locationIn, double offset,
                                                                              std::shared_ptr<ablate::domain::Region> mappingRegion)
    : ablate::domain::modifiers::MeshMapper(mathFunctions::Create(MappingFunction, this), std::move(mappingRegion)),
      direction(direction),
      start(startIn),
      size(PetscAbsReal(end - startIn)),
      beta(beta),
      location(PetscAbsReal(locationIn) < ablate::utilities::Constants::small ? ablate::utilities::Constants::small : locationIn),
      offset(PetscAbsReal(offset)) {
    // make sure that the direction is valid
    if (direction < 0 || direction > 2) {
        throw std::invalid_argument("The direction must be 0, 1, or 2. Direction " + std::to_string(direction) + " is invalid.");
    }
}
std::string ablate::domain::modifiers::TwoPointClusteringMapper::ToString() const {
    return "ablate::domain::modifiers::TwoPointClusteringMapper Dir(" + std::to_string(direction) + ") Start(" + std::to_string(start) + ") Size(" + std::to_string(size) + ") Beta(" +
           std::to_string(beta) + ") Loc(" + std::to_string(location) + ") Offset(" + std::to_string(offset) + ")";
}

PetscErrorCode ablate::domain::modifiers::TwoPointClusteringMapper::MappingFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto map = (TwoPointClusteringMapper *)ctx;

    // start by copying everything
    for (PetscInt d = 0; d < dim; d++) {
        u[d] = x[d];
    }

    // Find the size of the reduced domain
    PetscReal xScalar = x[map->direction] - map->location;

    // Determine if left or right clustering
    PetscReal sizeTmp;
    if (u[map->direction] > map->location) {
        sizeTmp = 2.0 * PetscAbsReal(map->start + map->size - map->location);
    } else {
        sizeTmp = 2.0 * PetscAbsReal(map->start - map->location);
    }

    // normalize to -1 -> 1
    PetscReal xtmp = PetscAbsReal(xScalar / (.5 * sizeTmp)) * PetscSignReal(xScalar);
    PetscReal term1 = 1 + (PetscExpReal(map->beta) - 1) * map->offset * 2. / sizeTmp;   // size[dir];
    PetscReal term2 = 1 + (PetscExpReal(-map->beta) - 1) * map->offset * 2. / sizeTmp;  // size[dir];
    PetscReal A = (1.0 / (2.0 * map->beta)) * PetscLogReal(term1 / term2);
    u[map->direction] = (2 * map->offset) * (1 + PetscSinhReal(map->beta * (PetscAbsReal(xtmp) - A)) / PetscSinhReal(map->beta * A)) * xtmp / (PetscAbsReal(xtmp) + ablate::utilities::Constants::tiny);
    u[map->direction] /= 2.;
    u[map->direction] += map->location;

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::TwoPointClusteringMapper,
         "Performs clustering mapping using an algebraic relationship around two point using equations derived from Hoffmann, Klaus A., and Steve T. Chiang. \"Computational fluid dynamics volume I. "
         "Forth Edition\" Engineering education system (2000).",
         ARG(int, "direction", "The direction (0, 1, 2) to perform the mapping"), ARG(double, "start", "The start of the domain in direction"),
         ARG(double, "end", "The end of the domain in direction"), ARG(double, "beta", "The clustering factor (0 -> infinity, where infinity is more clustering)"),
         ARG(double, "location", "The location to cluster center"), ARG(double, "offset", "The offset from the location center to perform the clustering"),
         OPT(ablate::domain::Region, "region", "optional region to apply this mapper (Default is everywhere)"));
