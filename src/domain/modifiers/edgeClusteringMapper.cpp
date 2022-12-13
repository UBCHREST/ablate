#include "edgeClusteringMapper.hpp"
#include "mathFunctions/functionFactory.hpp"

ablate::domain::modifiers::EdgeClusteringMapper::EdgeClusteringMapper(int direction, double startIn, double end, double beta)
    : ablate::domain::modifiers::MeshMapper(mathFunctions::Create(MappingFunction, this)), direction(direction), start(startIn), size(end - startIn), beta(beta) {
    // make sure that the direction is valid
    if (direction < 0 || direction > 2) {
        throw std::invalid_argument("The direction must be 0, 1, or 2. Direction " + std::to_string(direction) + " is invalid.");
    }
}
std::string ablate::domain::modifiers::EdgeClusteringMapper::ToString() const {
    return "ablate::domain::modifiers::EdgeClusteringMapper Dir(" + std::to_string(direction) + ") Start(" + std::to_string(start) + ") Size(" + std::to_string(size) + ") Beta(" +
           std::to_string(beta) + ")";
}

PetscErrorCode ablate::domain::modifiers::EdgeClusteringMapper::MappingFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto map = (EdgeClusteringMapper *)ctx;

    // start by copying everything
    for (PetscInt d = 0; d < dim; d++) {
        u[d] = x[d];
    }

    u[map->direction] -= map->start;
    PetscReal eta = u[map->direction] / map->size;
    PetscReal term1 = (map->beta + 1) - (map->beta - 1) * PetscPowReal((map->beta + 1) / (map->beta - 1), 1 - eta);
    PetscReal term2 = PetscPowReal((map->beta + 1) / (map->beta - 1), 1 - eta) + 1.;
    PetscReal newLocation = map->size * term1 / term2;
    if (!PetscIsInfOrNanReal(u[map->direction])) {
        u[map->direction] = newLocation;
    }
    u[map->direction] += map->start;
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::EdgeClusteringMapper,
         "Performs clustering mapping using an algebraic relationship at the edges of the domain using Equation 9-42 from Hoffmann, Klaus A., and Steve T. Chiang. \"Computational fluid dynamics "
         "volume I. Forth Edition\" Engineering education system (2000).",
         ARG(int, "direction", "The direction (0, 1, 2) to perform the mapping"), ARG(double, "start", "The start of the domain in direction"),
         ARG(double, "end", "The end of the domain in direction"), ARG(double, "beta", "The clustering factor (1 -> infinity, where 1 is more clustering)"));
