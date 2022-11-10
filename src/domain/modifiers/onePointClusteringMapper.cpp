#include "onePointClusteringMapper.hpp"
#include "mathFunctions/functionFactory.hpp"

ablate::domain::modifiers::OnePointClusteringMapper::OnePointClusteringMapper(int direction, double startIn, double end, double beta, double locationIn)
    : ablate::domain::modifiers::MeshMapper(mathFunctions::Create(MappingFunction, this)),
      direction(direction),
      start(startIn),
      size(PetscAbsReal(end - startIn)),
      beta(beta),
      location(locationIn - startIn) {
    // make sure that the direction is valid
    if (direction < 0 || direction > 2) {
        throw std::invalid_argument("The direction must be 0, 1, or 2. Direction " + std::to_string(direction) + " is invalid.");
    }
}
std::string ablate::domain::modifiers::OnePointClusteringMapper::ToString() const {
    return "ablate::domain::modifiers::OnePointClusteringMapper Dir(" + std::to_string(direction) + ") Start(" + std::to_string(start) + ") Size(" + std::to_string(size) + ") Beta(" +
           std::to_string(beta) + ") Loc(" + std::to_string(location) + ")";
}

PetscErrorCode ablate::domain::modifiers::OnePointClusteringMapper::MappingFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto map = (OnePointClusteringMapper *)ctx;

    // start by copying everything
    for (PetscInt d = 0; d < dim; d++) {
        u[d] = x[d];
    }

    // Get the org location
    PetscReal term1 = 1 + (PetscExpReal(map->beta) - 1.0) * map->location / map->size;
    PetscReal term2 = 1 + (PetscExpReal(-map->beta) - 1.0) * map->location / map->size;
    PetscReal A = (1.0 / (2.0 * map->beta)) * PetscLogReal(term1 / term2);
    u[map->direction] -= map->start;
    PetscReal eta = u[map->direction] / map->size;
    PetscReal newLocation = map->location * (1.0 + (PetscSinhReal(map->beta * (eta - A))) / PetscSinhReal(map->beta * A));
    if (!PetscIsInfOrNanReal(u[map->direction])) {
        u[map->direction] = newLocation;
    }
    u[map->direction] += map->start;
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(
    ablate::domain::modifiers::Modifier, ablate::domain::modifiers::OnePointClusteringMapper,
    "Performs clustering mapping using an algebraic relationship around one point using Equation 9-50 from Hoffmann, Klaus A., and Steve T. Chiang. \"Computational fluid dynamics volume I. "
    "Forth Edition\" Engineering education system (2000). $$x'=D \\left [ 1+\\frac{sinh[\\beta(x-A))]}{sinh(\\beta A)} \\right ]$$ where $$ A=\\frac{1}{2 \\beta}ln \\left [  \\frac{1+(e^\\beta - "
    "1)(D/H)}{1+(e^{-\\beta} - 1)(D/H)} \\right ] $$, $$ D = $$cluster location, and  $$\\beta = $$ cluster factor",
    ARG(int, "direction", "The direction (0, 1, 2) to perform the mapping"), ARG(double, "start", "The start of the domain in direction"), ARG(double, "end", "The end of the domain in direction."),
    ARG(double, "beta", "The clustering factor."), ARG(double, "location", "The location to perform the clustering in direction."));
