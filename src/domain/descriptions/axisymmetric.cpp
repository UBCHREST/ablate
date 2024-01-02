#include "axisymmetric.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::domain::descriptions::Axisymmetric::Axisymmetric(std::vector<PetscReal> startLocation, std::vector<PetscReal> endLocation, PetscInt numberWedges, PetscInt numberSlices)
    : startLocation(utilities::VectorUtilities::ToArray<PetscReal, 3>(startLocation)),
      endLocation(utilities::VectorUtilities::ToArray<PetscReal, 3>(endLocation)),
      numberWedges(numberWedges),
      numberSlices(numberSlices),
      numberCells(numberWedges * numberSlices),
      numberVertices((numberWedges + 1) * (numberSlices + 1))

{}
void ablate::domain::descriptions::Axisymmetric::BuildTopology(PetscInt cell, PetscInt *cellNodes) const {}
