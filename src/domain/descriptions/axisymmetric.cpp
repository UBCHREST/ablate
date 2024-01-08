#include "axisymmetric.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::domain::descriptions::Axisymmetric::Axisymmetric(const std::vector<PetscReal> &startLocation, PetscReal length, PetscInt numberWedges, PetscInt numberSlices)
    : startLocation(utilities::VectorUtilities::ToArray<PetscReal, 3>(startLocation)),
      length(length),
      numberWedges(numberWedges),
      numberSlices(numberSlices),
      numberCellsPerSlice(numberWedges),  // TODO: expand for more cells in radius
      numberVerticesPerHalfSlice(numberWedges + 1),
      numberCells(numberWedges * numberSlices),
      numberVertices((numberWedges + 1) * (numberSlices + 1))

{
    // make sure there are at least 4 wedges and one slice
    if (numberWedges < 4 || numberSlices < 1) {
        throw std::invalid_argument("Axisymmetric requires at least 4 wedges and 1 slice.");
    }
}

void ablate::domain::descriptions::Axisymmetric::BuildTopology(PetscInt cell, PetscInt *cellNodes) const {
    // determine which slice this is
    PetscInt slice = cell / numberCellsPerSlice;

    // determine which local wedge this is
    PetscInt localWedge = cell % numberWedges;

    // Set the center nodes
    const auto lowerCenter = slice * numberVerticesPerHalfSlice;
    const auto upperCenter = (slice + 1) * numberVerticesPerHalfSlice;
    cellNodes[0] = lowerCenter;
    cellNodes[3] = upperCenter;

    // set the lower nodes
    cellNodes[1] = localWedge + 1 + lowerCenter;
    cellNodes[2] = (localWedge + 2) == (numberWedges + 1) ? lowerCenter + 1 : lowerCenter + 2 + localWedge;  // checking for wrap around

    // repeat for the upper nodes
    cellNodes[5] = localWedge + 1 + upperCenter;
    cellNodes[4] = (localWedge + 2) == (numberWedges + 1) ? upperCenter + 1 : upperCenter + 2 + localWedge;  // checking for wrap around
}
void ablate::domain::descriptions::Axisymmetric::SetCoordinate(PetscInt node, PetscReal *coordinate) const {
    // start the coordinate at the start location
    coordinate[0] = startLocation[0];
    coordinate[1] = startLocation[1];
    coordinate[2] = startLocation[2];

    // start by determining offset in z
    auto localRow = node / numberVerticesPerHalfSlice;
    auto delta = length / numberSlices;
    auto offset = delta * localRow;
    coordinate[2] += offset;

    // determine the local node number
    auto localNode = node % numberVerticesPerHalfSlice;

    // if we are not at the center
    if (localNode != 0) {
        coordinate[0] += PetscCosReal(2.0 * (localNode - 1) * PETSC_PI / numberWedges);
        coordinate[1] += PetscSinReal(2.0 * (localNode - 1) * PETSC_PI / numberWedges);
    }
}
