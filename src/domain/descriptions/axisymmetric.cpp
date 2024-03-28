#include "axisymmetric.hpp"

#include <utility>
#include "utilities/vectorUtilities.hpp"

ablate::domain::descriptions::Axisymmetric::Axisymmetric(std::shared_ptr<ablate::domain::descriptions::AxisDescription> axis, std::shared_ptr<ablate::mathFunctions::MathFunction> radiusFunction,
                                                         PetscInt numberWedges, PetscInt numberShells, std::shared_ptr<ablate::mathFunctions::MathFunction> boundaryFunction)
    : axisDescription(std::move(axis)),
      radiusFunction(std::move(radiusFunction)),
      numberWedges(numberWedges),
      numberSlices(axisDescription->GetNumberVertices() - 1),
      numberShells(numberShells),
      numberCellsPerSlice(numberWedges * numberShells),
      numberCellsPerShell(numberWedges * numberSlices),
      numberVerticesPerShell(numberWedges * (numberSlices + 1)),
      numberCenterVertices(numberSlices + 1),
      numberCells(numberCellsPerSlice * numberSlices),
      numberVertices(numberVerticesPerShell * numberShells + numberCenterVertices),
      numberTriPrismCells(numberWedges * numberSlices),
      boundaryFunction(std::move(boundaryFunction)) {
    // make sure there are at least 4 wedges and one slice
    if (numberWedges < 3 || numberSlices < 1 || numberShells < 1) {
        throw std::invalid_argument("Axisymmetric requires at least 3 wedges, 1 slice, and 1 shell.");
    }
}

void ablate::domain::descriptions::Axisymmetric::BuildTopology(PetscInt cell, PetscInt *cellNodes) const {
    // Note that the cell/vertex ordering grows in shells so that tri prism elements are specified first and together.
    // flip the cell order so that hexes appear before tri prisms
    cell = CellReverser(cell);

    // determine the type of element
    if (cell >= numberTriPrismCells) {
        // this is a hex cell
        // determine which shell this is
        PetscInt cellShell = cell / numberCellsPerShell;

        // determine which slice this cell falls upon
        PetscInt cellSlice = (cell - cellShell * numberCellsPerShell) / numberWedges;

        // determine the local cell index (0 - numberWedges)
        PetscInt cellIndex = cell % numberWedges;

        PetscInt lowerSliceLowerShellOffset = (cellShell - 1) * numberVerticesPerShell + numberCenterVertices + cellSlice * numberWedges;
        PetscInt lowerSliceUpperShellOffset = cellShell * numberVerticesPerShell + numberCenterVertices + cellSlice * numberWedges;
        PetscInt upperSliceLowerShellOffset = (cellShell - 1) * numberVerticesPerShell + numberCenterVertices + (cellSlice + 1) * numberWedges;
        PetscInt upperSliceUpperShellOffset = cellShell * numberVerticesPerShell + numberCenterVertices + (cellSlice + 1) * numberWedges;

        cellNodes[0] = upperSliceLowerShellOffset + cellIndex;
        cellNodes[1] = lowerSliceLowerShellOffset + cellIndex;
        cellNodes[2] = lowerSliceUpperShellOffset + cellIndex;
        cellNodes[3] = upperSliceUpperShellOffset + cellIndex;

        cellNodes[4] = (cellIndex + 1 == numberWedges) ? upperSliceLowerShellOffset : upperSliceLowerShellOffset + cellIndex + 1 /*check for wrap around*/;
        cellNodes[5] = (cellIndex + 1 == numberWedges) ? upperSliceUpperShellOffset : upperSliceUpperShellOffset + cellIndex + 1 /*check for wrap around*/;
        cellNodes[6] = (cellIndex + 1 == numberWedges) ? lowerSliceUpperShellOffset : lowerSliceUpperShellOffset + cellIndex + 1 /*check for wrap around*/;
        cellNodes[7] = (cellIndex + 1 == numberWedges) ? lowerSliceLowerShellOffset : lowerSliceLowerShellOffset + cellIndex + 1 /*check for wrap around*/;
    } else {
        // This is a tri prism
        // determine which slice this is
        PetscInt slice = cell / numberWedges;

        // determine the local cell/wedge number (in this slice)
        PetscInt localWedge = cell % numberWedges;

        // Set the center nodes
        const auto lowerCenter = slice;
        const auto upperCenter = (slice + 1);

        cellNodes[0] = lowerCenter;
        cellNodes[3] = upperCenter;

        // Compute the offset for this slice
        const auto nodeSliceOffset = numberCenterVertices + numberWedges * slice;
        const auto nextNodeSliceOffset = numberCenterVertices + numberWedges * (slice + 1);

        // set the lower nodes
        cellNodes[2] = localWedge + nodeSliceOffset;
        cellNodes[1] = (localWedge + 1) == numberWedges ? nodeSliceOffset : localWedge + 1 + nodeSliceOffset;  // checking for wrap around

        // repeat for the upper nodes
        cellNodes[5] = (localWedge + 1) == numberWedges ? nextNodeSliceOffset : localWedge + 1 + nextNodeSliceOffset;  // checking for wrap around
        cellNodes[4] = localWedge + nextNodeSliceOffset;
    }
}
void ablate::domain::descriptions::Axisymmetric::SetCoordinate(PetscInt node, PetscReal *coordinate) const {
    // determine where this node is, there is a special case for the 0 node shell or center
    PetscInt nodeShell, nodeSlice, nodeRotationIndex;
    if (node < numberCenterVertices) {
        nodeShell = 0;
        nodeSlice = node;
        nodeRotationIndex = 0;
    } else {
        nodeShell = ((node - numberCenterVertices) / numberVerticesPerShell) + 1;
        nodeSlice = (node - numberCenterVertices - (nodeShell - 1) * numberVerticesPerShell) / numberWedges;
        nodeRotationIndex = (node - numberCenterVertices) % numberWedges;
    }

    // Get the coordinate along the axis.
    axisDescription->SetCoordinate(nodeSlice, coordinate);

    // compute the maximum radius at this coordinate
    auto radius = radiusFunction->Eval(coordinate, 3, NAN);

    // if we are not at the center
    auto radiusFactor = ((PetscReal)nodeShell) / ((PetscReal)numberShells);
    coordinate[0] += radius * radiusFactor * PetscCosReal(2.0 * nodeRotationIndex * PETSC_PI / numberWedges);
    coordinate[1] += radius * radiusFactor * PetscSinReal(2.0 * nodeRotationIndex * PETSC_PI / numberWedges);
}

DMPolytopeType ablate::domain::descriptions::Axisymmetric::GetCellType(PetscInt cell) const {
    // flip the cell order so that hexes appear before tri prisms
    return CellReverser(cell) < numberTriPrismCells ? DM_POLYTOPE_TRI_PRISM : DM_POLYTOPE_HEXAHEDRON;
}

std::shared_ptr<ablate::domain::Region> ablate::domain::descriptions::Axisymmetric::GetRegion(const std::set<PetscInt> &face) const {
    // check to see if each node in on the surface
    bool onOuterShell = true;
    bool onLowerEndCap = true;
    bool onUpperEndCap = true;

    // compute the outer shell start
    auto outerShellStart = numberCenterVertices + numberVerticesPerShell * (numberShells - 1);
    // Coordinate used to track the z location
    auto *coordinate = new PetscReal[]{0.0, 0.0, 0.0};
    PetscReal Zavg = 0.0;
    for (const auto &node : face) {
        // check if we are on the outer shell
        if (node < outerShellStart) {
            onOuterShell = false;
        }
        // check if we are on the end caps
        PetscInt nodeSlice;
        if (node < numberCenterVertices) {
            nodeSlice = node;
        } else {
            auto nodeShell = ((node - numberCenterVertices) / numberVerticesPerShell) + 1;
            nodeSlice = (node - numberCenterVertices - (nodeShell - 1) * numberVerticesPerShell) / numberWedges;
        }

        if (nodeSlice != 0) {
            onLowerEndCap = false;
        }
        if (nodeSlice != numberSlices) {
            onUpperEndCap = false;
        }
        axisDescription->SetCoordinate(nodeSlice, coordinate);
        Zavg += coordinate[2];
    }
    // determine what region to return
    if (onOuterShell) {
        if (!boundaryFunction) {
            return shellBoundary;
        } else {
            Zavg /= face.size();
            coordinate[2] = Zavg;
            auto val = boundaryFunction->Eval(coordinate, 3, NAN);
            return std::make_shared<ablate::domain::Region>("outerShell", int(val));
        }
    } else if (onLowerEndCap) {
        return lowerCapBoundary;
    } else if (onUpperEndCap) {
        return upperCapBoundary;
    }

    return nullptr;
}

#include "registrar.hpp"
REGISTER(ablate::domain::descriptions::MeshDescription, ablate::domain::descriptions::Axisymmetric, "The Axisymmetric MeshDescription is used to create an axisymmetric mesh around the z axis",
         ARG(ablate::domain::descriptions::AxisDescription, "axis", "describes the nodes along the z axis"),
         ARG(ablate::mathFunctions::MathFunction, "radius", "a radius function that describes the radius as a function of z"), ARG(int, "numberWedges", "wedges/pie slices in the circle"),
         ARG(int, "numberShells", "slicing of the cylinder along the radius"),
         OPT(ablate::mathFunctions::MathFunction, "outerBoundaryFunction", "Function to tag different outer shell boundary regions"));
