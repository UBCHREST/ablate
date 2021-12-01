#include "boxMeshBoundaryCells.hpp"
#include <domain/modifiers/createLabel.hpp>
#include <domain/modifiers/distributeWithGhostCells.hpp>
#include <domain/modifiers/ghostBoundaryCells.hpp>
#include <domain/modifiers/mergeLabels.hpp>
#include <domain/modifiers/tagLabelBoundary.hpp>
#include <mathFunctions/geom/box.hpp>
#include <stdexcept>
#include <utilities/mpiError.hpp>
#include "utilities/petscError.hpp"

ablate::domain::BoxMeshBoundaryCells::BoxMeshBoundaryCells(std::string name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors,
                                                           std::vector<std::shared_ptr<modifiers::Modifier>> modifiers, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper,
                                                           std::shared_ptr<domain::Region> mainRegion, std::shared_ptr<domain::Region> boundaryFaceRegion,
                                                           std::shared_ptr<domain::Region> boundaryCellRegion, bool simplex)
    : Domain(CreateBoxDM(name, faces, lower, upper, simplex), name, fieldDescriptors, AddBoundaryModifiers(lower, upper, mainRegion, boundaryFaceRegion, boundaryCellRegion, modifiers)) {}

ablate::domain::BoxMeshBoundaryCells::~BoxMeshBoundaryCells() {
    if (dm) {
        DMDestroy(&dm);
    }
}

DM ablate::domain::BoxMeshBoundaryCells::CreateBoxDM(std::string name, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, bool simplex) {
    std::size_t dimensions = faces.size();
    if ((dimensions != lower.size()) || (dimensions != upper.size())) {
        throw std::runtime_error("BoxMesh Error: The faces, lower, and upper vectors must all be the same dimension.");
    }

    // compute dx in each direction
    std::vector<double> dx(dimensions);
    for (std::size_t i = 0; i < dimensions; i++) {
        dx[i] = (upper[i] - lower[i]) / faces[i];
    }

    // Add two faces for each ghost cell
    for (auto& face : faces) {
        face += 2;
    }

    // Move in/out the lower upper dimension
    for (std::size_t i = 0; i < dimensions; i++) {
        lower[i] -= dx[i];
        upper[i] += dx[i];
    }

    // Make copy with PetscInt
    std::vector<PetscInt> facesPetsc(faces.begin(), faces.end());
    DM dm;
    DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, simplex ? PETSC_TRUE : PETSC_FALSE, &facesPetsc[0], &lower[0], &upper[0], NULL, PETSC_TRUE, &dm) >> checkError;
    return dm;
}
std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>> ablate::domain::BoxMeshBoundaryCells::AddBoundaryModifiers(std::vector<double> lower, std::vector<double> upper,
                                                                                                                             std::shared_ptr<domain::Region> mainRegion,
                                                                                                                             std::shared_ptr<domain::Region> boundaryFaceRegion,
                                                                                                                             std::shared_ptr<domain::Region> boundaryCellRegion,
                                                                                                                             std::vector<std::shared_ptr<modifiers::Modifier>> modifiers) {
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::GhostBoundaryCells>());
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>());
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(mainRegion, std::make_shared<ablate::mathFunctions::geom::Box>(lower, upper)));
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::TagLabelBoundary>(mainRegion, boundaryFaceRegion, boundaryCellRegion));

    return modifiers;
}

#include "registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::BoxMeshBoundaryCells, "simple uniform box mesh with boundary solver cells",
         ARG(std::string, "name", "the name of the domain/mesh object"),
         OPT(std::vector<ablate::domain::FieldDescriptor>, "fields", "a list of fields/field descriptors"),
         OPT(std::vector<ablate::domain::modifiers::Modifier>, "modifiers", "a list of domain modifier"),
         ARG(std::vector<int>, "faces", "the number of faces in each direction"),
         ARG(std::vector<double>, "lower", "the lower bound of the mesh"),
         ARG(std::vector<double>, "upper", "the upper bound of the mesh"),
         ARG(ablate::domain::Region, "mainRegion", "the label for the main region (no ghost cells)"),
         ARG(ablate::domain::Region, "boundaryFaceRegion", "the label for the new face cells between regions"),
         ARG(ablate::domain::Region, "boundaryCellRegion", "the label for the boundary cells)"),
         OPT(bool, "simplex", "sets if the elements/cells are simplex"));