#include "boxMeshBoundaryCells.hpp"
#include <domain/modifiers/createLabel.hpp>
#include <domain/modifiers/distributeWithGhostCells.hpp>
#include <domain/modifiers/mergeLabels.hpp>
#include <domain/modifiers/tagLabelBoundary.hpp>
#include <mathFunctions/geom/box.hpp>
#include <stdexcept>
#include <utilities/mpiError.hpp>
#include <utility>
#include "utilities/petscError.hpp"

ablate::domain::BoxMeshBoundaryCells::BoxMeshBoundaryCells(const std::string& name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors,
                                                           std::vector<std::shared_ptr<modifiers::Modifier>> preModifiers, std::vector<std::shared_ptr<modifiers::Modifier>> postModifiers,
                                                           std::vector<int> faces, const std::vector<double>& lower, const std::vector<double>& upper, bool simplex)
    : Domain(CreateBoxDM(name, std::move(faces), lower, upper, simplex), name, std::move(fieldDescriptors), AddBoundaryModifiers(lower, upper, std::move(preModifiers), std::move(postModifiers))) {}

ablate::domain::BoxMeshBoundaryCells::~BoxMeshBoundaryCells() {
    if (dm) {
        DMDestroy(&dm);
    }
}

DM ablate::domain::BoxMeshBoundaryCells::CreateBoxDM(const std::string& name, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, bool simplex) {
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
    DMPlexCreateBoxMesh(PETSC_COMM_WORLD, (PetscInt)dimensions, simplex ? PETSC_TRUE : PETSC_FALSE, &facesPetsc[0], &lower[0], &upper[0], nullptr, PETSC_TRUE, &dm) >> checkError;
    return dm;
}
std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>> ablate::domain::BoxMeshBoundaryCells::AddBoundaryModifiers(std::vector<double> lower, std::vector<double> upper,
                                                                                                                             std::vector<std::shared_ptr<modifiers::Modifier>> preModifiers,
                                                                                                                             std::vector<std::shared_ptr<modifiers::Modifier>> postModifiers) {
    auto modifiers = preModifiers;
    auto interiorLabel = std::make_shared<domain::Region>(interiorCellsLabel);
    auto boundaryFaceRegion = std::make_shared<domain::Region>(boundaryFacesLabel);
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(interiorLabel, std::make_shared<ablate::mathFunctions::geom::Box>(lower, upper)));

    // define a boundaryCellRegion
    auto boundaryCellRegion = std::make_shared<domain::Region>(boundaryCellsLabel);
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::TagLabelBoundary>(interiorLabel, boundaryFaceRegion, boundaryCellRegion));

    // define the ghost cells plus interior (leaves out corners)
    auto entireDomainRegion = std::make_shared<domain::Region>(entireDomainLabel);
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::MergeLabels>(entireDomainRegion, std::vector<std::shared_ptr<domain::Region>>{interiorLabel, boundaryCellRegion}));

    const int X = 0;
    const int Y = 1;
    const int Z = 2;
    const double min = std::numeric_limits<double>::lowest();
    const double max = std::numeric_limits<double>::max();

    // Define a subset for the other boundary regions
    switch (lower.size()) {
        case 3:
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsFront),
                std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X], lower[Y], upper[Z]}, std::vector<double>{upper[X], upper[Y], max})));
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsBack),
                std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X], lower[Y], min}, std::vector<double>{upper[X], upper[Y], lower[Z]})));
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsTop),
                std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X], upper[Y], lower[Z]}, std::vector<double>{upper[X], max, upper[Z]})));
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsBottom),
                std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X], min, lower[Z]}, std::vector<double>{upper[X], lower[Y], upper[Z]})));
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsRight),
                std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{upper[X], lower[Y], lower[Z]}, std::vector<double>{max, upper[Y], upper[Z]})));
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsLeft),
                std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{min, lower[Y], lower[Z]}, std::vector<double>{lower[X], upper[Y], upper[Z]})));
            break;
        case 2:
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsTop), std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X], upper[Y]}, std::vector<double>{upper[X], max})));
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsBottom),
                std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X], min}, std::vector<double>{upper[X], lower[Y]})));
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsRight), std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{upper[X], lower[Y]}, std::vector<double>{max, upper[Y]})));
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                std::make_shared<domain::Region>(boundaryCellsLeft), std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{min, lower[Y]}, std::vector<double>{lower[X], upper[Y]})));
            break;
        case 1:
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(std::make_shared<domain::Region>(boundaryCellsRight),
                                                                                         std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{upper[X]}, std::vector<double>{max})));
            modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(std::make_shared<domain::Region>(boundaryCellsLeft),
                                                                                         std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{min}, std::vector<double>{lower[X]})));
    }

    modifiers.insert(modifiers.end(), postModifiers.begin(), postModifiers.end());

    return modifiers;
}

#include "registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::BoxMeshBoundaryCells,
         "simple uniform box mesh with boundary solver cells.  Available labels are: interiorCells, domain (interior and boundary cells), boundaryFaces, boundaryCells, boundaryCellsLeft, "
         "boundaryCellsRight, "
         "boundaryCellsBottom, boundaryCellsTop, boundaryCellsFront, and boundaryCellsBack",
         ARG(std::string, "name", "the name of the domain/mesh object"), OPT(std::vector<ablate::domain::FieldDescriptor>, "fields", "a list of fields/field descriptors"),
         OPT(std::vector<ablate::domain::modifiers::Modifier>, "preModifiers", "a list of domain modifiers to apply before ghost labeling"),
         OPT(std::vector<ablate::domain::modifiers::Modifier>, "postModifiers", "a list of domain modifiers to apply after ghost labeling"),
         ARG(std::vector<int>, "faces", "the number of faces in each direction"), ARG(std::vector<double>, "lower", "the lower bound of the mesh"),
         ARG(std::vector<double>, "upper", "the upper bound of the mesh"), OPT(bool, "simplex", "sets if the elements/cells are simplex"));