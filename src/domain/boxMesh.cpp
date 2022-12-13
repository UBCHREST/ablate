#include "boxMesh.hpp"
#include <stdexcept>
#include <utilities/mpiError.hpp>
#include <utilities/petscOptions.hpp>
#include <utility>
#include "utilities/petscError.hpp"

ablate::domain::BoxMesh::BoxMesh(const std::string& name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers,
                                 std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary, bool simplex,
                                 std::shared_ptr<parameters::Parameters> options)
    : Domain(CreateBoxDM(name, std::move(faces), std::move(lower), std::move(upper), std::move(boundary), simplex), name, std::move(fieldDescriptors), std::move(modifiers), std::move(options)) {}

ablate::domain::BoxMesh::~BoxMesh() {
    if (dm) {
        DMDestroy(&dm);
    }
}

DM ablate::domain::BoxMesh::CreateBoxDM(const std::string& name, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary, bool simplex) {
    std::size_t dimensions = faces.size();
    if ((dimensions != lower.size()) || (dimensions != upper.size())) {
        throw std::runtime_error("BoxMesh Error: The faces, lower, and upper vectors must all be the same dimension.");
    }

    std::vector<DMBoundaryType> boundaryTypes(dimensions, DM_BOUNDARY_NONE);
    for (std::size_t d = 0; d < PetscMin(dimensions, boundary.size()); d++) {
        PetscBool found;
        PetscEnum index;
        PetscEnumFind(DMBoundaryTypes, boundary[d].c_str(), &index, &found) >> checkError;

        if (found) {
            boundaryTypes[d] = (DMBoundaryType)index;
        } else {
            throw std::invalid_argument("unable to find boundary type " + boundary[d]);
        }
    }

    // Make copy with PetscInt
    std::vector<PetscInt> facesPetsc(faces.begin(), faces.end());
    DM dm;
    DMPlexCreateBoxMesh(PETSC_COMM_WORLD, (PetscInt)dimensions, simplex ? PETSC_TRUE : PETSC_FALSE, &facesPetsc[0], &lower[0], &upper[0], &boundaryTypes[0], PETSC_TRUE, &dm) >> checkError;
    PetscObjectSetName((PetscObject)dm, name.c_str()) >> ablate::checkError;
    return dm;
}

#include "registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::BoxMesh,
         "Create a simple box mesh (1,2,3) dimension When used with the dm_plex_separate_marker each boundary \"marker\" or \"Face Sets\" as:\n"
         "\n### 1D\n\n"
         "| Direction | Description | Value |\n"
         "| --- | --- | --- |\n"
         "|  x- | left  |1|\n"
         "|  x+ | right |2|\n"
         "\n### 2D:\n\n"
         "| Direction | Description | Value |\n"
         "| --- | --- |-----|\n"
         "|y+  | top    | 3   |\n"
         "|y-  | bottom | 1   |\n"
         "|x+  | right  | 2   |\n"
         "|x-  | left   | 4   |\n"
         "\n### 3D:\n\n"
         "| Direction | Description | Value |\n"
         "| --- | --- |-----|\n"
         " | z- | bottom | 1 |\n"
         " | z+ | top    | 2|\n"
         " | y+ | front  | 3|\n"
         " | y- | back   | 4|\n"
         " | x+ | right  | 5|\n"
         " | x- | left   | 6|\n",
         ARG(std::string, "name", "the name of the domain/mesh object"), OPT(std::vector<ablate::domain::FieldDescriptor>, "fields", "a list of fields/field descriptors"),
         OPT(std::vector<ablate::domain::modifiers::Modifier>, "modifiers", "a list of domain modifier"), ARG(std::vector<int>, "faces", "the number of faces in each direction"),
         ARG(std::vector<double>, "lower", "the lower bound of the mesh"), ARG(std::vector<double>, "upper", "the upper bound of the mesh"),
         OPT(std::vector<std::string>, "boundary", "custom boundary types (NONE, GHOSTED, MIRROR, PERIODIC)"), OPT(bool, "simplex", "sets if the elements/cells are simplex"),
         OPT(ablate::parameters::Parameters, "options", "PETSc options specific to this dm.  Default value allows the dm to access global options."));
