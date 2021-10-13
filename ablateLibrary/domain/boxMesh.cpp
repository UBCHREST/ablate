#include "boxMesh.hpp"
#include <stdexcept>
#include <utilities/mpiError.hpp>
#include <utilities/petscOptions.hpp>
#include "utilities/petscError.hpp"

ablate::domain::BoxMesh::BoxMesh(std::string name, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary, bool simplex,
                               std::shared_ptr<parameters::Parameters> options)
    : Domain(name), petscOptions(NULL) {
    // Set the options
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }

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
    DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, simplex ? PETSC_TRUE : PETSC_FALSE, &facesPetsc[0], &lower[0], &upper[0], &boundaryTypes[0], PETSC_TRUE, &dm) >> checkError;
    PetscObjectSetOptions((PetscObject)dm, petscOptions) >> checkError;
    DMSetFromOptions(dm) >> checkError;
}
ablate::domain::BoxMesh::~BoxMesh() {
    if (dm) {
        DMDestroy(&dm);
    }
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(name, &petscOptions);
    }
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::BoxMesh, "simple uniform box mesh", ARG(std::string, "name", "the name of the domain/mesh object"),
         ARG(std::vector<int>, "faces", "the number of faces in each direction"), ARG(std::vector<double>, "lower", "the lower bound of the mesh"),
         ARG(std::vector<double>, "upper", "the upper bound of the mesh"), OPT(std::vector<std::string>, "boundary", "custom boundary types (NONE, GHOSTED, MIRROR, PERIODIC)"),
         OPT(bool, "simplex", "sets if the elements/cells are simplex"), OPT(ablate::parameters::Parameters, "options", "any PETSc options"));