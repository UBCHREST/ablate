#include "hdf5Initializer.hpp"
#include <petscviewerhdf5.h>
#include "domain/domain.hpp"
#include "utilities/petscUtilities.hpp"

ablate::domain::Hdf5Initializer::Hdf5Initializer(std::filesystem::path hdf5Path, std::shared_ptr<ablate::domain::Region> region) : hdf5Path(std::move(hdf5Path)), region(std::move(region)) {}

std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> ablate::domain::Hdf5Initializer::GetFieldFunctions(const std::vector<domain::Field>& fields) const {
    // Create a mesh that the field functions will share
    auto baseMesh = std::make_shared<Hdf5Mesh>(hdf5Path);

    std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> functions;

    // March over each requested field
    for (const auto& field : fields) {
        // Create the math function
        auto mathFunction = std::make_shared<Hdf5MathFunction>(baseMesh, ablate::domain::Domain::solution_vector_name + "_" + field.name);

        // Create a fieldFunction wrapper
        auto fieldFunction = std::make_shared<ablate::mathFunctions::FieldFunction>(field.name, mathFunction, nullptr, region);

        functions.push_back(fieldFunction);
    }
    return functions;
}

ablate::domain::Hdf5Initializer::Hdf5Mesh::Hdf5Mesh(const std::filesystem::path& hdf5Path) {
    // Create a viewer for the hdf5 file
    PetscViewerHDF5Open(PETSC_COMM_SELF, hdf5Path.c_str(), FILE_MODE_READ, &petscViewer) >> utilities::PetscUtilities::checkError;

    // create an dm that lives only on this rank
    DMCreate(PETSC_COMM_SELF, &dm) >> utilities::PetscUtilities::checkError;
    DMSetType(dm, DMPLEX) >> utilities::PetscUtilities::checkError;
    DMLoad(dm, petscViewer) >> utilities::PetscUtilities::checkError;

    // set the time stepping, so it is able to load time based history
    PetscViewerHDF5PushTimestepping(petscViewer) >> utilities::PetscUtilities::checkError;

    // extract the connected cell count and store them
    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
    numberCells = cEnd - cStart;
}
ablate::domain::Hdf5Initializer::Hdf5Mesh::~Hdf5Mesh() {
    // close the viewer
    PetscViewerDestroy(&petscViewer) >> utilities::PetscUtilities::checkError;

    // free the memory with the mesh
    DMDestroy(&dm) >> utilities::PetscUtilities::checkError;
}

ablate::domain::Hdf5Initializer::Hdf5MathFunction::Hdf5MathFunction(std::shared_ptr<Hdf5Mesh> baseMeshIn, std::string fieldIn) : field(std::move(fieldIn)), baseMesh(std::move(baseMeshIn)) {
    // Clone the dm for this field dm
    DMClone(baseMesh->dm, &fieldDm) >> utilities::PetscUtilities::checkError;

    // Load in the cells
    VecCreate(PETSC_COMM_WORLD, &fieldVec) >> utilities::PetscUtilities::checkError;
    auto fieldName = "/cell_fields/" + field;
    PetscObjectSetName((PetscObject)fieldVec, fieldName.c_str()) >> utilities::PetscUtilities::checkError;
    VecLoad(fieldVec, baseMesh->petscViewer) >> utilities::PetscUtilities::checkError;

    // Do some sanity checks
    PetscInt vecSize;
    VecGetSize(fieldVec, &vecSize) >> utilities::PetscUtilities::checkError;
    VecGetBlockSize(fieldVec, &components) >> utilities::PetscUtilities::checkError;

    // compute the number of points that should be in the vector
    PetscInt vecPoints = vecSize / components;
    if (vecPoints != baseMesh->numberCells) {
        throw std::invalid_argument("The number of points in the array " + field + " (" + std::to_string(vecPoints) + ") does not match the cells in the domain (" +
                                    std::to_string(baseMesh->numberCells) + ")");
    }

    // setup the dm to use this field
    DMGetDimension(fieldDm, &dim) >> utilities::PetscUtilities::checkError;
    PetscBool simplex;
    DMPlexIsSimplex(fieldDm, &simplex) >> utilities::PetscUtilities::checkError;
    PetscFE fe;
    PetscFECreateLagrange(PETSC_COMM_SELF, dim, components, simplex, 0, PETSC_DETERMINE, &fe) >> utilities::PetscUtilities::checkError;
    DMSetField(fieldDm, 0, nullptr, (PetscObject)fe) >> utilities::PetscUtilities::checkError;
    PetscFEDestroy(&fe);
    DMCreateDS(fieldDm);
}

ablate::domain::Hdf5Initializer::Hdf5MathFunction::~Hdf5MathFunction() {
    DMDestroy(&fieldDm) >> utilities::PetscUtilities::checkError;
    VecDestroy(&fieldVec) >> utilities::PetscUtilities::checkError;
}

PetscErrorCode ablate::domain::Hdf5Initializer::Hdf5MathFunction::Eval(PetscInt xyzDim, const PetscReal xyz[], PetscScalar* u) const {
    PetscFunctionBeginUser;

    // Create an interpolant
    DMInterpolationInfo interpolant;
    PetscCall(DMInterpolationCreate(PETSC_COMM_SELF, &interpolant));
    PetscCall(DMInterpolationSetDim(interpolant, dim));
    PetscCall(DMInterpolationSetDof(interpolant, components));

    // Create a point for interpolant
    PetscReal pt[3] = {0.0, 0.0, 0.0};

    // Copy over the values
    PetscArraycpy(pt, xyz, PetscMin(xyzDim, dim));
    PetscCall(DMInterpolationAddPoints(interpolant, 1, pt));

    // Setup the interpolant
    PetscCall(DMInterpolationSetUp(interpolant, fieldDm, PETSC_FALSE, PETSC_FALSE));

    // Create a vec to hold the information
    Vec fieldAtPoint;
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, components, components, u, &fieldAtPoint));

    // interpolate
    PetscCall(DMInterpolationEvaluate(interpolant, fieldDm, fieldVec, fieldAtPoint));

    // cleanup
    PetscCall(VecDestroy(&fieldAtPoint));
    PetscCall(DMInterpolationDestroy(&interpolant));

    PetscFunctionReturn(0);
}
double ablate::domain::Hdf5Initializer::Hdf5MathFunction::Eval(const double& x, const double& y, const double& z, const double& t) const {
    PetscFunctionBeginUser;
    PetscReal xyz[3] = {x, y, z};
    PetscScalar result[components];

    Eval(3, xyz, result) >> utilities::PetscUtilities::checkError;
    return result[0];
    PetscFunctionReturn(0);
}
double ablate::domain::Hdf5Initializer::Hdf5MathFunction::Eval(const double* xyz, const int& ndims, const double& t) const {
    PetscFunctionBeginUser;
    PetscScalar result[components];

    Eval(ndims, xyz, result) >> utilities::PetscUtilities::checkError;
    return result[0];
    PetscFunctionReturn(0);
}
void ablate::domain::Hdf5Initializer::Hdf5MathFunction::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    PetscFunctionBeginUser;
    PetscReal xyz[3] = {x, y, z};
    result.resize(components);

    Eval(3, xyz, result.data()) >> utilities::PetscUtilities::checkError;
    PetscFunctionReturnVoid();
}
void ablate::domain::Hdf5Initializer::Hdf5MathFunction::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    PetscFunctionBeginUser;
    result.resize(components);

    Eval(ndims, xyz, result.data()) >> utilities::PetscUtilities::checkError;
    PetscFunctionReturnVoid();
}

PetscErrorCode ablate::domain::Hdf5Initializer::Hdf5MathFunction::Hdf5PetscFunction(PetscInt dim, PetscReal time, const PetscReal* xyz, PetscInt nf, PetscScalar* u, void* ctx) {
    PetscFunctionBeginUser;
    auto hdf5MathFunction = (Hdf5MathFunction*)ctx;

    // Make sure that the result can hold the value (the hdf5 result may be smaller)
    if (nf < hdf5MathFunction->components) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "The PetscFunction in ablate::domain::Hdf5Initializer requires a size of %" PetscInt_FMT, hdf5MathFunction->components);
    }

    // Interpolate at this point
    PetscCall(hdf5MathFunction->Eval(dim, xyz, u));

    // Set any other values to zero
    for (PetscInt i = hdf5MathFunction->components; i < nf; ++i) {
        u[i] = 0;
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::domain::Initializer, ablate::domain::Hdf5Initializer, "Initialization a simulation based upon a previous result.  It does not need to be the same mesh.",
         ARG(std::filesystem::path, "path", "path to hdf5 file"));
