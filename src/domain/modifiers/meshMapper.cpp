#include "meshMapper.hpp"
#include "utilities/petscError.hpp"
ablate::domain::modifiers::MeshMapper::MeshMapper(std::shared_ptr<ablate::mathFunctions::MathFunction> mappingFunction) : mappingFunction(mappingFunction) {}

void ablate::domain::modifiers::MeshMapper::Modify(DM& dm) {
//    // get the coordinate local vector from the dm
//    Vec coordinateLocalVec;
//    DMGetCoordinatesLocal(dm, &coordinateLocalVec) >> checkError;
//
//    // get the number of dimensions in the coordinate
//    PetscInt cdim;
//    DMGetCoordinateDim(dm, &cdim) >> checkError;
//
//    // determine the number of points in the coordinate vector
//    PetscInt np;
//    VecGetLocalSize(coordinateLocalVec, &np);
//    np /= cdim;
//
//    // store the initial copy of xyz
//    PetscReal xyz[3];
//
//    // Get the petsc function and ctx from the math function
//    auto petscFunction = mappingFunction->GetPetscFunction();
//    auto petscCtx = mappingFunction->GetContext();
//
//    // Get access to the array for writing
//    PetscScalar* coords;
//    VecGetArrayWrite(coordinateLocalVec, &coords);
//    for (PetscInt p = 0; p < np; ++p) {
//        // extract the initial x, y, z values
//        for (PetscInt d = 0; d < cdim; ++d) {
//            xyz[d] = coords[p * cdim + d];
//        }
//
//        // call the mapping function
//        petscFunction(cdim, 0.0, xyz, cdim, coords + (p * cdim), petscCtx) >> checkError;
//    }
//
//    VecRestoreArrayWrite(coordinateLocalVec, &coords) >> checkError;
//    DMSetCoordinatesLocal(dm, coordinateLocalVec) >> checkError;
//    DMSetCoordinates(dm, NULL) >> checkError;
//
//    // Force rebuilding
//    Vec coordinateGlobalVec;
//    DMGetCoordinates(dm, &coordinateGlobalVec) >> checkError;
}
void ablate::domain::modifiers::MeshMapper::Modify(const std::vector<double>& in, std::vector<double>& out) const {
    out.resize(in.size());
    mappingFunction->Eval(in.data(), (int)in.size(), 0.0, out);
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::MeshMapper, "Maps the x,y,z coordinate of the domain mesh by the given function.",
                      ablate::mathFunctions::MathFunction);
