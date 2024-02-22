#include "meshMapper.hpp"
#include <petsc/private/dmpleximpl.h>

#include <utility>
#include "utilities/petscUtilities.hpp"

ablate::domain::modifiers::MeshMapper::MeshMapper(std::shared_ptr<ablate::mathFunctions::MathFunction> mappingFunction) : mappingFunction(std::move(mappingFunction)) {}

void ablate::domain::modifiers::MeshMapper::Modify(DM& dm) {
    // check for a coordinate space
    DM coordinateDm;
    DMGetCoordinateDM(dm, &coordinateDm) >> utilities::PetscUtilities::checkError;
    PetscInt coordinateDim;
    DMGetCoordinateDim(dm, &coordinateDim) >> utilities::PetscUtilities::checkError;
    PetscDS coordinateDs;
    DMGetDS(coordinateDm, &coordinateDs);
    PetscObject discretization;
    PetscDSGetDiscretization(coordinateDs, 0, &discretization) >> utilities::PetscUtilities::checkError;
    PetscClassId discretizationId;
    PetscObjectGetClassId(discretization, &discretizationId);

    // Get the petsc function and ctx from the math function
    auto petscFunction = mappingFunction->GetPetscFunction();
    auto petscCtx = mappingFunction->GetContext();

    if (discretizationId != PETSCFE_CLASSID) {
        // direct Modification
        Vec localCoordsVector;
        PetscSection coordsSection;
        PetscScalar* coordsArray;

        // get the vertex information
        PetscInt vStart, vEnd;
        DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd) >> utilities::PetscUtilities::checkError;
        DMGetCoordinateSection(dm, &coordsSection) >> utilities::PetscUtilities::checkError;
        DMGetCoordinatesLocal(dm, &localCoordsVector) >> utilities::PetscUtilities::checkError;
        VecGetArray(localCoordsVector, &coordsArray) >> utilities::PetscUtilities::checkError;

        // store the initial copy of xyz
        PetscReal xyz[3];

        for (PetscInt v = vStart; v < vEnd; ++v) {
            PetscInt off;
            PetscSectionGetOffset(coordsSection, v, &off);

            // extract the initial x, y, z values
            for (PetscInt d = 0; d < coordinateDim; ++d) {
                xyz[d] = coordsArray[off + d];
            }

            // call the mapping function
            petscFunction(coordinateDim, 0.0, xyz, coordinateDim, coordsArray + off, petscCtx) >> utilities::PetscUtilities::checkError;
        }
        VecRestoreArray(localCoordsVector, &coordsArray) >> utilities::PetscUtilities::checkError;
        DMSetCoordinatesLocal(dm, localCoordsVector) >> utilities::PetscUtilities::checkError;
    } else {
        // projection into the coordinate space
        // this is a duplication of PETSc DMPlexRemapGeometry that uses DMProjectFunctionLocal instead of DMProjectFieldLocal
        DM cdm;
        DMField cf;
        Vec lCoords, tmpCoords;

        DMGetCoordinateDM(dm, &cdm);
        DMGetCoordinatesLocal(dm, &lCoords);
        DMGetLocalVector(cdm, &tmpCoords);
        VecCopy(lCoords, tmpCoords);
        /* We have to do the coordinate field manually right now since the coordinate DM will not have its own */
        DMGetCoordinateField(dm, &cf);
        cdm->coordinates[0].field = cf;

        // use a DMProjectFunctionLocal instead of DMProjectFieldLocal
        PetscInt numberFields;
        DMGetNumFields(cdm, &numberFields) >> utilities::PetscUtilities::checkError;
        std::vector<mathFunctions::PetscFunction> fieldFunctionsPts(numberFields, nullptr);
        std::vector<void*> fieldContexts(numberFields, nullptr);
        fieldFunctionsPts[0] = petscFunction;
        fieldContexts[0] = petscCtx;
        DMProjectFunctionLocal(cdm, 0.0, fieldFunctionsPts.data(), fieldContexts.data(), INSERT_VALUES, lCoords) >> utilities::PetscUtilities::checkError;

        cdm->coordinates[0].field = nullptr;
        DMRestoreLocalVector(cdm, &tmpCoords);
        DMSetCoordinatesLocal(dm, lCoords);
    }
}

void ablate::domain::modifiers::MeshMapper::Modify(const std::vector<double>& in, std::vector<double>& out) const {
    out.resize(in.size());
    mappingFunction->Eval(in.data(), (int)in.size(), 0.0, out);
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::MeshMapper, "Maps the x,y,z coordinate of the domain mesh by the given function.",
                      ablate::mathFunctions::MathFunction);
