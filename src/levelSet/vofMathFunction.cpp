#include "vofMathFunction.hpp"
#include "petscdmplex.h"
#include "petscfe.h"
#include <utility>
#include "levelSetUtilities.hpp"
#include "utilities/petscSupport.hpp"

ablate::levelSet::VOFMathFunction::VOFMathFunction(std::shared_ptr<ablate::domain::Domain> domain, std::shared_ptr<ablate::mathFunctions::MathFunction> levelSet)
    : FunctionPointer(VOFMathFunctionPetscFunction, this), domain(std::move(domain)), levelSet(std::move(levelSet)) {}
#include <signal.h>
PetscErrorCode ablate::levelSet::VOFMathFunction::VOFMathFunctionPetscFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBegin;

    auto      vofMathFunction = (VOFMathFunction *)ctx;
    DM        dm = vofMathFunction->domain->GetDM();
    PetscReal h;
    PetscInt  cell;


    // Make the tolerance half of the smallest distance between a cell-center and a face.
    PetscCall(DMPlexGetMinRadius(dm, &h));
    h *= 0.5;

    PetscCall(DMPlexFindCell(dm, x, h, &cell));

    if (PetscDefined(USE_DEBUG)) {
      PetscInt cStart, cEnd;
      PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
      PetscCheck(cell > -1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No cell was found.\n");
      PetscCheck((cell >= cStart) && (cell < cEnd), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "The DAG point found is not a cell.\n");
    }

    // call the support call to compute vof in the cell
    try {
        ablate::levelSet::Utilities::VOF(dm, cell, vofMathFunction->levelSet, u, nullptr, nullptr);
    } catch (std::exception &exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exp.what());
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::levelSet::VOFMathFunction, " Return the vertex level set values assuming a straight interface in the cell with a given normal vector.",
         ARG(ablate::domain::Domain, "domain", "domain to enable access to the cell information at a given point"),
         ARG(ablate::mathFunctions::MathFunction, "levelSet", "function used to calculate the level set values at the vertices"));
