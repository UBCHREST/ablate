#include "vofMathFunction.hpp"

#include <utility>
#include "levelSetUtilities.hpp"
#include "utilities/petscSupport.hpp"

ablate::levelSet::VOFMathFunction::VOFMathFunction(std::shared_ptr<ablate::domain::Domain> domain, std::shared_ptr<ablate::mathFunctions::MathFunction> levelSet)
    : FunctionPointer(VOFMathFunctionPetscFunction, this), domain(std::move(domain)), levelSet(std::move(levelSet)) {}

PetscErrorCode ablate::levelSet::VOFMathFunction::VOFMathFunctionPetscFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBegin;
    auto vofMathFunction = (VOFMathFunction *)ctx;

    // Get the dm from the domain
    DM dm = vofMathFunction->domain->GetDM();

    // Determine the cell/element where this xyz resides
    PetscInt cell;
    PetscCall(DMPlexGetContainingCell(dm, x, &cell));

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