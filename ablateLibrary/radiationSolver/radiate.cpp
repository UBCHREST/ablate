#include "radiate.hpp"
#include "utilities/mathUtilities.hpp"

ablate::radiationSolver::radiate::radiate() {}

void ablate::radiationSolver::radiate::Initialize(ablate::radiationSolver::RadiationSolver &bSolver) {
    ablate::radiationSolver::radiate::Initialize(bSolver);
    //bSolver.RegisterFunction(radiateFunction, this, fieldNames, fieldNames, {});
}

PetscErrorCode ablate::radiationSolver::radiate::radiateFunction() {
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::radiationSolver::RadiationProcess, ablate::radiationSolver::radiate, "Radiates");