#include "../../ablateLibrary/radiationSolver/radiate.hpp"
#include "utilities/mathUtilities.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/speciesTransport.hpp"
#include "finiteVolume/processes/evTransport.hpp"

ablate::radiationSolver::radiate::radiate() {}

void ablate::radiationSolver::radiate::Initialize(ablate::radiationSolver::RadiationSolver &bSolver) {
    ablate::radiationSolver::radiate::Initialize(bSolver);
    bSolver.RegisterAuxFieldUpdate(UpdateAuxTemperatureField, &updateTemperatureData,
                             finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                             {finiteVolume::CompressibleFlowFields::EULER_FIELD});
}

PetscErrorCode ablate::radiationSolver::radiate::UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[],
                                                                           const PetscScalar* conservedValues, PetscScalar* auxField, void* ctx) {
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::radiationSolver::RadiationProcess, ablate::radiationSolver::radiate, "Radiates");