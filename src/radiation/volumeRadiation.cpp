#include "volumeRadiation.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::radiation::Radiation::Radiation(std::string solverId, const std::shared_ptr<domain::Region>& region, std::shared_ptr<domain::Region> fieldBoundary, const PetscInt raynumber,
                                        std::shared_ptr<parameters::Parameters> options, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
                                        std::shared_ptr<ablate::monitors::logs::Log> log)
    : solverId(std::move(solverId)),
      region((std::shared_ptr<domain::Region> &&) std::move(region)),
      options(std::move(options)),
      radiationModel(std::move(radiationModelIn)),
      fieldBoundary(std::move(fieldBoundary)),
      log(std::move(log)) {
    nTheta = raynumber;    //!< The number of angles to solve with, given by user input
    nPhi = 2 * raynumber;  //!< The number of angles to solve with, given by user input
}

ablate::radiation::Radiation::~Radiation() {
    if (radsolve) DMDestroy(&radsolve) >> checkError;  //!< Destroy the radiation particle swarm
    if (cellGeomVec) VecDestroy(&cellGeomVec) >> checkError;
    if (faceGeomVec) VecDestroy(&faceGeomVec) >> checkError;
}

void ablate::radiation::VolumeRadiation::Initialize() {
    solver::Range cellRange;
    GetCellRange(cellRange);
    ablate::radiation::Radiation::Initialize(cellRange);
    RestoreRange(cellRange);
}

PetscErrorCode ablate::radiation::VolumeRadiation::ComputeRHSFunction(PetscReal time, Vec solVec, Vec rhs) {
    PetscFunctionBegin;

    /** Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable. */
    const PetscScalar* rhsArray;
    VecGetArrayRead(rhs, &rhsArray);

    ablate::radiation::Radiation::Solve(solVec, rhs);

    solver::Range cellRange;
    GetCellRange(cellRange);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        PetscScalar* rhsValues;
        rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += origin[iCell].intensity;  // GetIntensity(iCell);  //!< Loop through the cells and update the equation of state
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
    REGISTER(ablate::solver::Solver, ablate::radiation::Radiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
             ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(ablate::domain::Region, "fieldBoundary", "boundary of the radiation region"),
             ARG(int, "rays", "number of rays used by the solver"), OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
             ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));