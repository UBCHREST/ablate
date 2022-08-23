#include "volumeRadiation.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::radiation::VolumeRadiation::VolumeRadiation(const std::string& solverId1, const std::shared_ptr<domain::Region>& region, std::shared_ptr<domain::Region> fieldBoundary,
                                                    const PetscInt raynumber, const PetscInt intervalIn, const std::shared_ptr<parameters::Parameters>& options,
                                                    std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : Radiation(solverId1, region, fieldBoundary, raynumber, intervalIn, radiationModelIn, log), CellSolver(solverId1, region, options) {}
ablate::radiation::VolumeRadiation::~VolumeRadiation() {}

void ablate::radiation::VolumeRadiation::Setup() {
    ablate::solver::CellSolver::Setup();
    ablate::radiation::Radiation::Setup();
}

void ablate::radiation::VolumeRadiation::Register(std::shared_ptr<ablate::domain::SubDomain> subDomain) {
    ablate::solver::Solver::Register(subDomain);
    ablate::radiation::Radiation::Register(subDomain);
}

void ablate::radiation::VolumeRadiation::Initialize() {
    solver::Range cellRange;
    GetCellRange(cellRange);  //!< Gets the cell range that should be applied to the radiation solver

    ablate::radiation::Radiation::Initialize(cellRange);  //!< Get the range of cells that the solver occupies in order for the radiation solver to give energy to the finite volume

    RestoreRange(cellRange);
}

PetscErrorCode ablate::radiation::VolumeRadiation::ComputeRHSFunction(PetscReal time, Vec solVec, Vec rhs) {
    PetscFunctionBegin;

    /** Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable. */
    const PetscScalar* rhsArray;
    VecGetArrayRead(rhs, &rhsArray);
    const auto& eulerFieldInfo = Radiation::subDomain->GetField("euler");

    origin = ablate::radiation::Radiation::Solve(solVec);

    solver::Range cellRange;
    GetCellRange(cellRange);  //!< Gets the cell range to iterate over when retrieving cell indexes from the solver

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        PetscScalar* rhsValues;
        DMPlexPointLocalFieldRead(Radiation::subDomain->GetDM(), iCell, eulerFieldInfo.id, rhsArray, &rhsValues);
        rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += GetIntensity(iCell);  //!< Loop through the cells and update the equation of state
    }
    RestoreRange(cellRange);
    VecRestoreArrayRead(rhs, &rhsArray);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::radiation::VolumeRadiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(ablate::domain::Region, "fieldBoundary", "boundary of the radiation region"),
         ARG(int, "rays", "number of rays used by the solver"), ARG(int, "interval", "number of time steps between the radiation solves"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));