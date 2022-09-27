#include "volumeRadiation.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "io/interval/fixedInterval.hpp"

ablate::radiation::VolumeRadiation::VolumeRadiation(const std::string& solverId1, const std::shared_ptr<domain::Region>& region, std::shared_ptr<domain::Region> fieldBoundary,
                                                    const PetscInt raynumber, std::shared_ptr<io::interval::Interval> intervalIn, const std::shared_ptr<parameters::Parameters>& options,
                                                    std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : Radiation(solverId1, region, fieldBoundary, raynumber, radiationModelIn, log),
      CellSolver(solverId1, region, options),
      interval((intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>())) {}
ablate::radiation::VolumeRadiation::~VolumeRadiation() {}

void ablate::radiation::VolumeRadiation::Setup() {
    solver::Range cellRange;
    GetCellRange(cellRange);  //!< Gets the cell range that should be applied to the radiation solver

    ablate::solver::CellSolver::Setup();
    ablate::radiation::Radiation::Setup(cellRange, GetSubDomain(), false);  //!< Insert the cell range of the solver here
    auto radiationPreStep = [this](auto&& PH1, auto&& PH2) { RadiationPreStep(std::forward<decltype(PH1)>(PH1)); };
    RegisterPreStep(radiationPreStep);
    RestoreRange(cellRange);
}

void ablate::radiation::VolumeRadiation::Register(std::shared_ptr<ablate::domain::SubDomain> subDomain) { ablate::solver::Solver::Register(subDomain); }

void ablate::radiation::VolumeRadiation::Initialize() {
    solver::Range cellRange;
    GetCellRange(cellRange);  //!< Gets the cell range that should be applied to the radiation solver

    ablate::radiation::Radiation::Initialize(cellRange, GetSubDomain());  //!< Get the range of cells that the solver occupies in order for the radiation solver to give energy to the finite volume

    RestoreRange(cellRange);
}

PetscErrorCode ablate::radiation::VolumeRadiation::RadiationPreStep(TS ts) {
    PetscFunctionBegin;

    /** Only update the radiation solution if the sufficient interval has passed */
    PetscInt step;
    PetscReal time;
    TSGetStepNumber(ts, &step) >> checkError;
    TSGetTime(ts, &time) >> checkError;
    if (interval->Check(PetscObjectComm((PetscObject)ts), step, time)) {
        ablate::radiation::Radiation::Solve(subDomain->GetSolutionVector(), subDomain->GetField("temperature"), subDomain->GetAuxVector());
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::radiation::VolumeRadiation::ComputeRHSFunction(PetscReal time, Vec solVec, Vec rhs) {
    PetscFunctionBegin;

    /** Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable. */
    const PetscScalar* rhsArray;
    VecGetArrayRead(rhs, &rhsArray);
    const auto& eulerFieldInfo = subDomain->GetField("euler");

    solver::Range cellRange;
    GetCellRange(cellRange);  //!< Gets the cell range to iterate over when retrieving cell indexes from the solver

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        PetscScalar* rhsValues;
        DMPlexPointLocalFieldRead(subDomain->GetDM(), iCell, eulerFieldInfo.id, rhsArray, &rhsValues);
        rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += GetIntensity(iCell);  //!< Loop through the cells and update the equation of state
    }
    RestoreRange(cellRange);
    VecRestoreArrayRead(rhs, &rhsArray);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::radiation::VolumeRadiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(ablate::domain::Region, "fieldBoundary", "boundary of the radiation region"),
         ARG(int, "rays", "number of rays used by the solver"), ARG(ablate::io::interval::Interval, "interval", "number of time steps between the radiation solves"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));