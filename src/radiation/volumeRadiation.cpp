#include "volumeRadiation.hpp"
#include "eos/radiationProperties/zimmer.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "io/interval/fixedInterval.hpp"

ablate::radiation::VolumeRadiation::VolumeRadiation(const std::string& solverId1, const std::shared_ptr<domain::Region>& region, const std::shared_ptr<io::interval::Interval>& intervalIn,
                                                    std::shared_ptr<radiation::Radiation> radiationIn, const std::shared_ptr<parameters::Parameters>& options,
                                                    const std::shared_ptr<ablate::monitors::logs::Log>& log)
    : CellSolver(solverId1, region, options), interval((intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>())), radiation(std::move(radiationIn)) {}
ablate::radiation::VolumeRadiation::~VolumeRadiation() = default;

void ablate::radiation::VolumeRadiation::Setup() {
    ablate::domain::Range cellRange;
    GetCellRange(cellRange);  //!< Gets the cell range that should be applied to the radiation solver

    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        PetscInt ghost = -1;
        if (ghostLabel) DMLabelGetValue(ghostLabel, iCell, &ghost) >> utilities::PetscUtilities::checkError;
        if (!(ghost >= 0)) radiationCellRange.Add(iCell);
    }

    ablate::solver::CellSolver::Setup();
    radiation->Setup(radiationCellRange.GetRange(), GetSubDomain());  //!< Insert the cell range of the solver here
    RestoreRange(cellRange);

    /**
     * Begins radiation properties model
     */
    absorptivityFunction = radiation->GetRadiationModel()->GetAbsorptionPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, subDomain->GetFields());
    if (absorptivityFunction.propertySize != 1) throw std::invalid_argument("The volume radiation solver currently only accepts one radiation wavelength.");
}

void ablate::radiation::VolumeRadiation::Register(std::shared_ptr<ablate::domain::SubDomain> subDomain) { ablate::solver::Solver::Register(subDomain); }

void ablate::radiation::VolumeRadiation::Initialize() {
    radiation->Initialize(radiationCellRange.GetRange(), GetSubDomain());  //!< Get the range of cells that the solver occupies in order for the radiation solver to give energy to the finite volume
}

PetscErrorCode ablate::radiation::VolumeRadiation::PreRHSFunction(TS ts, PetscReal time, bool initialStage, Vec locX) {
    PetscFunctionBegin;

    /** Only update the radiation solution if the sufficient interval has passed */
    PetscInt step;
    TSGetStepNumber(ts, &step) >> utilities::PetscUtilities::checkError;
    TSGetTime(ts, &time) >> utilities::PetscUtilities::checkError;
    if (initialStage && interval->Check(PetscObjectComm((PetscObject)ts), step, time)) {
        radiation->EvaluateGains(subDomain->GetSolutionVector(), subDomain->GetField("temperature"), subDomain->GetAuxVector());
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::radiation::VolumeRadiation::ComputeRHSFunction(PetscReal time, Vec solVec, Vec rhs) {
    PetscFunctionBegin;

    /** Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable. */
    const PetscScalar* rhsArray;
    VecGetArrayRead(rhs, &rhsArray);
    const auto& eulerFieldInfo = subDomain->GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    const auto& temperatureFieldInfo = subDomain->GetField(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD);

    /** Get the array of the solution vector. */
    const PetscScalar* solArray;
    DM solDm = subDomain->GetDM();
    VecGetArrayRead(solVec, &solArray);

    /** Get the array of the aux vector. */
    const PetscScalar* auxArray;
    DM auxDm = subDomain->GetAuxDM();
    VecGetArrayRead(subDomain->GetAuxGlobalVector(), &auxArray);

    /** Declare the basic information*/
    PetscReal* sol = nullptr;          //!< The solution value at any given location
    PetscReal* temperature = nullptr;  //!< The temperature at any given location
    double kappa = 1;                  //!< Absorptivity coefficient, property of each cell

    auto absorptivityFunctionContext = absorptivityFunction.context.get();  //!< Get access to the absorption function

    //!< This will iterate only though local cells
    auto& range = radiationCellRange.GetRange();
    for (PetscInt c = range.start; c < range.end; ++c) {
        const PetscInt iCell = range.GetPoint(c);  //!< Isolates the valid cells

        // compute absorptivity
        DMPlexPointLocalRead(solDm, iCell, solArray, &sol) >> utilities::PetscUtilities::checkError;

        if (sol) {
            DMPlexPointLocalFieldRead(auxDm, iCell, temperatureFieldInfo.id, auxArray, &temperature) >> utilities::PetscUtilities::checkError;

            absorptivityFunction.function(sol, *temperature, &kappa, absorptivityFunctionContext);

            PetscScalar* rhsValues;
            DMPlexPointLocalFieldRead(subDomain->GetDM(), iCell, eulerFieldInfo.id, rhsArray, &rhsValues);
            PetscReal intensity[1];  //! This implies that there is currently only support for one wavelength in the volumetric radiation solver.
            //! Implement a wavelength dependant absorption integration here if desired.
            radiation->GetIntensity(intensity, c, range, *temperature, kappa);              //!< Loop through the cells and update the equation of state
            rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += intensity[0];  //! Add the solution of this intensity.
        }
    }
    VecRestoreArrayRead(rhs, &rhsArray);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::radiation::VolumeRadiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(ablate::io::interval::Interval, "interval", "number of time steps between the radiation solves"),
         ARG(ablate::radiation::Radiation, "radiation", "a radiation solver to allow for choice between multiple implementations"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
