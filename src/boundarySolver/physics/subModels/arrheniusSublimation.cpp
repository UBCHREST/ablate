#include "arrheniusSublimation.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::boundarySolver::physics::subModels::ArrheniusSublimation::ArrheniusSublimation(const std::shared_ptr<ablate::parameters::Parameters>& properties,
                                                                                       const std::shared_ptr<ablate::mathFunctions::MathFunction>& initialization,
                                                                                       const std::shared_ptr<ablate::parameters::Parameters>& options)
    : properties(properties),
      initialization(initialization),
      options(options),
      latentHeatOfFusion(properties->GetExpect<PetscReal>("latentHeatOfFusion")),
      solidDensity(properties->GetExpect<PetscReal>("density")),
      preExponentialFactor(properties->GetExpect<PetscReal>("preExponentialFactor")),
      activationEnergy(properties->GetExpect<PetscReal>("activationEnergy")),
      parameterB(properties->Get<PetscReal>("B", 0.0)) {}

void ablate::boundarySolver::physics::subModels::ArrheniusSublimation::Initialize(ablate::boundarySolver::BoundarySolver& bSolver) {
    SublimationModel::Initialize(bSolver);

    // Get the rank
    PetscMPIInt rank;
    MPI_Comm_rank(bSolver.GetSubDomain().GetComm(), &rank) >> utilities::MpiUtilities::checkError;

    // Get the temperature field from the solver to set the init value
    auto temperatureField = bSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD);
    auto temperatureVec = bSolver.GetSubDomain().GetVec(temperatureField);
    auto temperatureDm = bSolver.GetSubDomain().GetFieldDM(temperatureField);

    // extract the array
    PetscScalar* temperatureArray;
    VecGetArray(temperatureVec, &temperatureArray) >> utilities::PetscUtilities::checkError;

    /** Initialize the solid boundary heat transfer model */
    for (const auto& geom : bSolver.GetBoundaryGeometry()) {
        // Create a unique name for this model based pon the faceId and rank
        auto uniqueId = std::to_string(rank) + "-" + std::to_string(geom.geometry.faceId);

        // by not having a maximum temperature we allow this to heat up as much as described
        oneDimensionHeatTransfer[geom.geometry.faceId] = std::make_shared<OneDimensionHeatTransfer>(uniqueId, properties, initialization, options);

        // Get the current surface temperature
        PetscReal currentSurfaceTemp;
        oneDimensionHeatTransfer[geom.geometry.faceId]->GetSurfaceTemperature(currentSurfaceTemp) >> utilities::PetscUtilities::checkError;

        // Get and set the temperature value
        PetscScalar* temperature;
        DMPlexPointGlobalFieldRef(temperatureDm, geom.cellId, temperatureField.id, temperatureArray, &temperature) >> utilities::PetscUtilities::checkError;
        temperature[0] = currentSurfaceTemp;
    }

    // restore
    VecRestoreArray(temperatureVec, &temperatureArray) >> utilities::PetscUtilities::checkError;
}

PetscErrorCode ablate::boundarySolver::physics::subModels::ArrheniusSublimation::Update(PetscInt faceId, PetscReal dt, PetscReal heatFluxToSurface, PetscReal& temperature) {
    PetscFunctionBegin;

    // compute the current mass flux used by arrhenius rat
    oneDimensionHeatTransfer[faceId]->GetSurfaceTemperature(temperature) >> utilities::PetscUtilities::checkError;
    PetscReal massFluxRate = ComputeMassFluxRate(temperature);        // kg/(m2-s)
    PetscReal energyMeltingRate = massFluxRate * latentHeatOfFusion;  // kg/(m2-s) * J/kg = J/(m2-s)

    // heatFluxToSurface
    heatFluxToSurface -= energyMeltingRate;

    // Step the time stepper in time
    PetscReal dummyVariable;
    PetscCall(oneDimensionHeatTransfer[faceId]->Solve(heatFluxToSurface, dt, temperature, dummyVariable));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ablate::boundarySolver::physics::subModels::ArrheniusSublimation::Compute(PetscInt faceId, PetscReal heatFluxToSurface,
                                                                                         ablate::boundarySolver::physics::subModels::SublimationModel::SurfaceState& surfaceState) {
    PetscFunctionBeginHot;
    PetscReal temperature;
    oneDimensionHeatTransfer[faceId]->GetSurfaceTemperature(temperature) >> utilities::PetscUtilities::checkError;

    // Compute the massFlux (we can only remove mass)
    surfaceState.massFlux = ComputeMassFluxRate(temperature);  // kg/(m2-s)
    surfaceState.regressionRate = surfaceState.massFlux / solidDensity;

    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode ablate::boundarySolver::physics::subModels::ArrheniusSublimation::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    for (auto& oneDimModel : oneDimensionHeatTransfer) {
        oneDimModel.second->Save(viewer, sequenceNumber, time);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ablate::boundarySolver::physics::subModels::ArrheniusSublimation::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    for (auto& oneDimModel : oneDimensionHeatTransfer) {
        oneDimModel.second->Restore(viewer, sequenceNumber, time);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal ablate::boundarySolver::physics::subModels::ArrheniusSublimation::ComputeMassFluxRate(PetscReal temperature) const {
    PetscReal rate = preExponentialFactor * PetscPowReal(temperature, parameterB) * PetscExpReal(-activationEnergy / (temperature * ugc));

    return rate * solidDensity;  // kg (m2*s)
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::physics::subModels::SublimationModel, ablate::boundarySolver::physics::subModels::ArrheniusSublimation,
         "Sublimation occurs at the specified temperature.  Extra heatFlux is used to heat the solid boundary",
         ARG(ablate::parameters::Parameters, "properties", "the heat transfer properties (specificHeat, conductivity, density, latentHeatOfFusion"),
         ARG(ablate::mathFunctions::MathFunction, "initialization", " math function to initialize the temperature"),
         OPT(ablate::parameters::Parameters, "options", "the petsc options for the solver/ts"));
