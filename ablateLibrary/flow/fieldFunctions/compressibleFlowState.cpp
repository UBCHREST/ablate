#include "compressibleFlowState.hpp"
#include <flow/processes/eulerAdvection.hpp>
ablate::flow::fieldFunctions::CompressibleFlowState::CompressibleFlowState(std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<mathFunctions::MathFunction> temperatureFunctionIn,
                                                                           std::shared_ptr<mathFunctions::MathFunction> pressureFunctionIn,
                                                                           std::shared_ptr<mathFunctions::MathFunction> velocityFunctionIn,
                                                                           std::shared_ptr<mathFunctions::FieldFunction> massFractionFunctionIn)
    : eos(eosIn), temperatureFunction(temperatureFunctionIn), pressureFunction(pressureFunctionIn), velocityFunction(velocityFunctionIn), massFractionFunction(massFractionFunctionIn) {
    // error checking
    // right now temperature and pressure are assumed, but this should be expended to handle any combination of primitive variables
    if (!temperatureFunction) {
        throw std::invalid_argument("The temperature must be specified in the flow::fieldSolutions::Euler initializer");
    }
    if (!pressureFunction) {
        throw std::invalid_argument("The pressure must be specified in the flow::fieldSolutions::Euler initializer");
    }
    if (!velocityFunction) {
        throw std::invalid_argument("The velocity must be specified in the flow::fieldSolutions::Euler initializer");
    }

    const auto &species = eos->GetSpecies();
    if (massFractionFunction == nullptr) {
        if (!species.empty()) {
            throw std::invalid_argument("The mass fractions must be specified because there are species in the EOS.");
        }
    }
}
PetscErrorCode ablate::flow::fieldFunctions::CompressibleFlowState::ComputeEulerFromState(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    auto flowState = (ablate::flow::fieldFunctions::CompressibleFlowState *)ctx;

    // get the temperature, pressure, and velocity
    PetscReal temperature;
    ierr = flowState->temperatureFunction->GetPetscFunction()(dim, time, x, 1, &temperature, flowState->temperatureFunction->GetContext());
    CHKERRQ(ierr);

    PetscReal pressure;
    ierr = flowState->pressureFunction->GetPetscFunction()(dim, time, x, 1, &pressure, flowState->pressureFunction->GetContext());
    CHKERRQ(ierr);

    PetscReal velocity[3];
    ierr = flowState->velocityFunction->GetPetscFunction()(dim, time, x, dim, velocity, flowState->velocityFunction->GetContext());
    CHKERRQ(ierr);

    // compute the mass fraction at this location
    std::vector<PetscReal> yi(flowState->eos->GetSpecies().size());
    if(flowState->massFractionFunction) {
        ierr = flowState->massFractionFunction->GetSolutionField().GetPetscFunction()(dim, time, x, yi.size(), &yi[0], flowState->massFractionFunction->GetSolutionField().GetContext());
        CHKERRQ(ierr);
    }

    // compute the density
    ierr = flowState->eos->GetComputeDensityFunctionFromTemperaturePressureFunction()(
        temperature, pressure, &yi[0], u + ablate::flow::processes::EulerAdvection::RHO, flowState->eos->GetComputeDensityFunctionFromTemperaturePressureContext());
    CHKERRQ(ierr);

    // compute the internal energy
    PetscReal sensibleInternalEnergy;
    ierr = flowState->eos->GetComputeSensibleInternalEnergyFunction()(
        temperature, u[ablate::flow::processes::EulerAdvection::RHO], &yi[0], &sensibleInternalEnergy, flowState->eos->GetComputeSensibleInternalEnergyContext());
    CHKERRQ(ierr);

    // convert to total sensibleEnergy
    PetscReal kineticEnergy = 0;
    for (PetscInt d = 0; d < dim; d++) {
        kineticEnergy += PetscSqr(velocity[d]);
    }
    kineticEnergy *= 0.5;
    u[ablate::flow::processes::EulerAdvection::RHOE] = u[ablate::flow::processes::EulerAdvection::RHO] * (kineticEnergy + sensibleInternalEnergy);

    // Set the vel*rho term
    for (PetscInt d = 0; d < dim; d++) {
        u[ablate::flow::processes::EulerAdvection::RHOU + d] = u[ablate::flow::processes::EulerAdvection::RHO] * velocity[d];
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::fieldFunctions::CompressibleFlowState::ComputeDensityYiFromState(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    auto flowState = (ablate::flow::fieldFunctions::CompressibleFlowState *)ctx;

    // make sure the that number of species is correct
    if ((PetscInt)flowState->eos->GetSpecies().size() != Nf) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "The number of species specified in the CompressibleFlowState does not match requested.");
    }

    // get the temperature, pressure, and velocity
    PetscReal temperature;
    ierr = flowState->temperatureFunction->GetPetscFunction()(dim, time, x, 1, &temperature, flowState->temperatureFunction->GetContext());
    CHKERRQ(ierr);

    PetscReal pressure;
    ierr = flowState->pressureFunction->GetPetscFunction()(dim, time, x, 1, &pressure, flowState->pressureFunction->GetContext());
    CHKERRQ(ierr);

    // compute the mass fraction at this location
    std::vector<PetscReal> yi(flowState->eos->GetSpecies().size());
    ierr = flowState->massFractionFunction->GetSolutionField().GetPetscFunction()(dim, time, x, yi.size(), &yi[0], flowState->massFractionFunction->GetSolutionField().GetContext());
    CHKERRQ(ierr);

    // compute the density
    PetscReal density;
    ierr =
        flowState->eos->GetComputeDensityFunctionFromTemperaturePressureFunction()(temperature, pressure, &yi[0], &density, flowState->eos->GetComputeDensityFunctionFromTemperaturePressureContext());
    CHKERRQ(ierr);

    // update the densityYi field
    for (PetscInt sp = 0; sp < Nf; sp++) {
        u[sp] = yi[sp] * density;
    }

    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTERDEFAULT(ablate::flow::fieldFunctions::CompressibleFlowState, ablate::flow::fieldFunctions::CompressibleFlowState,
                "a simple structure used to describe a compressible flow field using an EOS, T, pressure, vel, Yi", ARG(ablate::eos::EOS, "eos", "the eos used for the flow field"),
                ARG(ablate::mathFunctions::MathFunction, "temperature", "the temperature field (K)"), ARG(ablate::mathFunctions::MathFunction, "pressure", "the pressure field (Pa)"),
                ARG(ablate::mathFunctions::MathFunction, "velocity", "the velocity field (m/2)"),
                OPT(ablate::mathFunctions::FieldFunction, "massFractions", "a fieldFunctions used to describe all mass fractions"));
