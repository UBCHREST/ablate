#include "compressibleFlowState.hpp"
ablate::flow::fieldSolutions::CompressibleFlowState::CompressibleFlowState(std::shared_ptr<ablate::eos::EOS> eos, std::shared_ptr<mathFunctions::MathFunction> temperatureFunction,
                                                                           std::shared_ptr<mathFunctions::MathFunction> pressureFunction, std::shared_ptr<mathFunctions::MathFunction> velocityFunction,
                                                                           std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> yiFunctionsIn) {
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
    if (!species.empty()) {
        if (yiFunctionsIn.empty()) {
            throw std::invalid_argument("The mass fractions must be specified because there are species in the EOS.");
        }
    } else {
        // Map the mass fractions to species
        massFractionFunctions.resize(species.size(), nullptr);

        // march over every yiFunctionIn
        for (const auto &yiFunction : yiFunctionsIn) {
            auto it = std::find(species.begin(), species.end(), yiFunction->GetName());

            if (it != species.end()) {
                massFractionFunctions[std::distance(species.begin(), it)] = yiFunction->GetFieldFunction();
            } else {
                throw std::invalid_argument("Cannot find field species " + yiFunction->GetName());
            }
        }
    }
}
PetscErrorCode ablate::flow::fieldSolutions::CompressibleFlowState::ComputeEulerFromState(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    auto eulerInit = (ablate::flow::fieldSolutions::CompressibleFlowState *)ctx;

    // get the temperature, pressure, and velocity
    PetscReal temperature;
    ierr = eulerInit->temperatureFunction->GetPetscFunction()(dim, time, x, 1, &temperature, eulerInit->temperatureFunction->GetContext());
    CHKERRQ(ierr);

    PetscReal pressure;
    ierr = eulerInit->pressureFunction->GetPetscFunction()(dim, time, x, 1, &pressure, eulerInit->pressureFunction->GetContext());
    CHKERRQ(ierr);

    PetscReal velocity[3];
    ierr = eulerInit->velocityFunction->GetPetscFunction()(dim, time, x, dim, velocity, eulerInit->velocityFunction->GetContext());
    CHKERRQ(ierr);

    // compute the pressure

    PetscFunctionReturn(0);
}
