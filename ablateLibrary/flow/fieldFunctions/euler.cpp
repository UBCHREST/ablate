#include "euler.hpp"
#include <mathFunctions/functionPointer.hpp>

ablate::flow::fieldFunctions::Euler::Euler(std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<mathFunctions::MathFunction> temperatureFunctionIn,
                                           std::shared_ptr<mathFunctions::MathFunction> pressureFunctionIn, std::shared_ptr<mathFunctions::MathFunction> velocityFunctionIn)
    : ablate::mathFunctions::FieldFunction("euler", std::make_shared<ablate::mathFunctions::FunctionPointer>(EulerFromTemperatureAndPressure, this)),
      eos(eosIn),
      temperatureFunction(temperatureFunctionIn),
      pressureFunction(pressureFunctionIn),
      velocityFunction(velocityFunctionIn)

{
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
}

PetscErrorCode ablate::flow::fieldFunctions::Euler::EulerFromTemperatureAndPressure(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    auto eulerInit = (ablate::flow::fieldFunctions::Euler *)ctx;

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

    // compute

    PetscFunctionReturn(0);
}
