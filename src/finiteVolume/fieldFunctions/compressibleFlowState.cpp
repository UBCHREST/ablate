#include "compressibleFlowState.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "mathFunctions/functionWrapper.hpp"

ablate::finiteVolume::fieldFunctions::CompressibleFlowState::CompressibleFlowState(std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<mathFunctions::MathFunction> temperatureFunctionIn,
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

    const auto& species = eos->GetSpecies();
    if (massFractionFunction == nullptr) {
        if (!species.empty()) {
            throw std::invalid_argument("The mass fractions must be specified because there are species in the EOS.");
        }
    }
}

std::shared_ptr<ablate::mathFunctions::MathFunction> ablate::finiteVolume::fieldFunctions::CompressibleFlowState::GetFieldFunction(const std::string& field) const {
    auto eosFunction = eos->GetFieldFunctionFunction(field, eos::ThermodynamicProperty::Temperature, eos::ThermodynamicProperty::Pressure);

    auto fieldFunction = [eos = this->eos,
                          temperatureFunction = this->temperatureFunction,
                          pressureFunction = this->pressureFunction,
                          velocityFunction = this->velocityFunction,
                          massFractionFunction = this->massFractionFunction,
                          eosFunction](int dim, double time, const double x[], int nf, double* u, void* ctx) {
        // get the temperature, pressure, and velocity
        PetscReal temperature;
        PetscCall(temperatureFunction->GetPetscFunction()(dim, time, x, 1, &temperature, temperatureFunction->GetContext()));

        PetscReal pressure;
        PetscCall(pressureFunction->GetPetscFunction()(dim, time, x, 1, &pressure, pressureFunction->GetContext()));

        PetscReal velocity[3];
        PetscCall(velocityFunction->GetPetscFunction()(dim, time, x, dim, velocity, velocityFunction->GetContext()));

        // compute the mass fraction at this location
        std::vector<PetscReal> yi(eos->GetSpecies().size());
        if (massFractionFunction) {
            PetscCall(massFractionFunction->GetSolutionField().GetPetscFunction()(dim, time, x, yi.size(), yi.data(), massFractionFunction->GetSolutionField().GetContext()));
        }

        eosFunction(temperature, pressure, dim, velocity, yi.data(), u);
        return 0;
    };

    return std::make_shared<mathFunctions::FunctionWrapper>(fieldFunction);
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::finiteVolume::fieldFunctions::CompressibleFlowState, ablate::finiteVolume::fieldFunctions::CompressibleFlowState,
                 "a simple structure used to describe a compressible flow field using an EOS, T, pressure, vel, Yi", ARG(ablate::eos::EOS, "eos", "the eos used for the flow field"),
                 ARG(ablate::mathFunctions::MathFunction, "temperature", "the temperature field (K)"), ARG(ablate::mathFunctions::MathFunction, "pressure", "the pressure field (Pa)"),
                 ARG(ablate::mathFunctions::MathFunction, "velocity", "the velocity field (m/2)"),
                 OPT(ablate::mathFunctions::FieldFunction, "massFractions", "a fieldFunctions used to describe all mass fractions"));
