#include "constant.hpp"

ablate::eos::radiationProperties::Constant::Constant(std::shared_ptr<eos::EOS> eosIn, double absorptivity, double emissivity)
    : eos(std::move(eosIn)), absorptivityIn(absorptivity), emissivityIn(emissivity) {}

PetscErrorCode ablate::eos::radiationProperties::Constant::ConstantAbsorptionFunction(const PetscReal conserved[], PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = *((double *)ctx);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::Constant::ConstantAbsorptionTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = *((double *)ctx);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::Constant::ConstantEmissionFunction(const PetscReal conserved[], PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    /** This model depends on mass fraction, temperature, and density in order to predict the absorption properties of the medium. */
    auto functionContext = (FunctionContext *)ctx;
    double temperature;  //!< Variables to hold information gathered from the fields
    //!< Standard PETSc error code returned by PETSc functions

    PetscCall(functionContext->temperatureFunction.function(conserved, &temperature, functionContext->temperatureFunction.context.get()));  //!< Get the temperature value at this location

    PetscReal refractiveIndex = GetRefractiveIndex();  //! We may want to incorporate this at some point?
    *(property) = ablate::radiation::Radiation::GetBlackBodyTotalIntensity(temperature, refractiveIndex);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::Constant::ConstantEmissionTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal refractiveIndex = GetRefractiveIndex();  //! We may want to incorporate this at some point?
    *(property) = ablate::radiation::Radiation::GetBlackBodyTotalIntensity(temperature, refractiveIndex);
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::radiationProperties::Constant::GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field> &fields) const {
    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicFunction{
                .function = ConstantAbsorptionFunction,
                .context = std::make_shared<FunctionContext>(
                    FunctionContext{.absorptivity = absorptivityIn, .emissivity = emissivityIn, .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields)})};
        case RadiationProperty::Emissivity:
            return ThermodynamicFunction{
                .function = ConstantEmissionFunction,
                .context = std::make_shared<FunctionContext>(
                    FunctionContext{.absorptivity = absorptivityIn, .emissivity = emissivityIn, .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields)})};
        default:
            throw std::invalid_argument("Unknown radiationProperties property ablate::eos::radiationProperties::Constant");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::radiationProperties::Constant::GetRadiationPropertiesTemperatureFunction(RadiationProperty property,
                                                                                                                                    const std::vector<domain::Field> &fields) const {
    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicTemperatureFunction{
                .function = ConstantAbsorptionTemperatureFunction,
                .context = std::make_shared<FunctionContext>(
                    FunctionContext{.absorptivity = absorptivityIn, .emissivity = emissivityIn, .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields)})};
        case RadiationProperty::Emissivity:
            return ThermodynamicTemperatureFunction{
                .function = ConstantEmissionTemperatureFunction,
                .context = std::make_shared<FunctionContext>(
                    FunctionContext{.absorptivity = absorptivityIn, .emissivity = emissivityIn, .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields)})};
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::Constant");
    }
}

#include "registrar.hpp"
REGISTER(ablate::eos::radiationProperties::RadiationModel, ablate::eos::radiationProperties::Constant, "constant value transport model (often used for testing)",
         ARG(ablate::eos::EOS, "eos", "The EOS used to compute field properties"), ARG(double, "absorptivity", "radiative absorptivity"), ARG(double, "emissivity", "radiative emissivity"));