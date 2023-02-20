#include "cameraDetector.hpp"

ablate::eos::radiationProperties::CameraDetector::CameraDetector(std::shared_ptr<eos::EOS> eosIn) : eos(std::move(eosIn)) {}

PetscErrorCode ablate::eos::radiationProperties::CameraDetector::CameraFunction(const PetscReal *conserved, PetscReal *kappa, void *ctx) {
    PetscFunctionBeginUser;



    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::CameraDetector::CameraTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *kappa, void *ctx) {
    PetscFunctionBeginUser;



    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::radiationProperties::CameraDetector::GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field> &fields) const {

    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicFunction{.function = CameraFunction,
                                         .context = std::make_shared<FunctionContext>(FunctionContext{
                                             .densityYiCSolidCOffset = cOffset,
                                             .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                             .densityFunction = eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::Density, fields)})};  //!< Create a struct to hold the offsets
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::SootAbsorptionModel");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::radiationProperties::CameraDetector::GetRadiationPropertiesTemperatureFunction(RadiationProperty property,
                                                                                                                                              const std::vector<domain::Field> &fields) const {
    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicTemperatureFunction{
                .function = CameraTemperatureFunction,
                .context = std::make_shared<FunctionContext>(
                    FunctionContext{.densityYiCSolidCOffset = cOffset,
                                    .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                    .densityFunction = eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::Density, fields)})};  //!< Create a struct to hold the offsets
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::SootAbsorptionModel");
    }
}

#include "registrar.hpp"
REGISTER(ablate::eos::radiationProperties::RadiationModel, ablate::eos::radiationProperties::CameraDetector, "CameraDetector",
         ARG(ablate::eos::EOS, "eos", "The EOS used to compute field properties"));