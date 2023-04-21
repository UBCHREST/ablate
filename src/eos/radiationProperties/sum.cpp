#include "sum.hpp"

ablate::eos::radiationProperties::Sum::Sum(std::vector<std::shared_ptr<ablate::eos::radiationProperties::RadiationModel>> models) : models(std::move(models)) {
    if (this->models.empty()) {
        throw std::invalid_argument("The model list must not be empty");
    }
}

PetscErrorCode ablate::eos::radiationProperties::Sum::SumFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    auto vector = (std::vector<ThermodynamicFunction> *)ctx;

    *property = 0;
    for (const auto &[subFunction, subCtx, _] : *vector) {
        PetscReal propertyTmp = 0.0;
        PetscCall(subFunction(conserved, &propertyTmp, subCtx.get()));
        *property += propertyTmp;
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::radiationProperties::Sum::SumTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;

    auto vector = (std::vector<ThermodynamicTemperatureFunction> *)ctx;

    *property = 0;
    for (const auto &[subFunction, subCtx, _] : *vector) {
        PetscReal propertyTmp = 0.0;
        PetscCall(subFunction(conserved, temperature, &propertyTmp, subCtx.get()));
        *property += propertyTmp;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::Sum::EmissionFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;

    /**
     * Only take the first value of emission in the vector.
     * We don't want to sum the emissions because the emission value is only dependent on the temperature, not the volume / mass fraction of the contributor.
     */
    auto vector = (std::vector<ThermodynamicFunction> *)ctx;

    *property = 0;
    for (const auto &[subFunction, subCtx, _] : vector[0]) {
        PetscReal propertyTmp = 0.0;
        PetscCall(subFunction(conserved, &propertyTmp, subCtx.get()));
        *property += propertyTmp;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::Sum::EmissionTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;

    /**
     * Only take the first value of emission in the vector.
     * We don't want to sum the emissions because the emission value is only dependent on the temperature, not the volume / mass fraction of the contributor.
     */
    auto vector = (std::vector<ThermodynamicTemperatureFunction> *)ctx;

    *property = 0;
    for (const auto &[subFunction, subCtx, _] : vector[0]) {
        PetscReal propertyTmp = 0.0;
        PetscCall(subFunction(conserved, temperature, &propertyTmp, subCtx.get()));
        *property += propertyTmp;
    }

    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::radiationProperties::Sum::GetRadiationPropertiesFunction(ablate::eos::radiationProperties::RadiationProperty property,
                                                                                                         const std::vector<domain::Field> &fields) const {
    switch (property) {
        case RadiationProperty::Absorptivity: {  // Create the function
            auto contextVector = std::make_shared<std::vector<ThermodynamicFunction>>();
            auto function = ThermodynamicFunction{.function = SumFunction, .context = contextVector};

            // add each contribution
            for (auto &model : models) {
                contextVector->push_back(model->GetRadiationPropertiesFunction(property, fields));
            }

            return function;
        }
        case RadiationProperty::Emissivity: {
            auto contextVector = std::make_shared<std::vector<ThermodynamicFunction>>();
            auto function = ThermodynamicFunction{.function = EmissionFunction, .context = contextVector};

            // add each contribution
            for (auto &model : models) {
                contextVector->push_back(model->GetRadiationPropertiesFunction(property, fields));
            }

            return function;
        }
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::Constant");
    }
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::radiationProperties::Sum::GetRadiationPropertiesTemperatureFunction(ablate::eos::radiationProperties::RadiationProperty property,
                                                                                                                               const std::vector<domain::Field> &fields) const {
    switch (property) {
        case RadiationProperty::Absorptivity: {  // Create the function
            auto contextVector = std::make_shared<std::vector<ThermodynamicTemperatureFunction>>();
            auto function = ThermodynamicTemperatureFunction{.function = SumTemperatureFunction, .context = contextVector};

            // add each contribution
            for (auto &model : models) {
                contextVector->push_back(model->GetRadiationPropertiesTemperatureFunction(property, fields));
            }

            return function;
        }
        case RadiationProperty::Emissivity: {
            auto contextVector = std::make_shared<std::vector<ThermodynamicTemperatureFunction>>();
            auto function = ThermodynamicTemperatureFunction{.function = EmissionTemperatureFunction, .context = contextVector};

            // add each contribution
            for (auto &model : models) {
                contextVector->push_back(model->GetRadiationPropertiesTemperatureFunction(property, fields));
            }

            return function;
        }
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::Constant");
    }
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::eos::radiationProperties::RadiationModel, ablate::eos::radiationProperties::Sum, "sums the properties of the provided models",
                      std::vector<ablate::eos::radiationProperties::RadiationModel>);