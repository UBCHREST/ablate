#include "sootMeanAbsorption.hpp"

ablate::eos::radiationProperties::SootMeanAbsorption::SootMeanAbsorption(std::shared_ptr<eos::EOS> eosIn) : eos(std::move(eosIn)) {}

PetscErrorCode ablate::eos::radiationProperties::SootMeanAbsorption::SootFunction(const PetscReal *conserved, PetscReal *kappa, void *ctx) {
    PetscFunctionBeginUser;
    /** This model depends on mass fraction, temperature, and density in order to predict the absorption properties of the medium. */
    auto functionContext = (FunctionContext *)ctx;
    double temperature, density;  //!< Variables to hold information gathered from the fields
    PetscErrorCode ierr;          //!< Standard PETSc error code returned by PETSc functions

    ierr = functionContext->temperatureFunction.function(conserved, &temperature, functionContext->temperatureFunction.context.get());  //!< Get the temperature value at this location
    CHKERRQ(ierr);
    ierr = functionContext->densityFunction.function(conserved, &density, functionContext->densityFunction.context.get());  //!< Get the density value at this location
    CHKERRQ(ierr);

    PetscReal YinC = (functionContext->densityEVCOffset == -1) ? 0 : conserved[functionContext->densityEVCOffset] / density;  //!< Get the mass fraction of carbon here

    PetscReal fv = density * YinC / rhoC;

    *kappa = (3.72 * fv * C_0 * temperature) / C_2;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::SootMeanAbsorption::SootTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *kappa, void *ctx) {
    PetscFunctionBeginUser;
    /** This model depends on mass fraction, temperature, and density in order to predict the absorption properties of the medium. */
    auto functionContext = (FunctionContext *)ctx;
    double density;       //!< Variables to hold information gathered from the fields
    PetscErrorCode ierr;  //!< Standard PETSc error code returned by PETSc functions

    ierr = functionContext->densityFunction.function(conserved, &density, functionContext->densityFunction.context.get());  //!< Get the density value at this location
    CHKERRQ(ierr);

    PetscReal YinC = (functionContext->densityEVCOffset == -1) ? 0 : conserved[functionContext->densityEVCOffset] / density;  //!< Get the mass fraction of carbon here

    PetscReal fv = density * YinC / rhoC;

    *kappa = (3.72 * fv * C_0 * temperature) / C_2;

    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::radiationProperties::SootMeanAbsorption::GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field> &fields) const {
    const auto densityEVField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_EV_FIELD; });

    /** Check if the species exist in this run.
     * If all don't exist, throw an error.
     * If some don't exist, then the values should be set to zero for their mass fractions in all cases.
     * */
    if (densityEVField == fields.end()) {
        throw std::invalid_argument("Soot absorption model requires the density Yi field.");
    }

    /** Get the offsets that locate the position of the solid carbon field. */
    auto Coffset = GetFieldComponentOffset("c_solid", *densityEVField);

    if (Coffset == -1) {
        throw std::invalid_argument("Soot absorption model requires solid carbon.\n The Constant class allows the absorptivity of the medium to be set manually.");
    }

    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicFunction{.function = SootFunction,
                                         .context = std::make_shared<FunctionContext>(
                                             FunctionContext{.densityEVCOffset = Coffset,
                                                             .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                                             .densityFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Density, fields)})};  //!< Create a struct to hold the offsets
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::SootAbsorptionModel");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::radiationProperties::SootMeanAbsorption::GetRadiationPropertiesTemperatureFunction(RadiationProperty property,
                                                                                                                                              const std::vector<domain::Field> &fields) const {
    const auto densityEVField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_EV_FIELD; });

    /** Check if the species exist in this run.
     * If all don't exist, throw an error.
     * If some don't exist, then the values should be set to zero for their mass fractions in all cases.
     * */
    if (densityEVField == fields.end()) {
        throw std::invalid_argument("Soot absorption model requires the density Yi field.");
    }

    /** Get the offsets that locate the position of the solid carbon field. */
    auto Coffset = GetFieldComponentOffset("c_solid", *densityEVField);

    if (Coffset == -1) {
        throw std::invalid_argument("Soot absorption model requires solid carbon.\n The Constant class allows the absorptivity of the medium to be set manually.");
    }

    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicTemperatureFunction{.function = SootTemperatureFunction,
                                                    .context = std::make_shared<FunctionContext>(FunctionContext{
                                                        .densityEVCOffset = Coffset,
                                                        .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                                        .densityFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Density, fields)})};  //!< Create a struct to hold the offsets
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::SootAbsorptionModel");
    }
}

PetscInt ablate::eos::radiationProperties::SootMeanAbsorption::GetFieldComponentOffset(const std::string &str, const domain::Field &field) const {
    /** Returns the index where a certain field component can be found.
     * The index will be set to -1 if the component does not exist in the field.
     * Convert all component names to lower case for string comparison
     * */
    auto itr = std::find_if(field.components.begin(), field.components.end(), [&str](const auto &components) {
        std::string component = components;
        std::transform(component.begin(), component.end(), component.begin(), [](unsigned char c) { return std::tolower(c); });
        return component == str;
    });
    PetscInt ind = (itr == field.components.end()) ? -1 : std::distance(field.components.begin(), itr) + field.offset;
    return ind;
}