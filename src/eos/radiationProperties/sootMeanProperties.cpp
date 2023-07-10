#include "sootMeanProperties.hpp"

ablate::eos::radiationProperties::SootMeanProperties::SootMeanProperties(std::shared_ptr<eos::EOS> eosIn) : eos(std::move(eosIn)) {}

PetscErrorCode ablate::eos::radiationProperties::SootMeanProperties::SootEmissionTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *epsilon, void *ctx) {
    PetscFunctionBeginUser;

    PetscReal refractiveIndex = GetRefractiveIndex();
    *(epsilon) = ablate::radiation::Radiation::GetBlackBodyTotalIntensity(temperature, refractiveIndex);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::SootMeanProperties::SootAbsorptionTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *kappa, void *ctx) {
    PetscFunctionBeginUser;
    /** This model depends on mass fraction, temperature, and density in order to predict the absorption properties of the medium. */
    auto functionContext = (FunctionContext *)ctx;
    double density;  //!< Variables to hold information gathered from the fields
                     //!< Standard PETSc error code returned by PETSc functions

    PetscCall(functionContext->densityFunction.function(conserved, temperature, &density, functionContext->densityFunction.context.get()));  //!< Get the density value at this location
    PetscReal YinC = (functionContext->densityYiCSolidCOffset == -1) ? 0 : conserved[functionContext->densityYiCSolidCOffset] / density;     //!< Get the mass fraction of carbon here

    PetscReal fv = density * YinC / rhoC;

    *kappa = (3.72 * fv * C_0 * temperature) / C_2;

    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::radiationProperties::SootMeanProperties::GetRadiationPropertiesTemperatureFunction(RadiationProperty property,
                                                                                                                                              const std::vector<domain::Field> &fields) const {
    const auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });

    /** Check if the species exist in this run.
     * If all don't exist, throw an error.
     * If some don't exist, then the values should be set to zero for their mass fractions in all cases.
     * */
    if (densityYiField == fields.end()) {
        throw std::invalid_argument("Soot absorption model requires the density Yi field.");
    }

    /** Get the offsets that locate the position of the solid carbon field. */
    PetscInt cOffset = (PetscInt)densityYiField->ComponentOffset(TChemSoot::CSolidName);

    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicTemperatureFunction{
                .function = SootAbsorptionTemperatureFunction,
                .context = std::make_shared<FunctionContext>(
                    FunctionContext{.densityYiCSolidCOffset = cOffset,
                                    .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                    .densityFunction = eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::Density, fields)})};  //!< Create a struct to hold the offsets
        case RadiationProperty::Emissivity:
            return ThermodynamicTemperatureFunction{
                .function = SootEmissionTemperatureFunction,
                .context = std::make_shared<FunctionContext>(
                    FunctionContext{.densityYiCSolidCOffset = cOffset,
                                    .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                    .densityFunction = eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::Density, fields)})};  //!< Create a struct to hold the offsets
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::SootAbsorptionModel");
    }
}

#include "registrar.hpp"
REGISTER(ablate::eos::radiationProperties::RadiationModel, ablate::eos::radiationProperties::SootMeanProperties, "SootMeanAbsorption",
         ARG(ablate::eos::EOS, "eos", "The EOS used to compute field properties"));