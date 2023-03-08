#include "sootSpectrumProperties.hpp"

ablate::eos::radiationProperties::SootSpectrumProperties::SootSpectrumProperties(std::shared_ptr<eos::EOS> eosIn, int num, double min, double max, const std::vector<double> &wavelengths,
                                                                                 const std::vector<double> &bandwidths)
    : eos(std::move(eosIn)), wavelengthsIn(std::move(wavelengths)), bandwidthsIn(std::move(bandwidths)) {
    if ((std::empty(wavelengthsIn) && (num == 0)) || (!std::empty(wavelengthsIn) && (num != 0))) {
        throw std::invalid_argument("The spectrum soot model requires definition of either the number of wavelengths, or a vector of wavelengths to be integrated. One must be chosen.");
    }

    if (wavelengthsIn.size() != bandwidthsIn.size()) {
        throw std::invalid_argument("The size of the wavelengths and bandwidths inputs must be equal to one another.");
    }

    //! If a range is given, initialize a linear variation in wavelength over the desired range.
    if (std::empty(wavelengthsIn)) {
        wavelengthsIn.resize(num);
        double widths = (max - min) / num;
        for (int i = 0; i < num; i++) {
            wavelengthsIn[i] = min + ((double)i / (double)num) * max;
            bandwidthsIn[i] = widths; //! Default the bandwidths to cover the whole range.
        }
    }
}

// Bandwidth of 10 nanometers is assumed for the filters. Constant emissivity over the bandwidth.

PetscErrorCode ablate::eos::radiationProperties::SootSpectrumProperties::SootEmissionFunction(const PetscReal *conserved, PetscReal *epsilon, void *ctx) {
    PetscFunctionBeginUser;

    auto functionContext = (FunctionContext *)ctx;
    double temperature;                                                                                                                     //!< Variables to hold information gathered from the fields
    PetscCall(functionContext->temperatureFunction.function(conserved, &temperature, functionContext->temperatureFunction.context.get()));  //!< Get the temperature value at this location

    for (size_t i = 0; i < functionContext->wavelengths.size(); i++) {
        epsilon[i] = ablate::radiation::Radiation::GetBlackBodyWavelengthIntensity(
            temperature, functionContext->wavelengths[i], GetRefractiveIndex(functionContext->wavelengths[i]));  //! Get the black body intensity at the temperature and wavelength specified.
        epsilon[i] *= functionContext->bandwidths[i];  //! Multiply it by the bandwidth under constant assumption to get the power integration.
        /**
         * In other models we may want to implement a smarter integration.
         */
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::SootSpectrumProperties::SootEmissionTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *epsilon, void *ctx) {
    PetscFunctionBeginUser;

    auto functionContext = (FunctionContext *)ctx;

    for (size_t i = 0; i < functionContext->wavelengths.size(); i++) {
        epsilon[i] = ablate::radiation::Radiation::GetBlackBodyWavelengthIntensity(
            temperature, functionContext->wavelengths[i], GetRefractiveIndex(functionContext->wavelengths[i]));  //! Get the black body intensity at the temperature and wavelength specified.
        epsilon[i] *= functionContext->bandwidths[i];  //! Multiply it by the bandwidth under constant assumption to get the power integration.
        /**
         * In other models we may want to implement a smarter integration.
         */
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::SootSpectrumProperties::SootAbsorptionFunction(const PetscReal *conserved, PetscReal *kappa, void *ctx) {
    PetscFunctionBeginUser;

    /** This model depends on mass fraction, temperature, and density in order to predict the absorption properties of the medium. */
    auto functionContext = (FunctionContext *)ctx;
    double temperature, density;  //!< Variables to hold information gathered from the fields
    //!< Standard PETSc error code returned by PETSc functions

    PetscCall(functionContext->temperatureFunction.function(conserved, &temperature, functionContext->temperatureFunction.context.get()));   //!< Get the temperature value at this location
    PetscCall(functionContext->densityFunction.function(conserved, temperature, &density, functionContext->densityFunction.context.get()));  //!< Get the density value at this location

    PetscReal YinC = (functionContext->densityYiCSolidCOffset == -1) ? 0 : conserved[functionContext->densityYiCSolidCOffset] / density;  //!< Get the mass fraction of carbon here

    PetscReal n;
    PetscReal k;
    PetscReal fv = density * YinC / rhoC;
    for (size_t i = 0; i < functionContext->wavelengths.size(); i++) {
        PetscReal lambda = functionContext->wavelengths[i] * 1E6;
        //! This is the wavelength. (We must integrate over the valid range of wavelengths.)
        n = GetRefractiveIndex(lambda);  //! Fit of model to data.
        k = GetAbsorptiveIndex(lambda);  //! Fit of model to data.
        kappa[i] = (36 * ablate::utilities::Constants::pi * n * k * fv) / (((((n * n) - (k * k) + 2) * ((n * n) - (k * k) + 2)) + (4 * n * n * k * k)) * (lambda * 1E-6));
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::radiationProperties::SootSpectrumProperties::SootAbsorptionTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *kappa, void *ctx) {
    PetscFunctionBeginUser;

    /** This model depends on mass fraction, temperature, and density in order to predict the absorption properties of the medium. */
    auto functionContext = (FunctionContext *)ctx;
    double density;  //!< Variables to hold information gathered from the fields
                     //!< Standard PETSc error code returned by PETSc functions

    PetscCall(functionContext->densityFunction.function(conserved, temperature, &density, functionContext->densityFunction.context.get()));  //!< Get the density value at this location
    PetscReal YinC = (functionContext->densityYiCSolidCOffset == -1) ? 0 : conserved[functionContext->densityYiCSolidCOffset] / density;     //!< Get the mass fraction of carbon here

    PetscReal n;
    PetscReal k;
    PetscReal fv = density * YinC / rhoC;
    for (size_t i = 0; i < functionContext->wavelengths.size(); i++) {
        PetscReal lambda = functionContext->wavelengths[i] * 1E6;  //! Must convert to micrometers because of the model fit
        //! This is the wavelength. (We must integrate over the valid range of wavelengths.)
        n = GetRefractiveIndex(lambda);  //! Fit of model to data.
        k = GetAbsorptiveIndex(lambda);  //! Fit of model to data.

        kappa[i] = (36 * ablate::utilities::Constants::pi * n * k * fv) / (((((n * n) - (k * k) + 2) * ((n * n) - (k * k) + 2)) + (4 * n * n * k * k)) * (lambda * 1E-6));
    }

    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::radiationProperties::SootSpectrumProperties::GetRadiationPropertiesFunction(RadiationProperty property,
                                                                                                                            const std::vector<domain::Field> &fields) const {
    const auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });

    /** Check if the species exist in this run.
     * If all don't exist, throw an error.
     * If some don't exist, then the values should be set to zero for their mass fractions in all cases.
     * */
    if (densityYiField == fields.end()) {
        throw std::invalid_argument("Soot absorption model requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD.");
    }

    /** Get the offsets that locate the position of the solid carbon field. */
    PetscInt cOffset = (PetscInt)densityYiField->ComponentOffset(TChemSoot::CSolidName);

    if (cOffset == -1) {
        throw std::invalid_argument("Soot absorption model requires solid carbon.\n The Constant class allows the absorptivity of the medium to be set manually.");
    }

    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicFunction{
                .function = SootAbsorptionFunction,
                .context = std::make_shared<FunctionContext>(FunctionContext{.densityYiCSolidCOffset = cOffset,
                                                                             .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                                                             .densityFunction = eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::Density, fields),
                                                                             .wavelengths = wavelengthsIn}),
                .propertySize = (int)wavelengthsIn.size()};  //!< Create a struct to hold the offsets
        case RadiationProperty::Emissivity:
            return ThermodynamicFunction{
                .function = SootEmissionFunction,
                .context = std::make_shared<FunctionContext>(FunctionContext{.densityYiCSolidCOffset = cOffset,
                                                                             .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                                                             .densityFunction = eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::Density, fields),
                                                                             .wavelengths = wavelengthsIn,
                                                                             .bandwidths = bandwidthsIn}),
                .propertySize = (int)wavelengthsIn.size()};  //!< Create a struct to hold the offsets
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::SootAbsorptionModel");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::radiationProperties::SootSpectrumProperties::GetRadiationPropertiesTemperatureFunction(RadiationProperty property,
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
                .context = std::make_shared<FunctionContext>(FunctionContext{.densityYiCSolidCOffset = cOffset,
                                                                             .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                                                             .densityFunction = eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::Density, fields),
                                                                             .wavelengths = wavelengthsIn,
                                                                             .bandwidths = bandwidthsIn}),
                .propertySize = (int)wavelengthsIn.size()};  //!< Create a struct to hold the offsets
        case RadiationProperty::Emissivity:
            return ThermodynamicTemperatureFunction{
                .function = SootEmissionTemperatureFunction,
                .context = std::make_shared<FunctionContext>(FunctionContext{.densityYiCSolidCOffset = cOffset,
                                                                             .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                                                             .densityFunction = eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::Density, fields),
                                                                             .wavelengths = wavelengthsIn,
                                                                             .bandwidths = bandwidthsIn}),
                .propertySize = (int)wavelengthsIn.size()};  //!< Create a struct to hold the offsets
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::SootAbsorptionModel");
    }
}

#include "registrar.hpp"
REGISTER(ablate::eos::radiationProperties::RadiationModel, ablate::eos::radiationProperties::SootSpectrumProperties, "SootSpectrumAbsorption",
         ARG(ablate::eos::EOS, "eos", "The EOS used to compute field properties"), OPT(int, "num", "number of wavelengths that are integrated in the model"),
         OPT(double, "min", "number of wavelengths that are integrated in the model"), OPT(double, "max", "number of wavelengths that are integrated in the model"),
         OPT(std::vector<double>, "wavelengths", "number of wavelengths that are integrated in the model"), OPT(std::vector<double>, "bandwidths", "bandwidth of each associated wavelength"));