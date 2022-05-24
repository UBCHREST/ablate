#include "radiationRosseland.hpp"
#include "math.h"

ablate::eos::radiationProperties::Rosseland::Rosseland(std::shared_ptr<eos::EOS> eosIn) : eos(std::move(eosIn)) {}

PetscErrorCode ablate::eos::radiationProperties::Rosseland::RosselandFunction(const PetscReal *conserved, PetscReal *kappa, void *ctx) {
    PetscFunctionBeginUser;

    /** This model depends on mass fraction, temperature, and density in order to predict the absorption properties of the medium. */
    auto functionContext = (FunctionContext *)ctx;
    double temperature, density;  //!< Variables to hold information gathered from the fields
    PetscErrorCode ierr;          //!< Standard PETSc error code returned by PETSc functions

    ierr = functionContext->temperatureFunction.function(conserved, &temperature, functionContext->temperatureFunction.context.get());  //!< Get the temperature value at this location
    CHKERRQ(ierr);
    ierr = functionContext->densityFunction.function(conserved, &density, functionContext->densityFunction.context.get());  //!< Get the density value at this location
    CHKERRQ(ierr);

    /** Computing the Planck mean absorption coefficient for CO2 and H2O*/
    for (int j = 0; j < 7; j++) {  // std::array use
        kappaH2O += H2O_coeff->at(j) * pow(temperature / Tsurf, j);
        kappaCO2 += CO2_coeff->at(j) * pow(temperature / Tsurf, j);
    }
    kappaH2O = kapparef * pow(10, kappaH2O);
    kappaCO2 = kapparef * pow(10, kappaCO2);

    /** Computing the Planck mean absorption coefficient for CH4 and CO
     * The relationship is different with enough significance to use different models above or below 750 K.
     * */
    for (int j = 0; j < 5; j++) {
        kappaCH4 += CH4_coeff->at(j) * pow(temperature, j);
        if (temperature <= 750) {
            kappaCO += CO_1_coeff->at(j) * pow(temperature, j);
        } else {
            kappaCO += CO_2_coeff->at(j) * pow(temperature, j);
        }
    }

    /** Get the density mass fractions of the relevant species in order to compute their partial pressures
     * The conditional statement serves to set the mass fraction value to zero of the component does not exist in the field.
     * */
    PetscReal YinH2O = (functionContext->densityYiH2OOffset == -1) ? 0 : conserved[functionContext->densityYiH2OOffset] / density;
    PetscReal YinCO2 = (functionContext->densityYiCO2Offset == -1) ? 0 : conserved[functionContext->densityYiCO2Offset] / density;
    PetscReal YinCH4 = (functionContext->densityYiCH4Offset == -1) ? 0 : conserved[functionContext->densityYiCH4Offset] / density;
    PetscReal YinCO = (functionContext->densityYiCOOffset == -1) ? 0 : conserved[functionContext->densityYiCOOffset] / density;

    /** Computing the partial pressure of each species*/
    pCO2 = (density * UGC * YinCO2 * temperature) / (MWCO2 * 101325.);
    pH2O = (density * UGC * YinH2O * temperature) / (MWH2O * 101325.);
    pCH4 = (density * UGC * YinCH4 * temperature) / (MWCH4 * 101325.);
    pCO = (density * UGC * YinCO * temperature) / (MWCO * 101325.);

    /** The resulting absorptivity is an average of species absorptivity weighted by partial pressure. */
    *kappa = pCO2 * kappaCO2 + pH2O * kappaH2O + pCH4 * kappaCH4 + pCO * kappaCO;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::radiationProperties::Rosseland::RosselandTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *kappa, void *ctx) {
    PetscFunctionBeginUser;

    /** This model depends on mass fraction, temperature, and density in order to predict the absorption properties of the medium. */
    auto functionContext = (FunctionContext *)ctx;
    double density;       //!< Variables to hold information gathered from the fields
    PetscErrorCode ierr;  //!< Standard PETSc error code returned by PETSc functions

    ierr = functionContext->densityFunction.function(conserved, &density, functionContext->densityFunction.context.get());  //!< Get the density value at this location
    CHKERRQ(ierr);

    double kapparef = 1;  //!< Reference absorptivity
    double Tsurf = 300.;  //!< Reference temperature in Kelvin

    /** The Rosseland model uses a fit approximation of the absorptivity. This depends on the presence of four species which are present in combustion and shown below. */
    double kappaH2O = 0;
    double kappaCO2 = 0;
    double kappaCH4 = 0;
    double kappaCO = 0;
    double pCO2, pH2O, pCH4, pCO;

    /** These are the constants stored for the model. */
    std::array<double, 7> H2O_coeff[] = {{0.22317e1, -.15829e1, .1329601e1, -.50707, .93334e-1, -0.83108e-2, 0.28834e-3}};
    std::array<double, 7> CO2_coeff[] = {{0.38041E1, -0.27808E1, 0.11672E1, -0.284910E0, 0.38163E-1, -0.26292E-2, 0.73662E-4}};  //!< polynomial coefficients for H2O
    std::array<double, 5> CH4_coeff[] = {{6.6334E0, -3.5686E-3, 1.6682E-8, 2.5611E-10, -2.6558E-14}};                            //!< polynomial coefficients for CH4
    std::array<double, 5> CO_1_coeff[] = {{4.7869E0, -6.953E-2, 2.95775E-4, -4.25732E-7, 2.02894E-10}};                          //!< polynomial coefficients for CO with T <= 750 K
    std::array<double, 5> CO_2_coeff[] = {{1.0091E1, -1.183E-2, 4.7753E-6, -5.87209E-10, -2.5334E-14}};                          //!< polynomial coefficients for CO with T > 750 K
    double MWC = 1.2010700e+01;
    double MWO = 1.599940e+01;
    double MWH = 1.007940e+00;
    double UGC = 8314.4;  //!< Universal Gas Constant J/(K kmol)
    double MWCO = MWC + MWO;
    double MWCO2 = MWC + 2. * MWO;
    double MWCH4 = MWC + 4. * MWH;
    double MWH2O = 2. * MWH + MWO;

    /** Computing the Planck mean absorption coefficient for CO2 and H2O*/
    for (int j = 0; j < 7; j++) {  // std::array use
        kappaH2O += H2O_coeff->at(j) * pow(temperature / Tsurf, j);
        kappaCO2 += CO2_coeff->at(j) * pow(temperature / Tsurf, j);
    }
    kappaH2O = kapparef * pow(10, kappaH2O);
    kappaCO2 = kapparef * pow(10, kappaCO2);

    /** Computing the Planck mean absorption coefficient for CH4 and CO
     * The relationship is different with enough significance to use different models above or below 750 K.
     * */
    for (int j = 0; j < 5; j++) {
        kappaCH4 += CH4_coeff->at(j) * pow(temperature, j);
        if (temperature <= 750) {
            kappaCO += CO_1_coeff->at(j) * pow(temperature, j);
        } else {
            kappaCO += CO_2_coeff->at(j) * pow(temperature, j);
        }
    }

    /** Get the density mass fractions of the relevant species in order to compute their partial pressures
     * The conditional statement serves to set the mass fraction value to zero of the component does not exist in the field.
     * */
    PetscReal YinH2O = (functionContext->densityYiH2OOffset == -1) ? 0 : conserved[functionContext->densityYiH2OOffset] / density;
    PetscReal YinCO2 = (functionContext->densityYiCO2Offset == -1) ? 0 : conserved[functionContext->densityYiCO2Offset] / density;
    PetscReal YinCH4 = (functionContext->densityYiCH4Offset == -1) ? 0 : conserved[functionContext->densityYiCH4Offset] / density;
    PetscReal YinCO = (functionContext->densityYiCOOffset == -1) ? 0 : conserved[functionContext->densityYiCOOffset] / density;

    /** Computing the partial pressure of each species*/
    pCO2 = (density * UGC * YinCO2 * temperature) / (MWCO2 * 101325.);
    pH2O = (density * UGC * YinH2O * temperature) / (MWH2O * 101325.);
    pCH4 = (density * UGC * YinCH4 * temperature) / (MWCH4 * 101325.);
    pCO = (density * UGC * YinCO * temperature) / (MWCO * 101325.);

    /** The resulting absorptivity is an average of species absorptivity weighted by partial pressure. */
    *kappa = pCO2 * kappaCO2 + pH2O * kappaH2O + pCH4 * kappaCH4 + pCO * kappaCO;
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::radiationProperties::Rosseland::GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field> &fields) const {
    const auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });

    /** Check if the species exist in this run.
     * If all don't exist, throw an error.
     * If some don't exist, then the values should be set to zero for their mass fractions in all cases.
     * */
    if (densityYiField == fields.end()) {
        throw std::invalid_argument("Rosseland absorption model requires the density Yi field.");
    }

    /** Get the offsets that locate the position of the species mass fractions in the fields. */
    auto H2Ooffset = GetFieldComponentOffset("h2o", *densityYiField);
    auto CH4offset = GetFieldComponentOffset("ch4", *densityYiField);
    auto COoffset = GetFieldComponentOffset("co", *densityYiField);
    auto CO2offset = GetFieldComponentOffset("co2", *densityYiField);

    if (H2Ooffset == -1 && CH4offset == -1 && COoffset == -1 && CO2offset == -1) {
        throw std::invalid_argument("Rosseland absorption model requires at least one of the following: H2O, CH4, CO, CO2\n The Constant class allows the absorptivity of the medium to be set manually.");
    }

    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicFunction{.function = RosselandFunction,
                                         .context = std::make_shared<FunctionContext>(
                                             FunctionContext{.densityYiH2OOffset = H2Ooffset,
                                                             .densityYiCO2Offset = CO2offset,
                                                             .densityYiCOOffset = COoffset,
                                                             .densityYiCH4Offset = CH4offset,
                                                             .temperatureFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                                             .densityFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Density, fields)})};  //!< Create a struct to hold the offsets
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::Rosseland");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::radiationProperties::Rosseland::GetRadiationPropertiesTemperatureFunction(RadiationProperty property,
                                                                                                                                  const std::vector<domain::Field> &fields) const {
    const auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });

    /** Check if the species exist in this run.
     * If all don't exist, throw an error.
     * If some don't exist, then the values should be set to zero for their mass fractions in all cases.
     * */
    if (densityYiField == fields.end()) {
        throw std::invalid_argument("Rosseland absorption model requires the Density Yi field.");
    }

    /** Get the offsets that locate the position of the species mass fractions in the fields. */
    auto H2Ooffset = GetFieldComponentOffset("h2o", *densityYiField);
    auto CH4offset = GetFieldComponentOffset("ch4", *densityYiField);
    auto COoffset = GetFieldComponentOffset("co", *densityYiField);
    auto CO2offset = GetFieldComponentOffset("co2", *densityYiField);

    if (H2Ooffset == -1 && CH4offset == -1 && COoffset == -1 && CO2offset == -1) {
        throw std::invalid_argument("Rosseland absorption model requires at least one of the following: H2O, CH4, CO, CO2\n The Constant class allows the absorptivity of the medium to be set manually.");
    }

    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicTemperatureFunction{.function = RosselandTemperatureFunction,
                                                    .context = std::make_shared<FunctionContext>(FunctionContext{
                                                        .densityYiH2OOffset = H2Ooffset,
                                                        .densityYiCO2Offset = CO2offset,
                                                        .densityYiCOOffset = COoffset,
                                                        .densityYiCH4Offset = CH4offset,
                                                        .temperatureFunction = {},
                                                        .densityFunction = eos->GetThermodynamicFunction(ThermodynamicProperty::Density, fields)})};  //!< Create a struct to hold the offsets
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::Rosseland");
    }
}

PetscInt ablate::eos::radiationProperties::Rosseland::GetFieldComponentOffset(const std::string &str, const domain::Field &field) const {
    /** Returns the index where a certain field component can be found.
     * The index will be set to -1 if the component does not exist in the field.
     * */
    auto itr = std::find_if(field.components.begin(), field.components.end(), [&str](const auto &components) { return components == str; });
    PetscInt ind = (itr == field.components.end()) ? -1 : std::distance(field.components.begin(), itr) + field.offset;
    return ind;
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::eos::radiationProperties::RadiationModel, ablate::eos::radiationProperties::Rosseland, "Rosseland radiation properties model",
                 ARG(ablate::eos::EOS, "eos", "The EOS used to compute field properties"));