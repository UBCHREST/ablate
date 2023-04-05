#include "tChemSoot.hpp"
#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_CUDA
#include <utility>
#include "eos/tChemSoot/densityFcn.hpp"
#include "eos/tChemSoot/sensibleInternalEnergy.hpp"
#include "eos/tChemSoot/sensibleInternalEnergyFcn.hpp"
#include "eos/tChemSoot/sourceCalculatorSoot.hpp"
#include "eos/tChemSoot/specificHeatConstantPressure.hpp"
#include "eos/tChemSoot/specificHeatConstantVolume.hpp"
#include "eos/tChemSoot/speedOfSound.hpp"
#include "eos/tChemSoot/stateVectorSoot.hpp"
#include "eos/tChemSoot/temperature.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "monitors/logs/nullLog.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::eos::TChemSoot::TChemSoot(std::filesystem::path mechanismFile, std::filesystem::path thermoFile, std::shared_ptr<ablate::monitors::logs::Log> log,
                                  const std::shared_ptr<ablate::parameters::Parameters> &options)
    : TChemBase("TChemSoot", std::move(mechanismFile), std::move(thermoFile), std::move(log), options) {
    // Insert carbon as the first species
    species.insert(species.begin(), CSolidName);

    // Use the computed enthalpy and insert solid carbon as first index
    enthalpyReferenceDevice = real_type_1d_view("reference enthalpy", kineticsModelDataDevice->nSpec + 1);
    auto enthalpyReferenceWithCarbonHost = Kokkos::create_mirror_view(enthalpyReferenceDevice);

    // copy to enthalpyReference
    Kokkos::deep_copy(Kokkos::subview(enthalpyReferenceWithCarbonHost, std::make_pair(1, kineticsModelDataDevice->nSpec + 1)), enthalpyReferenceHost);

    // Now put in reference enthalpy for Carbon
    enthalpyReferenceWithCarbonHost(0) = CarbonEnthalpy_R_T(TREF) * TREF * kineticsModelDataDevice->Runiv / tChemSoot::MWCarbon;

    // Replace the org calc
    enthalpyReferenceHost = enthalpyReferenceWithCarbonHost;
    Kokkos::deep_copy(enthalpyReferenceDevice, enthalpyReferenceHost);
}

std::shared_ptr<ablate::eos::TChemSoot::FunctionContext> ablate::eos::TChemSoot::BuildFunctionContext(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TChem requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });
    if (densityYiField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TChem requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD Field");
    }

    // determine the state vector size
    const ordinal_type stateVecDim = tChemSoot::getStateVectorSootSize(kineticsModelDataDevice->nSpec);
    const ordinal_type batchSize = 1;

    // get the property string
    auto propertyName = std::string(eos::to_string(property));

    // set device information
    real_type_2d_view_host stateHost(propertyName + " state device", batchSize, stateVecDim);
    real_type_2d_view_host perSpeciesHost(propertyName + " perSpecies device", batchSize, kineticsModelDataDevice->nSpec + 1);  // Include YCarbon_Solid as Part of the species array
    real_type_1d_view_host mixtureHost(propertyName + " mixture device", batchSize);

    auto per_team_scratch_cp = tChemLib::Scratch<real_type_1d_view>::shmem_size(std::get<2>(thermodynamicFunctions.at(property))(kineticsModelDataDevice->nSpec));

    auto policy = tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type(batchSize, Kokkos::AUTO());
    policy.set_scratch_size(1, Kokkos::PerTeam((int)per_team_scratch_cp));

    return std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,    // Number of physical dimensions
                                                             .eulerOffset = eulerField->offset,          // Offset of data to eulerField
                                                             .densityYiOffset = densityYiField->offset,  // Offset of data to density filed

                                                             // copy host info
                                                             .stateHost = stateHost,
                                                             .perSpeciesHost = perSpeciesHost,
                                                             .mixtureHost = mixtureHost,

                                                             // store the reference enthalpy
                                                             .enthalpyReferenceHost = enthalpyReferenceHost,  // Full Reference enthalpy information

                                                             // policy
                                                             .policy = policy,

                                                             // kinetics data
                                                             .kineticsModelDataHost = kineticsModelDataHost});
}

// These Next 5 are the same as regular TCHEM
ablate::eos::ThermodynamicFunction ablate::eos::TChemSoot::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    return ThermodynamicFunction{.function = std::get<0>(thermodynamicFunctions.at(property)),
                                 .context = BuildFunctionContext(property, fields),
                                 .propertySize = speciesSizedProperties.count(property) ? (PetscInt)species.size() : 1};
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::TChemSoot::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    return ThermodynamicTemperatureFunction{.function = std::get<1>(thermodynamicFunctions.at(property)),
                                            .context = BuildFunctionContext(property, fields),
                                            .propertySize = speciesSizedProperties.count(property) ? (PetscInt)species.size() : 1};
}
void ablate::eos::TChemSoot::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tmechFile: " << mechanismFile << std::endl;
    if (!thermoFile.empty()) {
        stream << "\tthermoFile: " << thermoFile << std::endl;
    }
    stream << "\tnumberSpecies: " << species.size() << std::endl;
    tChemLib::exec_space().print_configuration(stream, true);
    tChemLib::host_exec_space().print_configuration(stream, true);
}

// Returns the Total Density of the Mixture, This is a conserved variable and can just be returned
PetscErrorCode ablate::eos::TChemSoot::DensityFunction(const PetscReal *conserved, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChemSoot::DensityTemperatureFunction(const PetscReal *conserved, PetscReal, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}

// Returns the Temperature of the System, Returned from an iterative solution on the total sensible energy
PetscErrorCode ablate::eos::TChemSoot::TemperatureFunction(const PetscReal *conserved, PetscReal *property, void *ctx) { return TemperatureTemperatureFunction(conserved, 300, property, ctx); }

PetscErrorCode ablate::eos::TChemSoot::TemperatureTemperatureFunction(const PetscReal *conserved, PetscReal temperatureGuess, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    // Compute the internal energy from total ener
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    // assumed eos
    PetscReal internalEnergyRef = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;

    // Fill the working array
    auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(functionContext->kineticsModelDataHost->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));

    FillWorkingVectorFromDensityMassFractions(density, temperatureGuess, conserved + functionContext->densityYiOffset, stateHost);
    functionContext->mixtureHost[0] = internalEnergyRef;

    // compute the temperature
    ablate::eos::tChemSoot::Temperature::runHostBatch(functionContext->policy,
                                                      functionContext->stateHost,
                                                      functionContext->mixtureHost,
                                                      functionContext->perSpeciesHost,
                                                      functionContext->enthalpyReferenceHost,
                                                      *functionContext->kineticsModelDataHost);

    // copy back the results
    *temperature = stateHost.Temperature();

    PetscFunctionReturn(0);
}

// Calculate the Internal Sensible Energy From the current state, i.e. Etot - KE
PetscErrorCode ablate::eos::TChemSoot::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    *sensibleInternalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChemSoot::InternalSensibleEnergyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *sensibleEnergyTemperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(functionContext->kineticsModelDataHost->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);

    ablate::eos::tChemSoot::SensibleInternalEnergy::runHostBatch(functionContext->policy,
                                                                 functionContext->stateHost,
                                                                 functionContext->mixtureHost,
                                                                 functionContext->perSpeciesHost,
                                                                 functionContext->enthalpyReferenceHost,
                                                                 *functionContext->kineticsModelDataHost);

    *sensibleEnergyTemperature = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

// Grab Pressure from conserved variables (All Pressure should be good!)
PetscErrorCode ablate::eos::TChemSoot::PressureFunction(const PetscReal *conserved, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    // Compute Temperature
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    // Compute pressure with the temperature now known
    ierr = PressureTemperatureFunction(conserved, temperature, pressure, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
// Compute Pressure Assuming we know what the Temperature is
PetscErrorCode ablate::eos::TChemSoot::PressureTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(functionContext->kineticsModelDataHost->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));

    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);

    // compute the Pressure With Temperature Stored in StateDevice
    ablate::eos::tChemSoot::Pressure::runHostBatch(functionContext->policy, functionContext->stateHost, *functionContext->kineticsModelDataHost);

    // copy back the results
    *pressure = stateHost.Pressure();  // 1 is pressure state spot

    PetscFunctionReturn(0);
}

// Compute Sensible Enthalpy Without the temperature known, just the conservative values
PetscErrorCode ablate::eos::TChemSoot::SensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // Compute the temperature
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    // Now we know the temperature! compute the sensible enthalpy
    ierr = SensibleEnthalpyTemperatureFunction(conserved, temperature, sensibleEnthalpy, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
// Compute it with the Temperature Known
PetscErrorCode ablate::eos::TChemSoot::SensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(functionContext->kineticsModelDataHost->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);

    ablate::eos::tChemSoot::SensibleEnthalpy::runHostBatch(functionContext->policy,
                                                           functionContext->stateHost,
                                                           functionContext->mixtureHost,
                                                           functionContext->perSpeciesHost,
                                                           functionContext->enthalpyReferenceHost,
                                                           *functionContext->kineticsModelDataHost);

    *sensibleEnthalpy = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChemSoot::SpeedOfSoundFunction(const PetscReal *conserved, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpeedOfSoundTemperatureFunction(conserved, temperature, speedOfSound, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChemSoot::SpeedOfSoundTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(functionContext->kineticsModelDataHost->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);

    ablate::eos::tChemSoot::SpeedOfSound::runHostBatch(functionContext->policy, functionContext->stateHost, functionContext->mixtureHost, *functionContext->kineticsModelDataHost);

    *speedOfSound = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChemSoot::SpecificHeatConstantPressureFunction(const PetscReal *conserved, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpecificHeatConstantPressureTemperatureFunction(conserved, temperature, cp, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChemSoot::SpecificHeatConstantPressureTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    // The specific heat is a mass weighted term and thus can be broken up into a gaseos and solid contribution scaled by their relative mass fractions i.e (1-Yc) and (Yc)
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(functionContext->kineticsModelDataHost->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);

    ablate::eos::tChemSoot::SpecificHeatConstantPressure::runHostBatch(functionContext->policy, functionContext->stateHost, functionContext->mixtureHost, *functionContext->kineticsModelDataHost);

    *cp = functionContext->mixtureHost(0);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChemSoot::SpecificHeatConstantVolumeFunction(const PetscReal *conserved, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpecificHeatConstantVolumeTemperatureFunction(conserved, temperature, cv, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChemSoot::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(functionContext->kineticsModelDataHost->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);

    ablate::eos::tChemSoot::SpecificHeatConstantVolume::runHostBatch(functionContext->policy, functionContext->stateHost, functionContext->mixtureHost, *functionContext->kineticsModelDataHost);

    *cv = functionContext->mixtureHost(0);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChemSoot::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpeciesSensibleEnthalpyTemperatureFunction(conserved, temperature, hi, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChemSoot::SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(functionContext->kineticsModelDataHost->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);

    ablate::eos::tChemSoot::SensibleEnthalpy::runHostBatch(functionContext->policy,
                                                           functionContext->stateHost,
                                                           functionContext->mixtureHost,
                                                           functionContext->perSpeciesHost,
                                                           functionContext->enthalpyReferenceHost,
                                                           *functionContext->kineticsModelDataHost);

    Kokkos::View<PetscReal *> hiHost(hi, functionContext->kineticsModelDataHost->nSpec + 1);
    Kokkos::deep_copy(hiHost, Kokkos::subview(functionContext->perSpeciesHost, 0, Kokkos::ALL()));
    PetscFunctionReturn(0);
}

void ablate::eos::TChemSoot::FillWorkingVectorFromDensityMassFractions(double density, double temperature, const double *densityYi,
                                                                       const tChemSoot::StateVectorSoot<real_type_1d_view_host> &stateVector) {
    // As a Reminder StateVector Assumed to follow -> {total Density, Pressure, Temperature, Total SpeciesMass Fraction of Gas states, Carbon Mass Fraction, Ndd}
    stateVector.Temperature() = temperature;
    stateVector.Density() = density;
    stateVector.Pressure() = NAN;           // Pressure set to NAN
    stateVector.SootNumberDensity() = NAN;  // should not be used
    // Ignore the First species as it is the carbon solid species
    real_type yiSum = densityYi[0] / density;  // start with carbon value
    stateVector.MassFractionCarbon() = yiSum;  // carbon index is 3+kmcd_numspecies = 2+totNumSpecies

    auto ys = stateVector.MassFractions();

    for (ordinal_type s = 0; s < stateVector.NumGasSpecies() - 1; s++) {  // Dilute species is totNumSpec -1 in density Yi, and 3+totNumSpec-2 in totalState vector
        ys[s] = PetscMax(0.0, densityYi[s + 1] / density);
        ys[s] = PetscMin(1.0, ys[s]);
        yiSum += ys[s];
    }
    if (yiSum > 1.0) {
        for (PetscInt s = 0; s < stateVector.NumGasSpecies() - 1; s++) {
            // Limit the bounds
            ys[s] /= yiSum;
        }
        stateVector.MassFractionCarbon() /= yiSum;  // have to do carbon out of the loop since it jumps the dilute last species in statevector
        ys[stateVector.NumGasSpecies() - 1] = 0.0;  // Set dilute species to 0
    } else {
        ys[stateVector.NumGasSpecies() - 1] = 1.0 - yiSum;  // Set dilute species to 1-YiSum
    }
}

ablate::eos::EOSFunction ablate::eos::TChemSoot::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2,
                                                                          std::vector<std::string> otherProperties) const {
    if (otherProperties != std::vector<std::string>{YI}) {
        throw std::invalid_argument("ablate::eos::TChemSoot expects the other properties to be Yi (Species Mass Fractions)");
    }

    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            // assume that the lambda are running on host
            using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
            using host_type = tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type::member_type;

            // create reusable data for the lambda (all on host)
            auto kineticsModelDataHost = tChemLib::createGasKineticModelConstData<host_device_type>(kineticsModel);
            real_type_1d_view_host stateHostView("state device", tChemSoot::getStateVectorSootSize(kineticsModelDataDevice->nSpec));  // 1 Extra for YCarbon
            auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, stateHostView);

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());
            auto per_team_extent = (int)tChemSoot::SensibleInternalEnergy::getWorkSpaceSize(kineticsModelDataDevice->nSpec + 1);
            policy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size(per_team_extent)));
            // store hi for the
            real_type_1d_view_host enthalpy("enthalpy", this->species.size());

            auto tp = [=](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                Kokkos::parallel_for(
                    "tp init", policy, KOKKOS_LAMBDA(const host_type &member) {
                        stateHost.Temperature() = temperature;
                        stateHost.Pressure() = pressure;
                        // fill the state
                        auto yiHost = stateHost.MassFractions();
                        // It is assumed that the first species are the gas species, so we rearrange the values here so Carbon is in the last spot
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataDevice->nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i + 1]; });
                        auto Yc = yi[0];
                        // Make Sure to add the Carbon Value into YiHost
                        stateHost.MassFractionCarbon() = Yc;

                        // compute density
                        PetscReal density = ablate::eos::tChemSoot::impl::densityFcn<real_type, host_device_type>::team_invoke(member, stateHost, kineticsModelDataHost);

                        // Create a Gaseous State Vector
                        real_type_1d_view_host state_at_i_gas = real_type_1d_view_host("Gaseous", ::TChem::Impl::getStateVectorSize(kineticsModelDataHost.nSpec));
                        // Get the Gaseous State Vector
                        Impl::StateVector<real_type_1d_view_host> sv_gas(kineticsModelDataHost.nSpec, state_at_i_gas);
                        stateHost.SplitYiState(sv_gas);

                        // compute the sensible energy
                        Scratch<real_type_1d_view_host> work(member.team_scratch(1), per_team_extent);
                        auto cpks = real_type_1d_view_host((real_type *)work.data(), per_team_extent);

                        auto sensibleInternalEnergy = ablate::eos::tChemSoot::impl::SensibleInternalEnergyFcn<real_type, host_device_type>::team_invoke(
                            member, Yc, sv_gas, enthalpy, cpks, enthalpyReferenceHost, kineticsModelDataHost);

                        // convert to total sensibleEnergy
                        PetscReal kineticEnergy = 0;
                        for (PetscInt d = 0; d < dim; d++) {
                            kineticEnergy += PetscSqr(velocity[d]);
                        }
                        kineticEnergy *= 0.5;

                        conserved[ablate::finiteVolume::CompressibleFlowFields::RHO] = density;
                        conserved[ablate::finiteVolume::CompressibleFlowFields::RHOE] = density * (kineticEnergy + sensibleInternalEnergy);
                        for (PetscInt d = 0; d < dim; d++) {
                            conserved[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density * velocity[d];
                        }
                    });
            };
            if (property1 == ThermodynamicProperty::Temperature) {
                return tp;
            } else {
                return [tp](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    tp(temperature, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        if ((property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy)) {
            // assume that the lambda are running on host
            using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
            using host_type = tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type::member_type;

            // create reusable data for the lambda (all on host)
            auto kineticsModelDataHost = tChemLib::createGasKineticModelConstData<host_device_type>(kineticsModel);
            real_type_2d_view_host stateHostView("state device", 1, tChemSoot::getStateVectorSootSize(kineticsModelDataDevice->nSpec));
            auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, Kokkos::subview(stateHostView, 0, Kokkos::ALL()));

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());
            auto per_team_extent = (int)tChemSoot::SensibleInternalEnergy::getWorkSpaceSize(kineticsModelDataDevice->nSpec + 1);
            policy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size(per_team_extent)));
            // store hi for the
            real_type_2d_view_host enthalpy("enthalpy", 1, kineticsModelDataDevice->nSpec);
            // store internal energy for the
            real_type_1d_view_host internalEnergy("internal energy", 1);

            auto iep = [=](PetscReal sensibleInternalEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                stateHost.Temperature() = 300.;  // Temperature Initial Guess
                stateHost.Pressure() = pressure;
                internalEnergy(0) = sensibleInternalEnergy;
                auto Yc = stateHost.MassFractionCarbon();
                // fill the state
                auto yiHost = stateHost.MassFractions();
                Kokkos::parallel_for(
                    "tp init", policy, KOKKOS_LAMBDA(const host_type &member) {
                        // fill the state
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataDevice->nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i + 1]; });
                    });
                // Make Sure to add the Carbon Value into YiHost
                stateHost.MassFractionCarbon() = Yc;

                // compute the temperature
                eos::tChemSoot::Temperature::runHostBatch(policy, stateHostView, internalEnergy, enthalpy, enthalpyReferenceDevice, kineticsModelDataHost);

                // Compute the Density
                Kokkos::parallel_for(
                    "tp Density Calc", policy, KOKKOS_LAMBDA(const host_type &member) {
                        PetscReal density = ablate::eos::tChemSoot::impl::densityFcn<real_type, host_device_type>::team_invoke(member, stateHost, kineticsModelDataHost);
                        // convert to total sensibleEnergy
                        PetscReal kineticEnergy = 0;
                        for (PetscInt d = 0; d < dim; d++) {
                            kineticEnergy += PetscSqr(velocity[d]);
                        }
                        kineticEnergy *= 0.5;

                        conserved[ablate::finiteVolume::CompressibleFlowFields::RHO] = density;
                        conserved[ablate::finiteVolume::CompressibleFlowFields::RHOE] = density * (kineticEnergy + sensibleInternalEnergy);
                        for (PetscInt d = 0; d < dim; d++) {
                            conserved[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density * velocity[d];
                        }
                    });
            };
            if (property1 == ThermodynamicProperty::InternalSensibleEnergy) {
                return iep;
            } else {
                return [iep](PetscReal pressure, PetscReal sensibleInternalEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    iep(sensibleInternalEnergy, pressure, dim, velocity, yi, conserved);
                };
            }
        }
        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::PerfectGas.");
    } else if (finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            // assume that the lambda are running on host
            using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
            using host_type = tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type::member_type;

            // create reusable data for the lambda (all on host)
            auto kineticsModelDataHost = tChemLib::createGasKineticModelConstData<host_device_type>(kineticsModel);
            real_type_1d_view_host stateHostView("state device", tChemSoot::getStateVectorSootSize(kineticsModelDataDevice->nSpec));  // 1 Extra for YCarbon

            auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, stateHostView);

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());

            auto densityYiFromTP = [=](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                Kokkos::parallel_for(
                    "densityYi init from T & P", policy, KOKKOS_LAMBDA(const host_type &member) {
                        // fill the state
                        stateHost.Temperature() = temperature;
                        stateHost.Pressure() = pressure;
                        auto yiHost = stateHost.MassFractions();
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i + 1]; });
                        auto Yc = yi[0];
                        // Make Sure to add the Carbon Value into YiHost
                        stateHost.MassFractionCarbon() = Yc;

                        // compute density
                        PetscReal density = ablate::eos::tChemSoot::impl::densityFcn<real_type, host_device_type>::team_invoke(member, stateHost, kineticsModelDataHost);
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec + 1), [&](const ordinal_type &i) { conserved[i] = density * yi[i]; });
                    });
            };

            if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure)) {
                return densityYiFromTP;
            } else {
                return [=](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    densityYiFromTP(temperature, pressure, dim, velocity, yi, conserved);
                };
            }

        } else if ((property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure) ||
                   (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy)) {
            // assume that the lambda are running on host
            using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
            using host_type = tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type::member_type;

            // create reusable data for the lambda (all on host)
            auto kineticsModelDataHost = tChemLib::createGasKineticModelConstData<host_device_type>(kineticsModel);
            real_type_2d_view_host stateHostView("state device", 1, tChemSoot::getStateVectorSootSize(kineticsModelDataDevice->nSpec));
            auto stateHost = tChemSoot::StateVectorSoot<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, Kokkos::subview(stateHostView, 0, Kokkos::ALL()));

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());
            auto per_team_extent = (int)tChemSoot::SensibleInternalEnergy::getWorkSpaceSize(kineticsModelDataDevice->nSpec + 1);
            policy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size(per_team_extent)));
            // store hi for the
            real_type_2d_view_host enthalpy("enthalpy", 1, kineticsModelDataDevice->nSpec);
            // store internal energy for the
            real_type_1d_view_host internalEnergy("internal energy", 1);

            auto densityYiFromIeP = [=](PetscReal sensibleInternalEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                stateHost.Temperature() = 300.;  // Temperature Initial Guess
                stateHost.Pressure() = pressure;
                internalEnergy(0) = sensibleInternalEnergy;
                auto Yc = yi[0];
                // fill the state
                auto yiHost = stateHost.MassFractions();
                Kokkos::parallel_for(
                    "tp init", policy, KOKKOS_LAMBDA(const host_type &member) {
                        // fill the state
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataDevice->nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i + 1]; });
                    });
                // Make Sure to add the Carbon Value into YiHost
                stateHost.MassFractionCarbon() = Yc;

                // compute the temperature
                eos::tChemSoot::Temperature::runHostBatch(policy, stateHostView, internalEnergy, enthalpy, enthalpyReferenceDevice, kineticsModelDataHost);

                // Compute the Density
                Kokkos::parallel_for(
                    "tp Density Calc", policy, KOKKOS_LAMBDA(const host_type &member) {
                        PetscReal density = ablate::eos::tChemSoot::impl::densityFcn<real_type, host_device_type>::team_invoke(member, stateHost, kineticsModelDataHost);
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec + 1), [&](const ordinal_type &i) { conserved[i] = density * yi[i]; });
                    });
            };

            if (property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure) {
                return densityYiFromIeP;
            } else {
                return [=](PetscReal pressure, PetscReal sensibleInternalEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    densityYiFromIeP(sensibleInternalEnergy, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::PerfectGas.");
    } else {
        throw std::invalid_argument("Unknown field type " + field + " for ablate::eos::PerfectGas.");
    }
}

std::map<std::string, double> ablate::eos::TChemSoot::GetElementInformation() const {
    auto eNamesHost = kineticsModel.eNames_.view_host();
    auto eMassHost = kineticsModel.eMass_.view_host();

    // Create the map
    std::map<std::string, double> elementInfo;

    for (ordinal_type i = 0; i < kineticsModel.nElem_; i++) {
        std::string elementName(&eNamesHost(i, 0));
        elementInfo[elementName] = eMassHost(i);
    }

    return elementInfo;
}

// should be good
std::map<std::string, std::map<std::string, int>> ablate::eos::TChemSoot::GetSpeciesElementalInformation() const {
    // build the element names
    auto eNamesHost = kineticsModel.eNames_.view_host();
    std::vector<std::string> elementNames;
    for (ordinal_type i = 0; i < kineticsModel.nElem_; ++i) {
        elementNames.emplace_back(&eNamesHost(i, 0));
    }

    std::map<std::string, std::map<std::string, int>> speciesElementInfo;

    // get the element info
    auto elemCountHost = kineticsModel.elemCount_.view_host();

    // march over each species
    for (std::size_t sp = 1; sp < species.size(); ++sp) {
        auto &speciesMap = speciesElementInfo[species[sp]];

        for (ordinal_type e = 0; e < kineticsModel.nElem_; ++e) {
            speciesMap[elementNames[e]] = elemCountHost(sp - 1, e);
        }
    }
    // Do Carbon by itself
    auto &speciesMap = speciesElementInfo[species[0]];
    for (ordinal_type e = 0; e < kineticsModel.nElem_; ++e) {
        if (elementNames[e] == "C")
            speciesMap["C"] = 1;
        else
            speciesMap[elementNames[e]] = 0;
    }
    return speciesElementInfo;
}

// Should be good
std::map<std::string, double> ablate::eos::TChemSoot::GetSpeciesMolecularMass() const {
    // march over each species
    auto sMass = kineticsModel.sMass_.view_host();

    std::map<std::string, double> mw;
    for (std::size_t sp = 1; sp < species.size(); ++sp) {
        mw[species[sp]] = sMass((ordinal_type)(sp - 1));
    }
    mw[species[0]] = ablate::eos::tChemSoot::MWCarbon;

    return mw;
}

std::shared_ptr<ablate::eos::ChemistryModel::SourceCalculator> ablate::eos::TChemSoot::CreateSourceCalculator(const std::vector<domain::Field> &fields, const ablate::domain::Range &cellRange) {
    return std::make_shared<ablate::eos::tChemSoot::SourceCalculatorSoot>(fields, shared_from_this(), constraints, cellRange);
}

#endif  // KOKKOS_ENABLE_CUDA

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::TChemSoot, "[TChemV2](https://github.com/sandialabs/TChem) ideal gas eos augmented with a soot formation mechanism",
         ARG(std::filesystem::path, "mechFile", "the mech file (CHEMKIN Format or Cantera Yaml)"), OPT(std::filesystem::path, "thermoFile", "the thermo file (CHEMKIN Format if mech file is CHEMKIN)"),
         OPT(ablate::monitors::logs::Log, "log", "An optional log for TChem echo output (only used with yaml input)"),
         OPT(ablate::parameters::Parameters, "options",
             "time stepping options (dtMin, dtMax, dtDefault, dtEstimateFactor, relToleranceTime, relToleranceTime, absToleranceTime, relToleranceNewton, absToleranceNewton, maxNumNewtonIterations, "
             "numTimeIterationsPerInterval, jacobianInterval, maxAttempts, thresholdTemperature)"));
