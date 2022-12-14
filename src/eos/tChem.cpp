#include "tChem.hpp"
#include <TChem_EnthalpyMass.hpp>
#include <utility>
#include "TChem_SpecificHeatCapacityConsVolumePerMass.hpp"
#include "TChem_SpecificHeatCapacityPerMass.hpp"
#include "eos/tChem/sensibleInternalEnergy.hpp"
#include "eos/tChem/sensibleInternalEnergyFcn.hpp"
#include "eos/tChem/speedOfSound.hpp"
#include "eos/tChem/temperature.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "monitors/logs/nullLog.hpp"
#include "utilities/kokkosUtilities.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::eos::TChem::TChem(std::filesystem::path mechanismFileIn, std::filesystem::path thermoFileIn, std::shared_ptr<ablate::monitors::logs::Log> logIn,
                          const std::shared_ptr<ablate::parameters::Parameters> &options)
    : ChemistryModel("TChem"), mechanismFile(std::move(mechanismFileIn)), thermoFile(std::move(thermoFileIn)), log(logIn ? logIn : std::make_shared<ablate::monitors::logs::NullLog>()) {
    // setup/use Kokkos
    ablate::utilities::KokkosUtilities::Initialize();

    // create/parse the kinetic data
    if (thermoFile.empty()) {
        // Create a file to record the output
        kineticsModel = tChemLib::KineticModelData(mechanismFile.string(), log->GetStream(), log->GetStream());
    } else {
        // TChem init reads/writes file it can only be done one at a time
        ablate::utilities::MpiUtilities::RoundRobin(PETSC_COMM_WORLD, [&](int rank) { kineticsModel = tChemLib::KineticModelData(mechanismFile.string(), thermoFile.string()); });
    }

    // get the device KineticsModelData
    kineticsModelDataDevice = std::make_shared<tChemLib::KineticModelGasConstData<typename Tines::UseThisDevice<exec_space>::type>>(
        tChemLib::createGasKineticModelConstData<typename Tines::UseThisDevice<exec_space>::type>(kineticsModel));

    // copy the species information
    const auto speciesNamesHost = Kokkos::create_mirror_view(kineticsModelDataDevice->speciesNames);
    Kokkos::deep_copy(speciesNamesHost, kineticsModelDataDevice->speciesNames);
    // resize the species data
    species.resize(kineticsModelDataDevice->nSpec);
    auto speciesArray = species.data();

    Kokkos::parallel_for(
        "speciesInit", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(0, kineticsModelDataDevice->nSpec), KOKKOS_LAMBDA(const auto i) {
            speciesArray[i] = std::string(&speciesNamesHost(i, 0));
        });
    Kokkos::fence();

    // compute the reference enthalpy
    enthalpyReference = real_type_1d_view("reference enthalpy", kineticsModelDataDevice->nSpec);

    {  // manually compute reference enthalpy on the device
        const auto per_team_extent_h = tChemLib::EnthalpyMass::getWorkSpaceSize(*kineticsModelDataDevice);
        const auto per_team_scratch_h = Scratch<real_type_1d_view>::shmem_size(per_team_extent_h);
        typename tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type policy_enthalpy(1, Kokkos::AUTO());
        policy_enthalpy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size(per_team_scratch_h)));

        // set the state
        real_type_2d_view stateDevice(" state device", 1, tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec));
        auto stateHostView = Kokkos::create_mirror_view(stateDevice);
        auto stateHost = Impl::StateVector<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, Kokkos::subview(stateHostView, 0, Kokkos::ALL()));

        // set reference information
        stateHost.Temperature() = TREF;
        Kokkos::deep_copy(stateDevice, stateHostView);

        // size up the other scratch information
        real_type_2d_view perSpeciesDevice("scratch perSpecies device", 1, kineticsModelDataDevice->nSpec);
        real_type_1d_view mixtureDevice("scratch mixture device", 1);

        tChemLib::EnthalpyMass::runDeviceBatch(policy_enthalpy, stateDevice, perSpeciesDevice, mixtureDevice, *kineticsModelDataDevice);

        // copy to enthalpyReference
        Kokkos::deep_copy(enthalpyReference, Kokkos::subview(perSpeciesDevice, 0, Kokkos::ALL()));
    }

    // set the chemistry constraints
    constraints.Set(options);
}

std::shared_ptr<ablate::eos::TChem::FunctionContext> ablate::eos::TChem::BuildFunctionContext(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields,
                                                                                              bool checkDensityYi) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TChem requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });
    if (checkDensityYi) {
        if (densityYiField == fields.end()) {
            throw std::invalid_argument("The ablate::eos::TChem requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD Field");
        }
    }

    // determine the state vector size
    const ordinal_type stateVecDim = tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec);
    const ordinal_type batchSize = 1;

    // get the property string
    auto propertyName = std::string(eos::to_string(property));

    // set device information
    real_type_2d_view stateDevice(propertyName + " state device", batchSize, stateVecDim);
    real_type_2d_view perSpeciesDevice(propertyName + " perSpecies device", batchSize, kineticsModelDataDevice->nSpec);
    real_type_1d_view mixtureDevice(propertyName + " mixture device", batchSize);

    auto per_team_scratch_cp = tChemLib::Scratch<real_type_1d_view>::shmem_size(std::get<2>(thermodynamicFunctions.at(property))(kineticsModelDataDevice->nSpec));

    auto policy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(batchSize, Kokkos::AUTO());
    policy.set_scratch_size(1, Kokkos::PerTeam((int)per_team_scratch_cp));

    return std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                             .eulerOffset = eulerField->offset,
                                                             .densityYiOffset = checkDensityYi ? densityYiField->offset : -1,

                                                             // set device information
                                                             .stateDevice = stateDevice,
                                                             .perSpeciesDevice = perSpeciesDevice,
                                                             .mixtureDevice = mixtureDevice,

                                                             // copy host info
                                                             .stateHost = Kokkos::create_mirror_view(stateDevice),
                                                             .perSpeciesHost = Kokkos::create_mirror_view(perSpeciesDevice),
                                                             .mixtureHost = Kokkos::create_mirror_view(mixtureDevice),

                                                             // store the reference enthalpy
                                                             .enthalpyReference = enthalpyReference,

                                                             // policy
                                                             .policy = policy,

                                                             // kinetics data
                                                             .kineticsModelDataDevice = kineticsModelDataDevice});
}

ablate::eos::ThermodynamicFunction ablate::eos::TChem::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    return ThermodynamicFunction{.function = std::get<0>(thermodynamicFunctions.at(property)), .context = BuildFunctionContext(property, fields)};
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::TChem::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    return ThermodynamicTemperatureFunction{.function = std::get<1>(thermodynamicFunctions.at(property)), .context = BuildFunctionContext(property, fields)};
}

ablate::eos::TChem::ThermodynamicMassFractionFunction ablate::eos::TChem::GetThermodynamicMassFractionFunction(ablate::eos::ThermodynamicProperty property,
                                                                                                               const std::vector<domain::Field> &fields) const {
    return ThermodynamicMassFractionFunction{.function = std::get<0>(thermodynamicMassFractionFunctions.at(property)), .context = BuildFunctionContext(property, fields, false)};
}

ablate::eos::TChem::ThermodynamicTemperatureMassFractionFunction ablate::eos::TChem::GetThermodynamicTemperatureMassFractionFunction(ablate::eos::ThermodynamicProperty property,
                                                                                                                                     const std::vector<domain::Field> &fields) const {
    return ThermodynamicTemperatureMassFractionFunction{.function = std::get<1>(thermodynamicMassFractionFunctions.at(property)), .context = BuildFunctionContext(property, fields, false)};
}

void ablate::eos::TChem::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tmechFile: " << mechanismFile << std::endl;
    if (!thermoFile.empty()) {
        stream << "\tthermoFile: " << thermoFile << std::endl;
    }
    stream << "\tnumberSpecies: " << species.size() << std::endl;
    tChemLib::exec_space().print_configuration(stream, true);
    tChemLib::host_exec_space().print_configuration(stream, true);
}

PetscErrorCode ablate::eos::TChem::DensityFunction(const PetscReal *conserved, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::DensityTemperatureFunction(const PetscReal *conserved, PetscReal, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::DensityMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::DensityTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::TemperatureFunction(const PetscReal *conserved, PetscReal *property, void *ctx) { return TemperatureTemperatureFunction(conserved, 300, property, ctx); }
PetscErrorCode ablate::eos::TChem::TemperatureTemperatureFunction(const PetscReal *conserved, PetscReal temperatureGuess, PetscReal *temperature, void *ctx) {
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
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperatureGuess, conserved + functionContext->densityYiOffset, stateHost);
    functionContext->mixtureHost[0] = internalEnergyRef;
    Kokkos::deep_copy(functionContext->mixtureDevice, functionContext->mixtureHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    // compute the temperature
    ablate::eos::tChem::Temperature::runDeviceBatch(functionContext->policy,
                                                    functionContext->stateDevice,
                                                    functionContext->mixtureDevice,
                                                    functionContext->perSpeciesDevice,
                                                    functionContext->enthalpyReference,
                                                    *functionContext->kineticsModelDataDevice);

    // copy back the results
    Kokkos::deep_copy(functionContext->stateHost, functionContext->stateDevice);
    *temperature = stateHost.Temperature();

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::TemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *property, void *ctx) {
    return TemperatureTemperatureMassFractionFunction(conserved, yi, 300, property, ctx);
}
PetscErrorCode ablate::eos::TChem::TemperatureTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperatureGuess, PetscReal *temperature, void *ctx) {
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
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromMassFractions(density, temperatureGuess, yi, stateHost);
    functionContext->mixtureHost[0] = internalEnergyRef;
    Kokkos::deep_copy(functionContext->mixtureDevice, functionContext->mixtureHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    // compute the temperature
    ablate::eos::tChem::Temperature::runDeviceBatch(functionContext->policy,
                                                    functionContext->stateDevice,
                                                    functionContext->mixtureDevice,
                                                    functionContext->perSpeciesDevice,
                                                    functionContext->enthalpyReference,
                                                    *functionContext->kineticsModelDataDevice);

    // copy back the results
    Kokkos::deep_copy(functionContext->stateHost, functionContext->stateDevice);
    *temperature = stateHost.Temperature();

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *sensibleInternalEnergy, void *ctx) {
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

PetscErrorCode ablate::eos::TChem::InternalSensibleEnergyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *sensibleEnergyTemperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChem::SensibleInternalEnergy::runDeviceBatch(functionContext->policy,
                                                               functionContext->stateDevice,
                                                               functionContext->mixtureDevice,
                                                               functionContext->perSpeciesDevice,
                                                               functionContext->enthalpyReference,
                                                               *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *sensibleEnergyTemperature = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::InternalSensibleEnergyMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    InternalSensibleEnergyFunction(conserved, sensibleInternalEnergy, ctx);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::InternalSensibleEnergyTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *sensibleEnergyTemperature,
                                                                                         void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromMassFractions(density, temperature, yi, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChem::SensibleInternalEnergy::runDeviceBatch(functionContext->policy,
                                                               functionContext->stateDevice,
                                                               functionContext->mixtureDevice,
                                                               functionContext->perSpeciesDevice,
                                                               functionContext->enthalpyReference,
                                                               *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *sensibleEnergyTemperature = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::PressureFunction(const PetscReal *conserved, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;

    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = PressureTemperatureFunction(conserved, temperature, pressure, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::PressureTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    // Compute the internal energy from total ener
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    // compute the temperature
    ablate::eos::tChem::Pressure::runDeviceBatch(functionContext->policy, functionContext->stateDevice, *functionContext->kineticsModelDataDevice);

    // copy back the results
    Kokkos::deep_copy(functionContext->stateHost, functionContext->stateDevice);
    *pressure = stateHost.Pressure();

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::PressureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;

    PetscReal temperature;
    PetscErrorCode ierr = TemperatureMassFractionFunction(conserved, yi, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = PressureTemperatureMassFractionFunction(conserved, yi, temperature, pressure, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::PressureTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    // Compute the internal energy from total ener
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromMassFractions(density, temperature, yi, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    // compute the temperature
    ablate::eos::tChem::Pressure::runDeviceBatch(functionContext->policy, functionContext->stateDevice, *functionContext->kineticsModelDataDevice);

    // copy back the results
    Kokkos::deep_copy(functionContext->stateHost, functionContext->stateDevice);
    *pressure = stateHost.Pressure();

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SensibleEnthalpyTemperatureFunction(conserved, temperature, sensibleEnthalpy, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChem::SensibleEnthalpy::runDeviceBatch(functionContext->policy,
                                                         functionContext->stateDevice,
                                                         functionContext->mixtureDevice,
                                                         functionContext->perSpeciesDevice,
                                                         functionContext->enthalpyReference,
                                                         *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *sensibleEnthalpy = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SensibleEnthalpyMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureMassFractionFunction(conserved, yi, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SensibleEnthalpyTemperatureMassFractionFunction(conserved, yi, temperature, sensibleEnthalpy, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SensibleEnthalpyTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromMassFractions(density, temperature, yi, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChem::SensibleEnthalpy::runDeviceBatch(functionContext->policy,
                                                         functionContext->stateDevice,
                                                         functionContext->mixtureDevice,
                                                         functionContext->perSpeciesDevice,
                                                         functionContext->enthalpyReference,
                                                         *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *sensibleEnthalpy = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpeedOfSoundFunction(const PetscReal *conserved, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpeedOfSoundTemperatureFunction(conserved, temperature, speedOfSound, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpeedOfSoundTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChem::SpeedOfSound::runDeviceBatch(functionContext->policy, functionContext->stateDevice, functionContext->mixtureDevice, *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *speedOfSound = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpeedOfSoundMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureMassFractionFunction(conserved, yi, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpeedOfSoundTemperatureMassFractionFunction(conserved, yi, temperature, speedOfSound, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpeedOfSoundTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromMassFractions(density, temperature, yi, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChem::SpeedOfSound::runDeviceBatch(functionContext->policy, functionContext->stateDevice, functionContext->mixtureDevice, *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *speedOfSound = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpecificHeatConstantPressureFunction(const PetscReal *conserved, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpecificHeatConstantPressureTemperatureFunction(conserved, temperature, cp, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpecificHeatConstantPressureTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    tChemLib::SpecificHeatCapacityPerMass::runDeviceBatch(
        functionContext->policy, functionContext->stateDevice, functionContext->perSpeciesDevice, functionContext->mixtureDevice, *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *cp = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpecificHeatConstantPressureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureMassFractionFunction(conserved, yi, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpecificHeatConstantPressureTemperatureMassFractionFunction(conserved, yi, temperature, cp, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpecificHeatConstantPressureTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromMassFractions(density, temperature, yi, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    tChemLib::SpecificHeatCapacityPerMass::runDeviceBatch(
        functionContext->policy, functionContext->stateDevice, functionContext->perSpeciesDevice, functionContext->mixtureDevice, *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *cp = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpecificHeatConstantVolumeFunction(const PetscReal *conserved, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpecificHeatConstantVolumeTemperatureFunction(conserved, temperature, cv, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    tChemLib::SpecificHeatCapacityConsVolumePerMass::runDeviceBatch(functionContext->policy, functionContext->stateDevice, functionContext->mixtureDevice, *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *cv = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpecificHeatConstantVolumeMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureMassFractionFunction(conserved, yi, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpecificHeatConstantVolumeTemperatureMassFractionFunction(conserved, yi, temperature, cv, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpecificHeatConstantVolumeTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromMassFractions(density, temperature, yi, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    tChemLib::SpecificHeatCapacityConsVolumePerMass::runDeviceBatch(functionContext->policy, functionContext->stateDevice, functionContext->mixtureDevice, *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *cv = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpeciesSensibleEnthalpyTemperatureFunction(conserved, temperature, hi, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChem::SensibleEnthalpy::runDeviceBatch(functionContext->policy,
                                                         functionContext->stateDevice,
                                                         functionContext->mixtureDevice,
                                                         functionContext->perSpeciesDevice,
                                                         functionContext->enthalpyReference,
                                                         *functionContext->kineticsModelDataDevice);

    Kokkos::View<PetscReal *> hiHost(hi, functionContext->kineticsModelDataDevice->nSpec);
    Kokkos::deep_copy(hiHost, Kokkos::subview(functionContext->perSpeciesDevice, 0, Kokkos::ALL()));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::SpeciesSensibleEnthalpyMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureMassFractionFunction(conserved, yi, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpeciesSensibleEnthalpyTemperatureMassFractionFunction(conserved, yi, temperature, hi, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpeciesSensibleEnthalpyTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal *yi, PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Impl::StateVector<real_type_1d_view_host>(functionContext->kineticsModelDataDevice->nSpec, Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL()));
    FillWorkingVectorFromMassFractions(density, temperature, yi, stateHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChem::SensibleEnthalpy::runDeviceBatch(functionContext->policy,
                                                         functionContext->stateDevice,
                                                         functionContext->mixtureDevice,
                                                         functionContext->perSpeciesDevice,
                                                         functionContext->enthalpyReference,
                                                         *functionContext->kineticsModelDataDevice);

    Kokkos::View<PetscReal *> hiHost(hi, functionContext->kineticsModelDataDevice->nSpec);
    Kokkos::deep_copy(hiHost, Kokkos::subview(functionContext->perSpeciesDevice, 0, Kokkos::ALL()));
    PetscFunctionReturn(0);
}

void ablate::eos::TChem::FillWorkingVectorFromDensityMassFractions(double density, double temperature, const double *densityYi, const Impl::StateVector<real_type_1d_view_host> &stateVector) {
    stateVector.Temperature() = temperature;
    stateVector.Density() = density;
    stateVector.Pressure() = NAN;

    auto ys = stateVector.MassFractions();
    real_type yiSum = 0.0;
    for (ordinal_type s = 0; s < stateVector.NumSpecies() - 1; s++) {
        ys[s] = PetscMax(0.0, densityYi[s] / density);
        ys[s] = PetscMin(1.0, ys[s]);
        yiSum += ys[s];
    }
    if (yiSum > 1.0) {
        for (PetscInt s = 0; s < stateVector.NumSpecies() - 1; s++) {
            // Limit the bounds
            ys[s] /= yiSum;
        }
        ys[stateVector.NumSpecies() - 1] = 0.0;
    } else {
        ys[stateVector.NumSpecies() - 1] = 1.0 - yiSum;
    }
}

void ablate::eos::TChem::FillWorkingVectorFromMassFractions(double density, double temperature, const double *yi, const Impl::StateVector<real_type_1d_view_host> &stateVector) {
    stateVector.Temperature() = temperature;
    stateVector.Density() = density;
    stateVector.Pressure() = NAN;

    auto ys = stateVector.MassFractions();
    real_type yiSum = 0.0;
    for (ordinal_type s = 0; s < stateVector.NumSpecies() - 1; s++) {
        ys[s] = PetscMax(0.0, yi[s]);
        ys[s] = PetscMin(1.0, ys[s]);
        yiSum += ys[s];
    }
    if (yiSum > 1.0) {
        for (PetscInt s = 0; s < stateVector.NumSpecies() - 1; s++) {
            // Limit the bounds
            ys[s] /= yiSum;
        }
        ys[stateVector.NumSpecies() - 1] = 0.0;
    } else {
        ys[stateVector.NumSpecies() - 1] = 1.0 - yiSum;
    }
}

ablate::eos::FieldFunction ablate::eos::TChem::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2) const {
    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            // assume that the lambda are running on host
            using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
            using host_type = tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type::member_type;

            // create reusable data for the lambda (all on host)
            auto kineticsModelDataHost = tChemLib::createGasKineticModelConstData<host_device_type>(kineticsModel);
            real_type_1d_view_host stateHostView("state device", tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec));
            auto stateHost = Impl::StateVector<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, stateHostView);

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());
            auto per_team_extent = (int)tChem::SensibleInternalEnergy::getWorkSpaceSize(kineticsModelDataDevice->nSpec);
            policy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size(per_team_extent)));
            // store hi for the
            real_type_1d_view_host enthalpy("enthalpy", kineticsModelDataDevice->nSpec);

            auto tp = [=](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                Kokkos::parallel_for(
                    "tp init", policy, KOKKOS_LAMBDA(const host_type &member) {
                        // fill the state
                        stateHost.Temperature() = temperature;
                        stateHost.Pressure() = pressure;
                        auto yiHost = stateHost.MassFractions();
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i]; });

                        // compute the mw of the mix
                        const real_type mwMix = tChemLib::Impl::MolarWeights<real_type, host_device_type>::team_invoke(member, yiHost, kineticsModelDataHost);
                        member.team_barrier();

                        // compute r
                        double R = kineticsModelDataHost.Runiv / mwMix;

                        // compute pressure p = rho*R*T
                        PetscReal density = pressure / (temperature * R);

                        // compute the sensible energy
                        Scratch<real_type_1d_view_host> work(member.team_scratch(1), per_team_extent);
                        auto cpks = real_type_1d_view_host((real_type *)work.data(), per_team_extent);

                        auto sensibleInternalEnergy = ablate::eos::tChem::impl::SensibleInternalEnergyFcn<real_type, host_device_type>::team_invoke(
                            member, temperature, yiHost, enthalpy, cpks, enthalpyReference, kineticsModelDataHost);

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
            real_type_2d_view_host stateHostView("state device", 1, tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec));
            auto stateHost = Impl::StateVector<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, Kokkos::subview(stateHostView, 0, Kokkos::ALL()));

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());
            auto per_team_extent = (int)tChem::Temperature::getWorkSpaceSize(kineticsModelDataDevice->nSpec);
            policy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size((int)per_team_extent)));

            // store internal energy for the
            real_type_1d_view_host internalEnergy("internal energy", 1);
            real_type_2d_view_host enthalpy("enthalpy", 1, kineticsModelDataDevice->nSpec);
            real_type_1d_view_host mwMix("mwMix", 1);

            auto iep = [=](PetscReal sensibleInternalEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // fill most of the host
                stateHost.Temperature() = 300;  // init guess
                stateHost.Pressure() = pressure;
                internalEnergy(0) = sensibleInternalEnergy;
                auto yiHost = stateHost.MassFractions();
                Kokkos::parallel_for(
                    "tp init", policy, KOKKOS_LAMBDA(const host_type &member) {
                        // fill the state
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i]; });

                        // compute the mw of the mix
                        mwMix(0) = tChemLib::Impl::MolarWeights<real_type, host_device_type>::team_invoke(member, yiHost, kineticsModelDataHost);
                    });

                double R = kineticsModelDataHost.Runiv / mwMix(0);

                // compute the temperature
                eos::tChem::Temperature::runHostBatch(policy, stateHostView, internalEnergy, enthalpy, enthalpyReference, kineticsModelDataHost);

                // compute pressure p = rho*R*T
                PetscReal density = pressure / (stateHost.Temperature() * R);

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
            };
            if (property1 == ThermodynamicProperty::InternalSensibleEnergy) {
                return iep;
            } else {
                return [iep](PetscReal pressure, PetscReal sensibleInternalEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    iep(sensibleInternalEnergy, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::TChem.");
    } else if (finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            // assume that the lambda are running on host
            using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
            using host_type = tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type::member_type;

            // create reusable data for the lambda (all on host)
            auto kineticsModelDataHost = tChemLib::createGasKineticModelConstData<host_device_type>(kineticsModel);
            real_type_1d_view_host stateHostView("state device", tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec));
            auto stateHost = Impl::StateVector<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, stateHostView);

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());

            auto densityYiFromTP = [=](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                Kokkos::parallel_for(
                    "densityYi init from T & P", policy, KOKKOS_LAMBDA(const host_type &member) {
                        // fill the state
                        stateHost.Temperature() = temperature;
                        stateHost.Pressure() = pressure;
                        auto yiHost = stateHost.MassFractions();
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i]; });

                        // compute the mw of the mix
                        const real_type mwMix = tChemLib::Impl::MolarWeights<real_type, host_device_type>::team_invoke(member, yiHost, kineticsModelDataHost);
                        member.team_barrier();

                        // compute r
                        double R = kineticsModelDataHost.Runiv / mwMix;

                        // compute pressure p = rho*R*T
                        PetscReal density = pressure / (temperature * R);

                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec), [&](const ordinal_type &i) { conserved[i] = density * yi[i]; });
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
            real_type_2d_view_host stateHostView("state device", 1, tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec));
            auto stateHost = Impl::StateVector<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, Kokkos::subview(stateHostView, 0, Kokkos::ALL()));

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());
            auto per_team_extent = (int)tChem::Temperature::getWorkSpaceSize(kineticsModelDataDevice->nSpec);
            policy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size(per_team_extent)));

            // store internal energy for the
            real_type_1d_view_host internalEnergy("internal energy", 1);
            real_type_2d_view_host enthalpy("enthalpy", 1, kineticsModelDataDevice->nSpec);
            real_type_1d_view_host mwMix("mwMix", 1);

            auto densityYiFromIeP = [=](PetscReal sensibleInternalEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // fill most of the host
                stateHost.Temperature() = 300;  // init guess
                stateHost.Pressure() = pressure;
                internalEnergy(0) = sensibleInternalEnergy;
                auto yiHost = stateHost.MassFractions();
                Kokkos::parallel_for(
                    policy, KOKKOS_LAMBDA(const host_type &member) {
                        // fill the state
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i]; });

                        // compute the mw of the mix
                        mwMix(0) = tChemLib::Impl::MolarWeights<real_type, host_device_type>::team_invoke(member, yiHost, kineticsModelDataHost);
                    });

                double R = kineticsModelDataHost.Runiv / mwMix(0);

                // compute the temperature
                eos::tChem::Temperature::runHostBatch(policy, stateHostView, internalEnergy, enthalpy, enthalpyReference, kineticsModelDataHost);

                // compute pressure p = rho*R*T
                PetscReal density = pressure / (stateHost.Temperature() * R);

                Kokkos::parallel_for(
                    policy, KOKKOS_LAMBDA(const host_type &member) {
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec), [&](const ordinal_type &i) { conserved[i] = density * yi[i]; });
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

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::TChem.");
    } else {
        throw std::invalid_argument("Unknown field type " + field + " for ablate::eos::TChem.");
    }
}

std::map<std::string, double> ablate::eos::TChem::GetElementInformation() const {
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

std::map<std::string, std::map<std::string, int>> ablate::eos::TChem::GetSpeciesElementalInformation() const {
    // build the element names
    auto eNamesHost = kineticsModel.eNames_.view_host();
    std::vector<std::string> elementNames;
    for (ordinal_type i = 0; i < kineticsModel.nElem_; ++i) {
        elementNames.push_back(std::string(&eNamesHost(i, 0)));
    }

    std::map<std::string, std::map<std::string, int>> speciesElementInfo;

    // get the element info
    auto elemCountHost = kineticsModel.elemCount_.view_host();

    // march over each species
    for (std::size_t sp = 0; sp < species.size(); ++sp) {
        auto &speciesMap = speciesElementInfo[species[sp]];

        for (ordinal_type e = 0; e < kineticsModel.nElem_; ++e) {
            speciesMap[elementNames[e]] = elemCountHost(sp, e);
        }
    }

    return speciesElementInfo;
}

std::map<std::string, double> ablate::eos::TChem::GetSpeciesMolecularMass() const {
    // march over each species
    auto sMass = kineticsModel.sMass_.view_host();

    std::map<std::string, double> mw;
    for (std::size_t sp = 0; sp < species.size(); ++sp) {
        mw[species[sp]] = sMass((ordinal_type)sp);
    }

    return mw;
}

std::shared_ptr<ablate::eos::ChemistryModel::SourceCalculator> ablate::eos::TChem::CreateSourceCalculator(const std::vector<domain::Field> &fields, const ablate::solver::Range &cellRange) {
    return std::make_shared<ablate::eos::tChem::SourceCalculator>(fields, shared_from_this(), constraints, cellRange);
}

#include "registrar.hpp"
REGISTER(ablate::eos::ChemistryModel, ablate::eos::TChem, "[TChemV2](https://github.com/sandialabs/TChem) ideal gas eos",
         ARG(std::filesystem::path, "mechFile", "the mech file (CHEMKIN Format or Cantera Yaml)"), OPT(std::filesystem::path, "thermoFile", "the thermo file (CHEMKIN Format if mech file is CHEMKIN)"),
         OPT(ablate::monitors::logs::Log, "log", "An optional log for TChem echo output (only used with yaml input)"),
         OPT(ablate::parameters::Parameters, "options",
             "time stepping options (dtMin, dtMax, dtDefault, dtEstimateFactor, relToleranceTime, relToleranceTime, absToleranceTime, relToleranceNewton, absToleranceNewton, maxNumNewtonIterations, "
             "numTimeIterationsPerInterval, jacobianInterval, maxAttempts, thresholdTemperature)"));
