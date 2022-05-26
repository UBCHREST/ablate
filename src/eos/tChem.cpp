#include "tChem.hpp"
#include <TChem_EnthalpyMass.hpp>
#include <utility>
#include "TChem_SpecificHeatCapacityConsVolumePerMass.hpp"
#include "TChem_SpecificHeatCapacityPerMass.hpp"
#include "eos/tChem/sensibleInternalEnergy.hpp"
#include "eos/tChem/speedOfSound.hpp"
#include "eos/tChem/temperature.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/kokkosUtilities.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::eos::TChem::TChem(std::filesystem::path mechanismFileIn, std::filesystem::path thermoFileIn) : EOS("TChem"), mechanismFile(std::move(mechanismFileIn)), thermoFile(std::move(thermoFileIn)) {
    // setup/use Kokkos
    ablate::utilities::KokkosUtilities::Initialize();

    // create/parse the kinetic data
    // TChem init reads/writes file it can only be done one at a time
    ablate::utilities::MpiUtilities::RoundRobin(PETSC_COMM_WORLD, [&](int rank) {
        if (thermoFile.empty()) {
            kineticsModel = tChemLib::KineticModelData(mechanismFile.string());
        } else {
            kineticsModel = tChemLib::KineticModelData(mechanismFile.string(), thermoFile.string());
        }
    });

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
        policy_enthalpy.set_scratch_size(1, Kokkos::PerTeam((int)per_team_scratch_h));

        // set the state
        real_type_2d_view stateDevice(" state device", 1, tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec));
        auto stateHostView = Kokkos::create_mirror_view(stateDevice);
        auto stateHost = Impl::StateVector<real_type_1d_view_host>(kineticsModelDataDevice->nSpec, Kokkos::subview(stateHostView, 0, Kokkos::ALL()));

        // set reference information
        stateHost.Temperature() = TREF;
        stateHost.MassFractions()() = 0.0;
        Kokkos::deep_copy(stateDevice, stateHostView);

        // size up the other scratch information
        real_type_2d_view perSpeciesDevice("scratch perSpecies device", 1, kineticsModelDataDevice->nSpec);
        real_type_1d_view mixtureDevice("scratch mixture device", 1);

        tChemLib::EnthalpyMass::runDeviceBatch(policy_enthalpy, stateDevice, perSpeciesDevice, mixtureDevice, *kineticsModelDataDevice);

        // copy to enthalpyReference
        Kokkos::deep_copy(enthalpyReference, Kokkos::subview(perSpeciesDevice, 0, Kokkos::ALL()));
    }
}

std::shared_ptr<ablate::eos::TChem::FunctionContext> ablate::eos::TChem::BuildFunctionContext(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
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
                                                             .densityYiOffset = densityYiField->offset,

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

void ablate::eos::TChem::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tmechFile: " << mechanismFile << std::endl;
    if (!thermoFile.empty()) {
        stream << "\tthermoFile: " << thermoFile << std::endl;
    }
    stream << "\tnumberSpecies: " << species.size() << std::endl;
    tChemLib::exec_space::print_configuration(stream, true);
    tChemLib::host_exec_space::print_configuration(stream, true);
}

PetscErrorCode ablate::eos::TChem::DensityFunction(const PetscReal *conserved, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::DensityTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *density, void *ctx) {
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

    Kokkos::View<PetscReal*> hiHost(hi,functionContext->kineticsModelDataDevice->nSpec);
    Kokkos::deep_copy(hiHost, Kokkos::subview(functionContext->perSpeciesDevice, 0, Kokkos::ALL()));
    PetscFunctionReturn(0);
}

void ablate::eos::TChem::FillWorkingVectorFromDensityMassFractions(double density, double temperature, const double *densityYi, const Impl::StateVector<real_type_1d_view_host>& stateVector) {
    stateVector.Temperature() = temperature;
    stateVector.Density() = density;
    stateVector.Pressure() = NAN;

    auto ys = stateVector.MassFractions();
    real_type yiSum = 0.0;
    for (ordinal_type s = 0; s < stateVector.NumSpecies(); s++) {
        ys[s] = PetscMax(0.0, densityYi[s] / density);
        ys[s] = PetscMin(1.0, ys[s]);
        yiSum += ys[s];
    }
    if (yiSum > 1.0) {
        for (PetscInt s = 0; s < stateVector.NumSpecies() - 1; s++) {
            // Limit the bounds
            ys[s] /= yiSum;
        }
        ys[stateVector.NumSpecies()] = 0.0;
    } else {
        ys[stateVector.NumSpecies()] = 1.0 - yiSum;
    }
}