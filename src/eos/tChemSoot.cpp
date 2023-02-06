#include "tChemSoot.hpp"
#include <TChem_EnthalpyMass.hpp>
#include <utility>
#include "TChem_SpecificHeatCapacityConsVolumePerMass.hpp"
#include "TChem_SpecificHeatCapacityPerMass.hpp"
#include "eos/tChemSoot/densityFcn.hpp"
#include "eos/tChemSoot/sensibleInternalEnergy.hpp"
#include "eos/tChemSoot/sensibleInternalEnergyFcn.hpp"
#include "eos/tChemSoot/speedOfSound.hpp"
#include "eos/tChemSoot/temperature.hpp"
#include "eos/tChemSoot/specificHeatConstantPressure.hpp"
#include "eos/tChemSoot/specificHeatConstantVolume.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/kokkosUtilities.hpp"
#include "utilities/mpiUtilities.hpp"

//should be good
ablate::eos::TChemSoot::TChemSoot(std::filesystem::path mechanismFileIn, std::filesystem::path thermoFileIn) : EOS("TChemSoot"), mechanismFile(std::move(mechanismFileIn)), thermoFile(std::move(thermoFileIn)) {
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
    species.resize(kineticsModelDataDevice->nSpec+1);
    auto speciesArray = species.data();

    //Add Solid carbon species to the front
    speciesArray[0] = "C_solid";
    Kokkos::parallel_for(
        "speciesInit", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(0, kineticsModelDataDevice->nSpec), KOKKOS_LAMBDA(const auto i) {
            speciesArray[i+1] = std::string(&speciesNamesHost(i, 0));
        });
    Kokkos::fence();


    // compute the reference enthalpy
    enthalpyReference = real_type_1d_view("reference enthalpy", kineticsModelDataDevice->nSpec+1);
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
        Kokkos::deep_copy(Kokkos::subview(enthalpyReference,std::make_pair(1,kineticsModelDataDevice->nSpec+1)), Kokkos::subview(perSpeciesDevice, 0, Kokkos::ALL()));
        //Now put in reference enthalpy for Carbon
        enthalpyReference(0) = CarbonEnthalpy_R_T(TREF)*TREF*kineticsModelDataDevice->Runiv / MWCarbon; //TODO::Check that Runiv is correct units
    }
}

std::shared_ptr<ablate::eos::TChemSoot::FunctionContext> ablate::eos::TChemSoot::BuildFunctionContext(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    //STILL NEED TO THINK OF WHAT I WANT TO PASS TO EACH FUNCTION


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
    const ordinal_type stateVecDim = tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec)+1; //Extra 1 for YC
    const ordinal_type batchSize = 1;

    // get the property string
    auto propertyName = std::string(eos::to_string(property));

    // set device information
    real_type_2d_view stateDevice(propertyName + " state device", batchSize, stateVecDim);
    real_type_2d_view perSpeciesDevice(propertyName + " perSpecies device", batchSize, kineticsModelDataDevice->nSpec+1); //Include YCarbon_Solid as Part of the species array
    real_type_1d_view mixtureDevice(propertyName + " mixture device", batchSize);

    auto per_team_scratch_cp = tChemLib::Scratch<real_type_1d_view>::shmem_size(std::get<2>(thermodynamicFunctions.at(property))(kineticsModelDataDevice->nSpec));

    auto policy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(batchSize, Kokkos::AUTO());
    policy.set_scratch_size(1, Kokkos::PerTeam((int)per_team_scratch_cp));

    return std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2, //Number of physical dimensions
                                                             .eulerOffset = eulerField->offset, //Offset of data to eulerField
                                                             .densityYiOffset = densityYiField->offset, //Offset of data to density filed

                                                             // set device information
                                                             .stateDevice = stateDevice,
                                                             .perSpeciesDevice = perSpeciesDevice,
                                                             .mixtureDevice = mixtureDevice,

                                                             // copy host info
                                                             .stateHost = Kokkos::create_mirror_view(stateDevice),
                                                             .perSpeciesHost = Kokkos::create_mirror_view(perSpeciesDevice),
                                                             .mixtureHost = Kokkos::create_mirror_view(mixtureDevice),

                                                             // store the reference enthalpy
                                                             .enthalpyReference = enthalpyReference, //Full Reference enthalpy information

                                                             // policy
                                                             .policy = policy,

                                                             // kinetics data
                                                             .kineticsModelDataDevice = kineticsModelDataDevice});
}

//These Next 5 are the same as regular TCHEM
ablate::eos::ThermodynamicFunction ablate::eos::TChemSoot::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    return ThermodynamicFunction{.function = std::get<0>(thermodynamicFunctions.at(property)), .context = BuildFunctionContext(property, fields)};
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::TChemSoot::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    return ThermodynamicTemperatureFunction{.function = std::get<1>(thermodynamicFunctions.at(property)), .context = BuildFunctionContext(property, fields)};
}
void ablate::eos::TChemSoot::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tmechFile: " << mechanismFile << std::endl;
    if (!thermoFile.empty()) {
        stream << "\tthermoFile: " << thermoFile << std::endl;
    }
    stream << "\tnumberSpecies: " << species.size() << std::endl;
//    tChemLib::exec_space::print_configuration(stream, true);
//    tChemLib::host_exec_space::print_configuration(stream, true);
}

//Returns the Total Density of the Mixture, This is a conserved variable and can just be returned
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

//Returns the Temperature of the System, Returned from an iterative solution on the total sensible energy
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
    auto stateHost = Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL());
    FillWorkingVectorFromDensityMassFractions(density, temperatureGuess, conserved + functionContext->densityYiOffset, stateHost, functionContext->kineticsModelDataDevice->nSpec+1);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);
    functionContext->mixtureHost[0] = internalEnergyRef;
    Kokkos::deep_copy(functionContext->mixtureDevice, functionContext->mixtureHost);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    // compute the temperature
    ablate::eos::tChemSoot::Temperature::runDeviceBatch(functionContext->policy,
                                                    functionContext->stateDevice,
                                                    functionContext->mixtureDevice,
                                                    functionContext->perSpeciesDevice,
                                                    functionContext->enthalpyReference,
                                                    *functionContext->kineticsModelDataDevice);

    // copy back the results
    Kokkos::deep_copy(functionContext->stateHost, functionContext->stateDevice);
    *temperature = stateHost(2);

    PetscFunctionReturn(0);
}


//Calculate the Internal Sensible Energy From the current state, i.e Etot - KE
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
    auto stateHost = Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL());
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost, functionContext->kineticsModelDataDevice->nSpec+1);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChemSoot::SensibleInternalEnergy::runDeviceBatch(functionContext->policy,
                                                               functionContext->stateDevice,
                                                               functionContext->mixtureDevice,
                                                               functionContext->perSpeciesDevice,
                                                               functionContext->enthalpyReference,
                                                               *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
    *sensibleEnergyTemperature = functionContext->mixtureHost(0);

    PetscFunctionReturn(0);
}


//Grab Pressure from conserved variables (All Pressure should be good!)
PetscErrorCode ablate::eos::TChemSoot::PressureFunction(const PetscReal *conserved, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    //Compute Temperature
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    //Compute pressure with the temperature now known
    ierr = PressureTemperatureFunction(conserved, temperature, pressure, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
//Compute Pressure Assuming we know what the Temperature is
PetscErrorCode ablate::eos::TChemSoot::PressureTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL());
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost, functionContext->kineticsModelDataDevice->nSpec+1);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    // compute the Pressure With Temperature Stored in StateDevice
    ablate::eos::tChemSoot::Pressure::runDeviceBatch(functionContext->policy, functionContext->stateDevice, *functionContext->kineticsModelDataDevice);

    // copy back the results
    Kokkos::deep_copy(functionContext->stateHost, functionContext->stateDevice);
    *pressure = stateHost[1]; //1 is pressure state spot

    PetscFunctionReturn(0);
}



//Compute Sensible Enthalpy Without the temperature known, just the conservative values
PetscErrorCode ablate::eos::TChemSoot::SensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    //Compute the temperature
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    //Now we know the temperature! compute the sensible enthalpy
    ierr = SensibleEnthalpyTemperatureFunction(conserved, temperature, sensibleEnthalpy, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
//Compute it with the Temperature Known
PetscErrorCode ablate::eos::TChemSoot::SensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL());
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost, functionContext->kineticsModelDataDevice->nSpec+1);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChemSoot::SensibleEnthalpy::runDeviceBatch(functionContext->policy,
                                                         functionContext->stateDevice,
                                                         functionContext->mixtureDevice,
                                                         functionContext->perSpeciesDevice,
                                                         functionContext->enthalpyReference,
                                                         *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
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
    auto stateHost = Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL());
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost, functionContext->kineticsModelDataDevice->nSpec+1);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChemSoot::SpeedOfSound::runDeviceBatch(functionContext->policy, functionContext->stateDevice, functionContext->mixtureDevice, *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
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
    //The specific heat is a mass weighted term and thus can be broken up into a gaseos and solid contribution scaled by their relative mass fractions i.e (1-Yc) and (Yc)
    auto functionContext = (FunctionContext *)ctx;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    auto stateHost = Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL());
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost, functionContext->kineticsModelDataDevice->nSpec+1);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChemSoot::SpecificHeatConstantPressure::runDeviceBatch(functionContext->policy, functionContext->stateDevice, functionContext->mixtureDevice, *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
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
    auto stateHost = Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL());
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost, functionContext->kineticsModelDataDevice->nSpec+1);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChemSoot::SpecificHeatConstantVolume::runDeviceBatch(functionContext->policy, functionContext->stateDevice, functionContext->mixtureDevice, *functionContext->kineticsModelDataDevice);

    Kokkos::deep_copy(functionContext->mixtureHost, functionContext->mixtureDevice);
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
    auto stateHost = Kokkos::subview(functionContext->stateHost, 0, Kokkos::ALL());
    FillWorkingVectorFromDensityMassFractions(density, temperature, conserved + functionContext->densityYiOffset, stateHost, functionContext->kineticsModelDataDevice->nSpec+1);
    Kokkos::deep_copy(functionContext->stateDevice, functionContext->stateHost);

    ablate::eos::tChemSoot::SensibleEnthalpy::runDeviceBatch(functionContext->policy,
                                                         functionContext->stateDevice,
                                                         functionContext->mixtureDevice,
                                                         functionContext->perSpeciesDevice,
                                                         functionContext->enthalpyReference,
                                                         *functionContext->kineticsModelDataDevice);

    Kokkos::View<PetscReal *> hiHost(hi, functionContext->kineticsModelDataDevice->nSpec);
    Kokkos::deep_copy(hiHost, Kokkos::subview(functionContext->perSpeciesDevice, 0, Kokkos::ALL()));
    PetscFunctionReturn(0);
}



void ablate::eos::TChemSoot::FillWorkingVectorFromDensityMassFractions(double &density, double &temperature, const double *densityYi, const real_type_1d_view_host &totalStateVector,const int &totNumSpec) {
    //As a Reminder StateVector Assumed to follow -> {total Density, Pressure, Temperature, Total SpeciesMass Fraction of Gas states, Carbon Mass Fraction, Ndd}
    totalStateVector(2) = temperature;
    totalStateVector(0) = density;
    totalStateVector(1) = NAN; //Pressure set to NAN
    //Ignore the First species as it is the carbon solid species
    real_type yiSum = densityYi[0]/ density;//start with carbon value
    totalStateVector[totNumSpec+2] = yiSum;//carbon index is 3+kmcd_numspecies = 2+totNumSpecies
    for (ordinal_type s = 0; s < totNumSpec-2; s++) { // Dilute species is totNumSpec -1 in density Yi, and 3+totNumSpec-2 in totalState vector
        totalStateVector[3+s] = PetscMax(0.0, densityYi[s+1] / density);
        totalStateVector[3+s] = PetscMin(1.0, totalStateVector[3+s]);
        yiSum += totalStateVector[3+s];
    }
    if (yiSum > 1.0) {
        for (PetscInt s = 0; s < totNumSpec - 2; s++) {
            // Limit the bounds
            totalStateVector[3+s] /= yiSum;
        }
        totalStateVector[2+totNumSpec] /= yiSum; // have to do carbon out of the loop since it jumps the dilute last species in statevector
        totalStateVector[2+totNumSpec-1] = 0.0; //Set dilute species to 0
    } else {
        totalStateVector[2+totNumSpec-1] = 1.0 - yiSum; //Set dilute species to 1-YiSum
    }
}



ablate::eos::FieldFunction ablate::eos::TChemSoot::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2) const {
    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {

        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {

            // assume that the lambda are running on host
            using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
            using host_type = tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type::member_type;

            // create reusable data for the lambda (all on host)
            auto kineticsModelDataHost = tChemLib::createGasKineticModelConstData<host_device_type>(kineticsModel);
            real_type_1d_view_host stateHostView("state device", tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec)+1);//1 Extra for YCarbon

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());
            auto per_team_extent = (int)tChemSoot::SensibleInternalEnergy::getWorkSpaceSize(kineticsModelDataDevice->nSpec+1);
            policy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size(per_team_extent)));
            // store hi for the
            real_type_1d_view_host enthalpy("enthalpy", this->species.size());

            auto tp = [=](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                Kokkos::parallel_for(
                    "tp init", policy, KOKKOS_LAMBDA(const host_type &member) {
                        stateHostView(2) = temperature;
                        stateHostView(1) = pressure;
                        // fill the state
                        auto yiHost = Kokkos::subview( stateHostView,std::make_pair(3,3+kineticsModelDataDevice->nSpec) );
                        //It is assumed that the first species are the gas species, so we rearrange the values here so Carbon is in the last spot
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataDevice->nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i+1];});
                        auto Yc = yi[0];
                        //Make Sure to add the Carbon Value into YiHost
                        yiHost[kineticsModelDataDevice->nSpec] = Yc;

                        //Create a Gaseous State Vector
                        real_type_1d_view_host state_at_i_gas = real_type_1d_view_host("Gaseous",TChem::Impl::getStateVectorSize(kineticsModelDataHost.nSpec));
                        ablate::eos::TChemSoot::SplitYiState<host_device_type,real_type_1d_view_host> (stateHostView,state_at_i_gas,kineticsModelDataHost);
                        //Get the Gaseous State Vector
                        const Impl::StateVector<real_type_1d_view_host> sv_gas(kineticsModelDataHost.nSpec, state_at_i_gas);

                        // compute density
                        PetscReal density = ablate::eos::tChemSoot::impl::densityFcn<real_type, host_device_type>::team_invoke(member,sv_gas,Yc, kineticsModelDataHost);

                        // compute the sensible energy
                        Scratch<real_type_1d_view_host> work(member.team_scratch(1), per_team_extent);
                        auto cpks = real_type_1d_view_host((real_type *)work.data(), per_team_extent);

                        auto sensibleInternalEnergy = ablate::eos::tChemSoot::impl::SensibleInternalEnergyFcn<real_type, host_device_type>::team_invoke(
                            member, Yc, sv_gas, enthalpy, cpks, enthalpyReference, kineticsModelDataHost);

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
            real_type_2d_view_host stateHostView("state device", 1, tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec)+1); //1 extra for Ycarbon
            auto stateHost = Kokkos::subview(stateHostView, 0, Kokkos::ALL());

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());
            auto per_team_extent = (int)tChemSoot::SensibleInternalEnergy::getWorkSpaceSize(kineticsModelDataDevice->nSpec+1);
            policy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size(per_team_extent)));
            // store hi for the
            real_type_2d_view_host enthalpy("enthalpy", 1, kineticsModelDataDevice->nSpec);
            // store internal energy for the
            real_type_1d_view_host internalEnergy("internal energy", 1);

            auto iep = [=](PetscReal sensibleInternalEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                stateHost(2) = 300.; //Temperature Initial Guess
                stateHost(1) = pressure;
                internalEnergy(0) = sensibleInternalEnergy;
                auto Yc = yi[0];
                // fill the state
                auto yiHost = Kokkos::subview( stateHost,std::make_pair(3,3+kineticsModelDataDevice->nSpec) );
                Kokkos::parallel_for(
                    "tp init", policy, KOKKOS_LAMBDA(const host_type &member) {
                       // fill the state
                       Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataDevice->nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i+1]; });
                });
                //Make Sure to add the Carbon Value into YiHost
                yiHost[kineticsModelDataDevice->nSpec] = Yc;
                //Create a Gaseous State Vector
                real_type_1d_view_host state_at_i_gas = real_type_1d_view_host("Gaseous",TChem::Impl::getStateVectorSize(kineticsModelDataHost.nSpec));
                ablate::eos::TChemSoot::SplitYiState<host_device_type,real_type_1d_view_host> (stateHost,state_at_i_gas,kineticsModelDataHost);
                //Get the Gaseous State Vector
                const Impl::StateVector<real_type_1d_view_host> sv_gas(kineticsModelDataHost.nSpec, state_at_i_gas);

                // compute the temperature
                eos::tChemSoot::Temperature::runHostBatch(policy, stateHostView, internalEnergy, enthalpy, enthalpyReference, kineticsModelDataHost);

                //Compute the Density
                Kokkos::parallel_for(
                    "tp Density Calc", policy, KOKKOS_LAMBDA(const host_type &member) {
                        PetscReal density = ablate::eos::tChemSoot::impl::densityFcn<real_type, host_device_type>::team_invoke(member, sv_gas, Yc, kineticsModelDataHost);
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
    }
    else if (finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD == field) {

        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {

            // assume that the lambda are running on host
            using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
            using host_type = tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type::member_type;

            // create reusable data for the lambda (all on host)
            auto kineticsModelDataHost = tChemLib::createGasKineticModelConstData<host_device_type>(kineticsModel);
            real_type_1d_view_host stateHostView("state device", tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec)+1);//1 Extra for YCarbon

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());

            auto densityYiFromTP = [=](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                Kokkos::parallel_for(
                    "densityYi init from T & P", policy, KOKKOS_LAMBDA(const host_type &member) {
                        // fill the state
                        stateHostView(2) = temperature;
                        stateHostView(1) = pressure;
                        auto yiHost = Kokkos::subview( stateHostView,std::make_pair(3,3+kineticsModelDataDevice->nSpec) );
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i+1]; });
                        auto Yc = yi[0];
                        //Make Sure to add the Carbon Value into YiHost
                        yiHost[kineticsModelDataDevice->nSpec] = Yc;

                        //Create a Gaseous State Vector
                        real_type_1d_view_host state_at_i_gas = real_type_1d_view_host("Gaseous",TChem::Impl::getStateVectorSize(kineticsModelDataHost.nSpec));
                        ablate::eos::TChemSoot::SplitYiState<host_device_type,real_type_1d_view_host> (stateHostView,state_at_i_gas,kineticsModelDataHost);
                        //Get the Gaseous State Vector
                        const Impl::StateVector<real_type_1d_view_host> sv_gas(kineticsModelDataHost.nSpec, state_at_i_gas);

                        // compute density
                        PetscReal density = ablate::eos::tChemSoot::impl::densityFcn<real_type, host_device_type>::team_invoke(member,sv_gas,Yc, kineticsModelDataHost);
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec+1), [&](const ordinal_type &i) { conserved[i] = density * yi[i]; });
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
            real_type_2d_view_host stateHostView("state device", 1, tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec)+1); //1 extra for Ycarbon
            auto stateHost = Kokkos::subview(stateHostView, 0, Kokkos::ALL());

            // prepare to compute SensibleInternalEnergy
            typename tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy(1, Kokkos::AUTO());
            auto per_team_extent = (int)tChemSoot::SensibleInternalEnergy::getWorkSpaceSize(kineticsModelDataDevice->nSpec+1);
            policy.set_scratch_size(1, Kokkos::PerTeam((int)tChemLib::Scratch<real_type_1d_view>::shmem_size(per_team_extent)));
            // store hi for the
            real_type_2d_view_host enthalpy("enthalpy", 1, kineticsModelDataDevice->nSpec);
            // store internal energy for the
            real_type_1d_view_host internalEnergy("internal energy", 1);

            auto densityYiFromIeP = [=](PetscReal sensibleInternalEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                stateHost(2) = 300.; //Temperature Initial Guess
                stateHost(1) = pressure;
                internalEnergy(0) = sensibleInternalEnergy;
                auto Yc = yi[0];
                // fill the state
                auto yiHost = Kokkos::subview( stateHost,std::make_pair(3,3+kineticsModelDataDevice->nSpec) );
                Kokkos::parallel_for(
                    "tp init", policy, KOKKOS_LAMBDA(const host_type &member) {
                        // fill the state
                        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataDevice->nSpec), [&](const ordinal_type &i) { yiHost[i] = yi[i+1]; });
                    });
                //Make Sure to add the Carbon Value into YiHost
                yiHost[kineticsModelDataDevice->nSpec] = Yc;
                //Create a Gaseous State Vector
                real_type_1d_view_host state_at_i_gas = real_type_1d_view_host("Gaseous",TChem::Impl::getStateVectorSize(kineticsModelDataHost.nSpec));
                ablate::eos::TChemSoot::SplitYiState<host_device_type,real_type_1d_view_host> (stateHost,state_at_i_gas,kineticsModelDataHost);
                //Get the Gaseous State Vector
                const Impl::StateVector<real_type_1d_view_host> sv_gas(kineticsModelDataHost.nSpec, state_at_i_gas);

                // compute the temperature
                eos::tChemSoot::Temperature::runHostBatch(policy, stateHostView, internalEnergy, enthalpy, enthalpyReference, kineticsModelDataHost);

                //Compute the Density
                Kokkos::parallel_for(
                    "tp Density Calc", policy, KOKKOS_LAMBDA(const host_type &member) {
                    PetscReal density = ablate::eos::tChemSoot::impl::densityFcn<real_type, host_device_type>::team_invoke(member, sv_gas, Yc, kineticsModelDataHost);
                    Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kineticsModelDataHost.nSpec+1), [&](const ordinal_type &i) { conserved[i] = density * yi[i]; });
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


//should be good
std::map<std::string, std::map<std::string, int>> ablate::eos::TChemSoot::GetSpeciesElementalInformation() const {
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
    for (std::size_t sp = 1; sp < species.size(); ++sp) {
        auto &speciesMap = speciesElementInfo[species[sp]];

        for (ordinal_type e = 0; e < kineticsModel.nElem_; ++e) {
            speciesMap[elementNames[e]] = elemCountHost(sp-1, e);
        }
    }
    //Do Carbon by itself
    auto &speciesMap = speciesElementInfo[species[0]];
    for (ordinal_type e = 0; e < kineticsModel.nElem_; ++e) {
        if(elementNames[e] == "C")
            speciesMap["C"] = 1;
        else speciesMap[elementNames[e]] = 0;
    }
    return speciesElementInfo;
}

//Should be good
std::map<std::string, double> ablate::eos::TChemSoot::GetSpeciesMolecularMass() const {
    // march over each species
    auto sMass = kineticsModel.sMass_.view_host();

    std::map<std::string, double> mw;
    for (std::size_t sp = 1; sp < species.size(); ++sp) {
        mw[species[sp]] = sMass((ordinal_type)(sp-1) );
    }
    mw[0] = MWCarbon;

    return mw;
}

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::TChemSoot, "[TChemV2](https://github.com/sandialabs/TChem) ideal gas eos Agumented with a soot formation mechanism", ARG(std::filesystem::path, "mechFile", "the mech file (CHEMKIN Format or Cantera Yaml)"),
         OPT(std::filesystem::path, "thermoFile", "the thermo file (CHEMKIN Format if mech file is CHEMKIN)"));