#include "tChemBase.hpp"
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

ablate::eos::TChemBase::TChemBase(const std::string &eosName, std::filesystem::path mechanismFileIn, std::filesystem::path thermoFileIn, std::shared_ptr<ablate::monitors::logs::Log> logIn,
                                  const std::shared_ptr<ablate::parameters::Parameters> &options)
    : ChemistryModel(eosName), mechanismFile(std::move(mechanismFileIn)), thermoFile(std::move(thermoFileIn)), log(logIn ? logIn : std::make_shared<ablate::monitors::logs::NullLog>()) {
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
        real_type_2d_view stateDevice("state device", 1, tChemLib::Impl::getStateVectorSize(kineticsModelDataDevice->nSpec));
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

void ablate::eos::TChemBase::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tmechFile: " << mechanismFile << std::endl;
    if (!thermoFile.empty()) {
        stream << "\tthermoFile: " << thermoFile << std::endl;
    }
    stream << "\tnumberSpecies: " << species.size() << std::endl;
    tChemLib::exec_space().print_configuration(stream, true);
    tChemLib::host_exec_space().print_configuration(stream, true);
}
