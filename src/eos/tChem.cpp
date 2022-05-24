#include "tChem.hpp"
#include "utilities/kokkosUtilities.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::eos::TChem::TChem(std::filesystem::path mechanismFileIn, std::filesystem::path thermoFileIn) : EOS("TChem"), mechanismFile(mechanismFileIn), thermoFile(thermoFileIn) {
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
    kineticsModelDataDevice = tChemLib::createGasKineticModelConstData<typename Tines::UseThisDevice<exec_space>::type>(kineticsModel);

    // copy the species information
    const auto speciesNamesHost = Kokkos::create_mirror_view(kineticsModelDataDevice.speciesNames);
    Kokkos::deep_copy(speciesNamesHost, kineticsModelDataDevice.speciesNames);
    // resize the species data
    species.resize(kineticsModelDataDevice.nSpec);
    auto speciesArray = species.data();

    Kokkos::parallel_for(
        "speciesInit", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(0, kineticsModelDataDevice.nSpec), KOKKOS_LAMBDA(const auto i) {
            speciesArray[i] = std::string(&speciesNamesHost(i, 0));
        });
    Kokkos::fence();
}

void ablate::eos::TChem::View(std::ostream& stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tmechFile: " << mechanismFile << std::endl;
    if (!thermoFile.empty()) {
        stream << "\tthermoFile: " << thermoFile << std::endl;
    }
    stream << "\tnumberSpecies: " << species.size() << std::endl;
    tChemLib::exec_space::print_configuration(stream, true);
    tChemLib::host_exec_space::print_configuration(stream, true);
}
