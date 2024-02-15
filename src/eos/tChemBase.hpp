#ifndef ABLATELIBRARY_TCHEMBASE_HPP
#define ABLATELIBRARY_TCHEMBASE_HPP

#include <filesystem>
#include <map>
#include <memory>
#include "TChem_KineticModelData.hpp"
#include "chemistryModel.hpp"
#include "eos.hpp"
#include "eos/tChem/pressure.hpp"
#include "eos/tChem/sensibleEnthalpy.hpp"
#include "eos/tChem/sensibleInternalEnergy.hpp"
#include "eos/tChem/sourceCalculator.hpp"
#include "eos/tChem2/sourceCalculator2.hpp"
#include "eos/tChem/speedOfSound.hpp"
#include "eos/tChem/temperature.hpp"
#include "monitors/logs/log.hpp"
#include "parameters/parameters.hpp"
#include "utilities/intErrorChecker.hpp"
#include "utilities/vectorUtilities.hpp"
#include "zerork_cfd_plugin.h"

namespace ablate::eos {

namespace tChemLib = TChem;

class TChemBase : public ChemistryModel {
   protected:
    //! hold a copy of the constrains that can be used for single or batch source calculation
    tChem::SourceCalculator::ChemistryConstraints constraints;

    //! the mechanismFile may be chemkin or yaml based
    const std::filesystem::path mechanismFile;

//    const std::filesystem::path reactionFile;
//
//    const std::filesystem::path thermoFile;



    //! an optional log file for tchem echo redirection
    std::shared_ptr<ablate::monitors::logs::Log> log;

    /**
     * The kinetic model data
     */
    tChemLib::KineticModelData kineticsModel;

    /**
     * Store the primary kinetics data on the device
     */
    std::shared_ptr<tChemLib::KineticModelGasConstData<typename Tines::UseThisDevice<host_exec_space>::type>> kineticsModelDataHost;

    /**
     * Store the primary kinetics data on the device
     */
    std::shared_ptr<tChemLib::KineticModelGasConstData<typename Tines::UseThisDevice<exec_space>::type>> kineticsModelDataDevice;

    /**
     * keep the species names as
     */
    std::vector<std::string> species;

    /**
     * The reference enthalpy per species
     */
    real_type_1d_view enthalpyReferenceDevice;

    /**
     * The reference enthalpy per species
     */
    real_type_1d_view_host enthalpyReferenceHost;

   public:
    static const std::filesystem::path reactionFile;

    const std::filesystem::path thermoFile;
//    zerork_handle zrm_handle;
    /**
     * The tChem EOS can utilize either a mechanical & thermo file using the Chemkin file format for a modern yaml file.
     * @param mechFile
     * @param optionalThermoFile
     */
    explicit TChemBase(const std::string& eosName, std::filesystem::path mechanismFile, std::filesystem::path reactionFile, std::filesystem::path thermoFile, const std::shared_ptr<ablate::monitors::logs::Log>& = {},
                       const std::shared_ptr<ablate::parameters::Parameters>& options = {});



    /**
     * Returns all elements tracked in this mechanism and their molecular mass
     * @return
     */
    [[nodiscard]] virtual std::map<std::string, double> GetElementInformation() const = 0;

    /**
     * no. of atoms of each element in each species
     * @return
     */
    [[nodiscard]] virtual std::map<std::string, std::map<std::string, int>> GetSpeciesElementalInformation() const = 0;

    /**
     * the MW of each species
     * @return
     */
    [[nodiscard]] virtual std::map<std::string, double> GetSpeciesMolecularMass() const = 0;

    /**
     * Print the details of this eos
     * @param stream
     */
    void View(std::ostream& stream) const override;

    /**
     * return reference to kinetic data for other users
     */
    tChemLib::KineticModelData& GetKineticModelData() { return kineticsModel; }

    /**
     * Get the  reference enthalpy per species
     */
    real_type_1d_view GetEnthalpyOfFormation() { return enthalpyReferenceDevice; };

    /**
     * Species supported by this EOS
     * species model functions
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetSpeciesVariables() const override { return species; }

    /**
     * Returns a vector of all extra variables required to utilize the equation of state
     * @return
     */
    [[nodiscard]] virtual const std::vector<std::string>& GetProgressVariables() const override { return ablate::utilities::VectorUtilities::Empty<std::string>; }

    /**
     * Helper function to get the specific EnthalpyOfFormation
     * @param species
     * @return
     */
    inline double GetEnthalpyOfFormation(std::string_view speciesName) const {
        auto it = std::find_if(species.begin(), species.end(), [speciesName](const auto& component) { return component == speciesName; });
        // If element was found
        if (it != species.end()) {
            return enthalpyReferenceHost[std::distance(species.begin(), it)];
        } else {
            throw std::invalid_argument(std::string("Cannot locate species ") + std::string(speciesName) + " in EOS " + type);
        }
    }

   protected:
    /**
     * only allow modern input files
     */
    static const inline std::array<std::string, 2> validFileExtensions = {".yaml", ".yml"};

    struct FunctionContext {
        // memory access locations for fields
        PetscInt dim;
        PetscInt eulerOffset;
        PetscInt densityYiOffset;

        //! per species state
        real_type_2d_view_host stateHost;
        //! per species array
        real_type_2d_view_host perSpeciesHost;
        //! mass weighted mixture
        real_type_1d_view_host mixtureHost;

        //! store the enthalpyReferencePerSpecies
        real_type_1d_view_host enthalpyReferenceHost;

        //! the kokkos team policy for this function
        tChemLib::UseThisTeamPolicy<tChemLib::host_exec_space>::type policy;

        //! the kinetics data
        std::shared_ptr<tChemLib::KineticModelGasConstData<typename Tines::UseThisDevice<host_exec_space>::type>> kineticsModelDataHost;
    };

   public:
    // Private static helper functions
    inline const static double TREF = 298.15;
};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_TCHEMBASE_HPP
