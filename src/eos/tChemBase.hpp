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
#include "eos/tChem/speedOfSound.hpp"
#include "eos/tChem/temperature.hpp"
#include "monitors/logs/log.hpp"
#include "parameters/parameters.hpp"
#include "utilities/intErrorChecker.hpp"
#include "utilities/vectorUtilities.hpp"

namespace ablate::eos {

namespace tChemLib = TChem;

class TChemBase : public ChemistryModel {
   protected:
    //! hold a copy of the constrains that can be used for single or batch source calculation
    tChem::SourceCalculator::ChemistryConstraints constraints;

    //! the mechanismFile may be chemkin or yaml based
    const std::filesystem::path mechanismFile;

    //! the thermoFile may be empty when using yaml input file
    const std::filesystem::path thermoFile;

    //! an optional log file for tchem echo redirection
    std::shared_ptr<ablate::monitors::logs::Log> log;

    /**
     * The kinetic model data
     */
    tChemLib::KineticModelData kineticsModel;

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
    real_type_1d_view enthalpyReference;

   public:
    /**
     * The tChem EOS can utilize either a mechanical & thermo file using the Chemkin file format for a modern yaml file.
     * @param mechFile
     * @param optionalThermoFile
     */
    explicit TChemBase(const std::string& eosName, std::filesystem::path mechanismFile, std::filesystem::path thermoFile = {}, std::shared_ptr<ablate::monitors::logs::Log> = {},
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
    real_type_1d_view GetEnthalpyOfFormation() { return enthalpyReference; };

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

   protected:
    struct FunctionContext {
        // memory access locations for fields
        PetscInt dim;
        PetscInt eulerOffset;
        PetscInt densityYiOffset;

        //! per species state
        real_type_2d_view stateDevice;
        //! per species array
        real_type_2d_view perSpeciesDevice;
        //! mass weighted mixture
        real_type_1d_view mixtureDevice;

        //! per species state
        real_type_2d_view stateHost;
        //! per species array
        real_type_2d_view perSpeciesHost;
        //! mass weighted mixture
        real_type_1d_view mixtureHost;

        //! store the enthalpyReferencePerSpecies
        real_type_1d_view enthalpyReference;

        //! the kokkos team policy for this function
        tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type policy;

        //! the kinetics data
        std::shared_ptr<tChemLib::KineticModelGasConstData<typename Tines::UseThisDevice<exec_space>::type>> kineticsModelDataDevice;
    };

   public:
    // Private static helper functions
    inline const static double TREF = 298.15;


};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_TCHEM_HPP
