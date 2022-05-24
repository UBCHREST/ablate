#ifndef ABLATELIBRARY_TCHEM_HPP
#define ABLATELIBRARY_TCHEM_HPP

#include <filesystem>
#include <map>
#include <memory>
#include "TChem_KineticModelData.hpp"
#include "eos.hpp"
#include "utilities/intErrorChecker.hpp"

namespace ablate::eos {

namespace tChemLib = TChem;

class TChem : public EOS {
   private:
    //! the mechanismFile may be chemkin or yaml based
    const std::filesystem::path mechanismFile;

    //! the thermoFile may be empty when using yaml input file
    const std::filesystem::path thermoFile;

    /**
     * The kinetic model data
     */
    tChemLib::KineticModelData kineticsModel;

    /**
     * Store the primary kinetics data on the device
     */
    tChemLib::KineticModelGasConstData<typename Tines::UseThisDevice<exec_space>::type> kineticsModelDataDevice;

    /**
     * keep the species names as
     */
    std::vector<std::string> species;

   public:
    /**
     * The tChem EOS can utzlie either a mechanical & thermo file using the Chemkin file format for a modern yaml file.
     * @param mechFile
     * @param optionalThermoFile
     */
    explicit TChem(std::filesystem::path mechanismFile, std::filesystem::path thermoFile = {});

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override { return {}; }

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override { return {}; }

    /**
     * Single function to produce fieldFunction function for any two properties, velocity, and species mass fractions.  These calls can be slower and should be used for init/output only
     * @param field
     * @param property1
     * @param property2
     */
    [[nodiscard]] FieldFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2) const override { return {}; }

    /**
     * Species supported by this EOS
     * species model functions
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetSpecies() const override { return species; }

    /**
     * Print the details of this eos
     * @param stream
     */
    void View(std::ostream& stream) const override;
};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_TCHEM_HPP
