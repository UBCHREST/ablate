#ifndef ABLATECLIENTTEMPLATE_TCHEM_HPP
#define ABLATECLIENTTEMPLATE_TCHEM_HPP

#include "eos.hpp"
#include <filesystem>
#include "utilities/intErrorChecker.hpp"

namespace ablate::eos {

class TChem : public EOS {
   private:
    // hold an error checker for the tchem outside library
    const utilities::IntErrorChecker errorChecker;

    // path to the input files
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;

    // prestore all species
    std::vector<std::string> species;
    int numberSpecies;

    // write/reproduce the periodic table
    static const char* periodicTable;
    inline static const char* periodicTableFileName = "periodictable.dat";

   public:
    TChem(std::filesystem::path mechFile, std::filesystem::path thermoFile );
    ~TChem() override;

    // general functions
    void View(std::ostream& stream) const override;

    // species model functions
    const std::vector<std::string>& GetSpecies() const override;

    // EOS functions
    decodeStateFunction GetDecodeStateFunction() override { return nullptr; }
    void* GetDecodeStateContext() override { return nullptr; }
    computeTemperatureFunction GetComputeTemperatureFunction() override { return nullptr; }
    void* GetComputeTemperatureContext() override { return nullptr; }

};

}
#endif  // ABLATECLIENTTEMPLATE_TCHEM_HPP
