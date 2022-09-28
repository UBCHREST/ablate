#ifndef ABLATELIBRARY_TCHEMSOOT_HPP
#define ABLATELIBRARY_TCHEMSOOT_HPP

#include <filesystem>
#include "eos/tChem.hpp"

namespace ablate::eos {

namespace tChemLib = TChem;

class TChemSoot : public TChem {
   public:
    /**
     * The tChem EOS can utzlie either a mechanical & thermo file using the Chemkin file format for a modern yaml file.
     * @param mechFile
     * @param optionalThermoFile
     */
    explicit TChemSoot(std::filesystem::path mechanismFile, std::filesystem::path thermoFile = {});

};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_TCHEM_HPP
